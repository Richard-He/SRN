import os.path as osp
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
import torch
import pickle
import argparse
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, CitationFull, Reddit, Yelp, Flickr
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
# from torch_geometric.nn import SplineConv
from layers import GEN, GAT, SAGE, GCN, MLP
# from torch_geometric.data import RandomNodeSampler
from loguru import logger
import numpy as np
from utils import Pruner, find_rate
# from utils import Pruner

parser = argparse.ArgumentParser(description='Greedy_SRM_old')
parser.add_argument('--runs',type=int, default=10)
parser.add_argument('--gnn', type=str, default='MLP')
parser.add_argument('--reset',type=lambda x: (str(x).lower() == 'true'), default=False)

parser.add_argument('--dataset',type=str, default='Pubmed')

parser.add_argument('--ratio', type=float, default=0.1)
parser.add_argument('--layers', type=int, default=3)
parser.add_argument('--epochs', type=int, default=800)
parser.add_argument('--early', type=int, default=30)
parser.add_argument('--wd', type=float,default=0)
parser.add_argument('--dropout', type=float, default=0)
# parser.add_argument('--prate',type=float,default=0.5)
parser.add_argument('--pratio', type=float, default=0.8)
parser.add_argument('--style', type=str, default='random')
args = parser.parse_args()



gnn = args.gnn
gnndict = {'GAT': GAT, 'SAGE': SAGE, 'GCN': GCN, 'GEN': GEN, 'MLP': MLP}
reset = args.reset
ratio = args.ratio
dataset_n = args.dataset
t_layers = args.layers
log_name = f'./result/Full_Batch_GNN_{gnn}_style_{args.style}_dropout_{args.dropout}_weight_decay_{args.wd}_dataset_{dataset_n}'
path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset_n)
if dataset_n == 'arxiv':
    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=T.ToSparseTensor())
    data = dataset[0]
    row, col, val = data.adj_t.coo()
    N = int(row.max()+1)
    row = torch.cat([torch.arange(0, N), row],dim=0)
    col = torch.cat([torch.arange(0, N), col],dim=0)
    edge_index = torch.cat([row,col]).view(2, -1)
    data.edge_index = edge_index
    split_idx = dataset.get_idx_split()
    data.train_mask = torch.zeros(data.num_nodes)
    data.train_mask[split_idx['train']]=1
    data.train_mask = data.train_mask.bool()
    data.test_mask = ~ (data.train_mask)
    data.y = data.y.squeeze()

else:
    if dataset_n == 'dblp':
        dataset = CitationFull(path, dataset_n)
    else:
        dataset = Planetoid(path, dataset_n)
    data = dataset[0]
    train_split = pickle.load(open(f'./datasetsplit/{dataset_n.lower()}_train', "rb") )
    test_split = pickle.load(open(f'./datasetsplit/{dataset_n.lower()}_train', "rb") )
    rand = torch.cat([train_split, test_split])
    thold = int(data.num_nodes * ratio)
    train_split = rand[:thold]
    test_split = rand[thold:]
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[train_split] = 1
    data.val_mask = None
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask[test_split] = 1

    

criteria = CrossEntropyLoss()

out_channels = (torch.max(data.y) + 1).item()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = gnndict[gnn](in_channels=data.x.size(-1), hidden_channels=64, num_layers=t_layers, dropout=args.dropout, out_channels=out_channels).to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=args.wd)
logger.add(log_name)

def train(pruner):
    model.train()
    optimizer.zero_grad()
    mask = pruner.prune(data.edge_index)
    edge_index = data.edge_index[:,mask]
    out = model(data.x, edge_index)[data.train_mask]
    criteria(out ,data.y[data.train_mask]).backward()
    optimizer.step()

@torch.no_grad()
def test():
    model.eval()
    log_probs, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = log_probs[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

best_val = 0
val_list = []
std_list = []
ratio_list = []
pruner = Pruner(style=args.style)
for j in torch.linspace(0.05, 1, 20):
    print('j',j)
    r_list = []
    pruner.set_rate(j)
    for i in range(args.runs):
        model.reset_parameters()
        best_val = 0
        best_val_epoch = 0
        for epoch in range(1, args.epochs+1):
            train(pruner=pruner)
            train_acc, test_acc = test()
            if test_acc > best_val :
                best_val = test_acc
                best_val_epoch = epoch
                #logger.info(f'num_layers:{layers}, epochs: {epoch}, train: {train_acc:.4f}, new_best_val: {test_acc:.4f}')
            # log = 'Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}'
            # print(log.format(epoch, train_acc, test_acc))
            if epoch - best_val_epoch > args.early:
                r_list.append(best_val)
                break
    logger.info(f'ratio:{j:.4f} test_acc: {np.mean(r_list):.4f}, std: {np.std(r_list):.4f}')
    ratio_list = ratio_list + [j.item()] * len(r_list)
    val_list = val_list + r_list
logger.info(f'ratio_list = {np.array(ratio_list)}, val_list = {np.array(val_list)}')