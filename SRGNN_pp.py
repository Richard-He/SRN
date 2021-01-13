import torch
import argparse
import torch.nn.functional as F
import torch.nn as nn
from torch_scatter import scatter
from ogb.nodeproppred import PygNodePropPredDataset
from evaluate import Evaluator
from layers import GAT, SAGE, GCN, GEN, MLP
from torch_geometric.nn import GENConv, DeepGCNLayer
from sample import RandomNodeSampler
from torch_geometric.datasets import Reddit, Yelp, Flickr
from loguru import logger
from copy import deepcopy
import pandas as pd
from utils import Pruner
import numpy as np

# args = parser.parse_args()
parser = argparse.ArgumentParser(description='AdaGNN')
parser.add_argument('--gnn', type=str, default='GCN')
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--dataset',type=str, default='protein')
parser.add_argument('--reset',type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--early', type=int, default=50)
parser.add_argument('--pstyle', type=str, default='bbp')
parser.add_argument('--num_test_parts', type=int, default=5)
parser.add_argument('--num_train_parts', type=int, default=40)
parser.add_argument('--style', type=str, default='random')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--wd', type=float,default=5e-3)
parser.add_argument('--layers', type=int, default=3)
parser.add_argument('--start', type=float, default=0.05)
parser.add_argument('--end', type=float, default=1)
args = parser.parse_args()

gnn = args.gnn
data_n = args.dataset
layers = args.layers
epochs = args.epochs
reset = args.reset
metrics = 'f1' if data_n == 'protein' else 'acc'
gnndict = {'GAT': GAT, 'SAGE': SAGE, 'GCN': GCN, 'GEN': GEN, 'MLP':MLP}

# Set split indices to masks.
if data_n == 'protein':
    dataset = PygNodePropPredDataset('ogbn-proteins', root='../data')
elif data_n == 'product':
    dataset = PygNodePropPredDataset('ogbn-products', root='../data/products')
splitted_idx = dataset.get_idx_split()
data = dataset[0]
data.n_id = torch.arange(data.num_nodes)
#data.node_species = None
if data_n == 'protein':
    row, col = data.edge_index
    data.y = data.y.to(torch.float)
    data.x = scatter(data.edge_attr, col, 0, dim_size=data.num_nodes, reduce='add')
elif data_n == 'product':
    data.y = data.y.squeeze()
log_name = f'./result/SRGNN_{gnn}_dataset_{data_n}_dropout_{args.dropout}_wd_{args.wd}_style_{args.style}_type_{args.pstyle}'
# log_name = f'logs_version{version}_{times}'
# Initialize features of nodes by aggregating edge features.
row, col = data.edge_index
    
# Set split indices to masks.
for split in ['train', 'valid', 'test']:
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[splitted_idx[split]] = True
    data[f'{split}_mask'] = mask
data['test_mask'] = data['valid_mask'] | data['test_mask']


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if data_n == 'protein':
    model = gnndict[gnn](in_channels=data.x.size(-1), hidden_channels=64, out_channels=data.y.size(-1), num_layers=layers, dropout=args.dropout).to(device)
elif data_n == 'product':
    model = gnndict[gnn](in_channels=data.x.size(-1), hidden_channels=64, out_channels=data.y.max().int().item()+1, num_layers=layers, dropout=args.dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=args.wd)
# criterion = torch.nn.BCEWithLogitsLoss()
if data_n == 'product':
    evaluator = Evaluator(data_n, t_ype=2)
else:
    evaluator = Evaluator(data_n)

logger.add(log_name)
logger.info('logname: {}'.format(log_name))
criterion = nn.BCEWithLogitsLoss() if data_n == 'protein' else nn.CrossEntropyLoss()

pruner = Pruner(style=args.style)

train_loader = RandomNodeSampler(data,pruner=pruner, num_parts=args.num_train_parts, shuffle=True,
                                 num_workers=5)
test_loader = RandomNodeSampler(data, num_parts=args.num_test_parts, num_workers=5)

def train():
    model.train()

    total_loss = total_examples = 0
    
    if args.pstyle == 'pbb':
        train_loader.prune()

    for data in train_loader:
        optimizer.zero_grad()
        if args.pstyle == 'bbp':
            mask = pruner.prune(data.edge_index).to(device)
            data = data.to(device)
            out = model(data.x, data.edge_index[:,mask], data.edge_attr[mask])
        else:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())


    return total_loss / total_examples


@torch.no_grad()
def test():
    model.eval()
    total_loss_train = total_loss_test = total_examples_train = total_examples_test = 0

    y_true = {'train': [], 'valid': [], 'test': []}
    y_pred = {'train': [], 'valid': [], 'test': []}

    for data in test_loader:
        data = data.to(device)
        if data_n == 'protein':
            out = model(data.x, data.edge_index, data.edge_attr)
        elif data_n == 'product':
            out = model(data.x, data.edge_index)
        total_loss_train += float(criterion(out[data.train_mask], data.y[data.train_mask]))* int(data.train_mask.sum())
        total_examples_train += int(data.train_mask.sum())
        total_loss_test += float(criterion(out[data.test_mask], data.y[data.test_mask]))* int(data.test_mask.sum())
        total_examples_test += int(data.test_mask.sum())
        
        if data_n == 'protein':
            for split in y_true.keys():
                mask = data[f'{split}_mask']
                y_true[split].append(data.y[mask].cpu())
                y_pred[split].append(out[mask].cpu())
            

        elif data_n == 'product':
            vals, pred_lb = torch.max(out, 1)
            for split in y_true.keys():
                mask = data[f'{split}_mask']
                y_true[split].append(data.y[mask].cpu())
                y_pred[split].append(pred_lb[mask].cpu())


    train_m = evaluator.eval({
        'y_true': torch.cat(y_true['train'], dim=0),
        'y_pred': torch.cat(y_pred['train'], dim=0),
    }, metrics)

    valid_m = evaluator.eval({
        'y_true': torch.cat(y_true['valid'], dim=0),
        'y_pred': torch.cat(y_pred['valid'], dim=0),
    }, metrics)

    test_m = evaluator.eval({
        'y_true': torch.cat(y_true['test'], dim=0),
        'y_pred': torch.cat(y_pred['test'], dim=0),
    }, metrics)

    return train_m, valid_m, test_m, total_loss_train / total_examples_train, total_loss_test / total_examples_test
# evaluator.num_tasks = data.y.size(1)
# evaluator.eval_metric = 'acc'
# num_layers = torch.arange(1, num_layers+1, args.intval)
# train_list = []
# val_list = []
# test_list = []
# layer_list = []
# epoch_list = []
best_val = 0
val_list = []
train_list = []
std_list = []
ratio_list = []
train_loss_l =[]
test_loss_l = []
for j in torch.linspace(args.start, args.end, int((args.end-args.start)/0.05) + 1):
    r_list = []
    tr_list = []
    train_l_l =[]
    test_l_l = []
    pruner.set_rate(j)
    for i in range(args.runs):
        model.reset_parameters()
        best_val = 0
        best_train = 0
        best_val_epoch = 0
        bst_train_loss = 0
        bst_test_loss = 0
        for epoch in range(1, args.epochs+1):
            train()
            train_acc, _, test_acc, train_loss, test_loss = test()
            if test_acc > best_val :
                best_val = test_acc
                best_val_epoch = epoch
                best_train = train_acc
                bst_train_loss = train_loss
                bst_test_loss = test_loss
            if epoch - best_val_epoch > args.early:
                r_list.append(best_val)
                tr_list.append(best_train)
                train_l_l.append(bst_train_loss)
                test_l_l.append(bst_test_loss)
                break
    logger.info(f'ratio:{j:.4f} test_acc: {np.mean(r_list):.4f}, std: {np.std(r_list):.4f}')
    ratio_list = ratio_list + [j.item()] * len(r_list)
    val_list = val_list + r_list
    train_list = train_list + tr_list
    train_loss_l = train_loss_l + train_l_l
    test_loss_l = test_loss_l + test_l_l
logger.info(f'ratio_list = {np.array(ratio_list)}, train_list = {np.array(train_list)}, val_list = {np.array(val_list)}, train_loss = {np.array(train_loss_l)},'
            f'test_loss = {np.array(test_loss_l)}')

# for layers in num_layers:
#     # best_train = 0
#     best_val = 10
#     best_val_epoch = 0
#     model.unfix(layers)
#     lo_tr = []
#     lo_te = []
#     for runs in range(args.runs):
#         if reset == True:
#             model.reset_parameters()
#         for epoch in range(1, args.epochs+1):
#             loss = train(epoch)
#             train_rocauc, valid_rocauc, test_rocauc, tr_loss, te_loss = test()
#             if te_loss < best_val:
#                 best_val = te_loss
#                 best_val_epoch = epoch
#                 # logger.info(f'num_layers: {layers}, epochs: {epoch},train: {train_rocauc:.4f} new best valid: {best_val:.4f}, test: {test_rocauc:.4f}')
#                 # train_list.append(train_rocauc)
#                 # val_list.append(valid_rocauc)
#                 # test_list.append(test_rocauc)
#                 # layer_list.append(layers)
#                 # epoch_list.append(epoch)
#                 print(f'Loss: {loss:.4f}, Train: {tr_loss:.4f}, '
#                     f' Test: {te_loss:.4f}')
#             if epoch - best_val_epoch > args.early or epoch == args.epochs:
#                 # losses.append(loss)
#                 lr.append(layers)
#                 ep.append(epoch)
#                 lo_tr.append(tr_loss)
#                 lo_te.append(te_loss)
#                 tr_losses.append(tr_loss)
#                 te_losses.append(te_loss)
#                 break
#     logger.info(f'num_layers: {layers}, train: {np.mean(lo_tr):.4f}, std: {np.std(lo_tr):.4f}, test: {np.mean(lo_te):.4f}, std: {np.std(lo_te):.4f}')

# t_dic = {'layers': lr, 'epochs': ep, 'train_loss':tr_losses, 'test_loss':te_losses}
# df = pd.DataFrame.from_dict(t_dic)
# df.to_pickle(log_name+'_save_')
# df = pd.DataFrame({'layers':layer_list, 'epochs':epoch_list, 'train':train_list, 'val': val_list, 'test': test_list})
# df.to_pickle('vanilla_dgcn')