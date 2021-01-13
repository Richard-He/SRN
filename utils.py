import torch
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_scatter import scatter
from inspect import getargspec
import logging
import math

def _filter(data, node_idx):
    """
    presumably data_n_id and new_n_id are sorted

    """
    new_data = Data()
    N = data.num_nodes
    E = data.edge_index.size(1)
    adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                       value=data.edge_attr, sparse_sizes=(N, N))
    new_adj, edge_idx = adj.saint_subgraph(node_idx)
    row, col, value = new_adj.coo()
    
    for key, item in data:
        if item.size(0) == N:
            new_data[key] = item[node_idx]
        elif item.size(0) == E:
            new_data[key] = item[edge_idx]
        else:
            new_data[key] = item
    
    new_data.edge_index = torch.stack([row, col], dim=0)
    new_data.num_nodes = len(node_idx)
    new_data.edge_attr = value
    return new_data

def find_rate(edge_index):
    E = edge_index.size(1)
    src = torch.ones(E).to(edge_index.device)
    deg_hist = scatter(src, edge_index[1], reduce ='sum')
    min_deg = deg_hist.min()
    if min_deg == 0:
        return 1 / deg_hist.max()
    else:
        return min_deg / deg_hist.max()

class Pruner(object):
    def __init__(self,style, ratio=0.9):
        self.style = style
        if style == 'random':
            self.ratio1=ratio
        elif style == 'truncate':
            self.ratio2=ratio
        else:
            self.ratio1 = self.ratio2 = 1-ratio/2
    
    def set_rate(self,ratio):
        if self.style == 'random':
            self.ratio1=ratio
        elif self.style == 'truncate':
            self.ratio2=ratio
        else:
            self.ratio1 = self.ratio2 = 1-ratio/2

    def Randompruning(self, edge_index, ratio=0.9, ise_id=False):
        E = edge_index.size(1)
        #print(E)
        e_id = torch.randperm(E)[:int(ratio*E)]
        mask = torch.zeros(E)
        mask[e_id] = 1
        if ise_id == False:
            return mask.bool()
        else:
            return e_id
    
    def Truncation(self,edge_index, ratio=0.9, ise_id=False):
        E = edge_index.size(1)
        src = torch.ones(E).to(edge_index.device)
        deg_hist = scatter(src, edge_index[1], reduce ='sum')
        aaa = deg_hist.min()
        if aaa !=0:
            minmaxrate = aaa / deg_hist.max()
        else:
            minmaxrate = 1 / deg_hist.max()

        ratio = math.pow(1 / minmaxrate, ratio-1) + 1e-5
        thold = int(torch.max(deg_hist)*ratio)
        if thold < 1:
            thold = 1
        deg_cnt = deg_hist[edge_index[1]]
        num = (deg_hist > thold).nonzero().size(0)
        tot_e_cnt = num * thold
        e_id = (deg_cnt >thold).nonzero().squeeze()
        e_id = e_id[torch.randperm(e_id.size(0))[tot_e_cnt:]]
        mask = torch.ones(E)
        mask[e_id]=0
        if ise_id == False:
            return mask.bool()
        else:
            return torch.arange(E)[mask.bool()]
    
    def HybridPrune(self, edge_index, ratio1=0.9, ratio2=0.9):
        E = edge_index.size(1)
        e_id = self.Truncation(edge_index,ratio1,ise_id=True)
        f_e_ids = self.Randompruning(edge_index[:,e_id], ratio2, ise_id=True)
        mask = torch.zeros(E)
        mask[e_id[f_e_ids]] = 1
        return mask.bool()

    def prune(self, edge_index):
        if self.style == 'random':
            return self.Randompruning(edge_index, self.ratio1)
        elif self.style == 'truncate':
            return self.Truncation(edge_index, self.ratio2)
        elif self.style == 'hybrid':
            return self.HybridPrune(edge_index, self.ratio1, self.ratio2)
        elif self.style == 'none':
            return torch.ones(edge_index.size(1)).bool()
        else:
            print('not implemented')
            return edge_index

    