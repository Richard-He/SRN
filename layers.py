import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GENConv, GATConv, SAGEConv, GCNConv


class AdaGNNLayer(torch.nn.Module):
    def __init__(self, conv=None, norm=None, act=None,
                    dropout=0., ckpt_grad=False, lin=False):
        super(AdaGNNLayer, self).__init__()

        self.conv = conv
        self.norm = norm
        self.act = act
        self.dropout = dropout
        self.ckpt_grad = ckpt_grad
        self.fixed = True
        self.lin = lin

    def unfix(self):
        self.fixed = False

    def forward(self, *args, **kwargs):
        args = list(args)
        x = args.pop(0)
        if self.fixed == True:
            return x
        else:
            if self.norm is not None:
                h = self.norm(x)
            if self.act is not None:
                h = self.act(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.conv is not None and self.ckpt_grad and h.requires_grad:
                h = checkpoint(self.conv, h, *args, **kwargs)
            else:
                if self.lin == False:
                    h = self.conv(h, *args, **kwargs)
                else:
                    h = self.conv(h)
            return h + x
    
    def reset_parameters(self):
        self.conv.reset_parameters()

class SAGE(torch.nn.Module):
    def __init__(self,in_channels, hidden_channels, out_channels, num_layers,dropout=0.5):
        super(SAGE, self).__init__()
        self.edge_encoder = Linear(in_channels, hidden_channels)
        self.node_encoder = Linear(in_channels, hidden_channels)
        self.layers = torch.nn.ModuleList()

        for i in range(1, num_layers+1):
            conv = SAGEConv(hidden_channels, hidden_channels)
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = AdaGNNLayer(conv,norm,act,dropout=dropout)
            self.layers.append(layer)
        
        self.lin = Linear(hidden_channels, out_channels)
        self.currenlayer = 1
        self.layers[0].unfix()
        self.num_layers = num_layers
    
    def get_fixed_layer_cnt(self):
        return self.currenlayer

    def unfix(self, until_num_layer=1):
        for i in range(until_num_layer-self.currenlayer):
            if self.currenlayer > len(self.layers):
                return self.currenlayer
            else:
                self.layers[self.currenlayer].unfix()
                self.currenlayer +=1
        return self.currenlayer

    
    def forward(self, x, edge_index, edge_attr=None):
        x = self.node_encoder(x)
        x = self.layers[0].conv(x, edge_index)

        for layer in self.layers:
            x = layer(x, edge_index)

        x = self.layers[0].norm(x)
        x = self.layers[0].act(x)
        x = F.dropout(x, p=0.1, training=self.training)
        return self.lin(x)
    

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.lin.reset_parameters()
        self.node_encoder.reset_parameters()
        if self.edge_encoder != None:
            self.edge_encoder.reset_parameters()

class GCN(torch.nn.Module):
    def __init__(self,in_channels, hidden_channels, out_channels, num_layers,dropout=0.5):
        super(GCN, self).__init__()
        self.edge_encoder = Linear(in_channels, hidden_channels)
        self.node_encoder = Linear(in_channels, hidden_channels)
        self.layers = torch.nn.ModuleList()

        for i in range(1, num_layers+1):
            conv = GCNConv(hidden_channels, hidden_channels)
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = AdaGNNLayer(conv,norm,act,dropout=dropout)
            self.layers.append(layer)
        
        self.lin = Linear(hidden_channels, out_channels)
        self.currenlayer = 1
        self.layers[0].unfix()
        self.num_layers = num_layers
    
    def get_fixed_layer_cnt(self):
        return self.currenlayer

    def unfix(self, until_num_layer=1):
        for i in range(until_num_layer-self.currenlayer):
            if self.currenlayer > len(self.layers):
                return self.currenlayer
            else:
                self.layers[self.currenlayer].unfix()
                self.currenlayer +=1
        return self.currenlayer

    
    def forward(self, x, edge_index, edge_attr=None):
        x = self.node_encoder(x)
        # edge_attr = self.edge_encoder(edge_attr)
        #print(x.size(), edge_index.size())
        x = self.layers[0].conv(x, edge_index)

        for layer in self.layers:
            x = layer(x, edge_index)

        x = self.layers[0].norm(x)
        x = self.layers[0].act(x)
        x = F.dropout(x, p=0.1, training=self.training)
        return self.lin(x)
    

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.lin.reset_parameters()
        self.node_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()

class GAT(torch.nn.Module):
    def __init__(self,in_channels, hidden_channels, out_channels, num_layers, dropout=0.5):
        super(GAT, self).__init__()
        self.edge_encoder = Linear(in_channels, hidden_channels)
        self.node_encoder = Linear(in_channels, hidden_channels)
        self.layers = torch.nn.ModuleList()
        
        for i in range(1, num_layers+1):
            conv = GATConv(hidden_channels, hidden_channels)
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = AdaGNNLayer(conv,norm,act,dropout=dropout)
            self.layers.append(layer)
        
        self.lin = Linear(hidden_channels, out_channels)
        self.currenlayer = 1
        self.layers[0].unfix()
        self.num_layers = num_layers
    
    def get_fixed_layer_cnt(self):
        return self.currenlayer

    def unfix(self, until_num_layer=1):
        for i in range(until_num_layer-self.currenlayer):
            if self.currenlayer > len(self.layers):
                return self.currenlayer
            else:
                self.layers[self.currenlayer].unfix()
                self.currenlayer +=1
        return self.currenlayer

    
    def forward(self, x, edge_index, edge_attr=None):
        x = self.node_encoder(x)
        # edge_attr = self.edge_encoder(edge_attr)
        #print(x.size(), edge_index.size())
        x = self.layers[0].conv(x, edge_index)

        for layer in self.layers:
            x = layer(x, edge_index)

        x = self.layers[0].norm(x)
        x = self.layers[0].act(x)
        x = F.dropout(x, p=0.1, training=self.training)
        return self.lin(x)
    

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.lin.reset_parameters()
        self.node_encoder.reset_parameters()
        if self.edge_encoder != None:
            self.edge_encoder.reset_parameters()


class MLP(torch.nn.Module):
    def __init__(self,in_channels, hidden_channels, out_channels, num_layers,dropout=0.5):
        super(MLP, self).__init__()
        self.node_encoder = Linear(in_channels, hidden_channels)
        self.layers = torch.nn.ModuleList()

        for i in range(1, num_layers+1):
            conv = Linear(hidden_channels, hidden_channels)
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = AdaGNNLayer(conv,norm,act,dropout=dropout, lin=True)
            self.layers.append(layer)
        
        self.lin = Linear(hidden_channels, out_channels)
        self.currenlayer = 1
        self.layers[0].unfix()
        self.num_layers = num_layers
    
    def get_fixed_layer_cnt(self):
        return self.currenlayer

    def unfix(self, until_num_layer=1):
        for i in range(until_num_layer-self.currenlayer):
            if self.currenlayer > len(self.layers):
                return self.currenlayer
            else:
                self.layers[self.currenlayer].unfix()
                self.currenlayer +=1
        return self.currenlayer

    
    def forward(self, x, edge_index, edge_attr=None):
        x = self.node_encoder(x)
        # edge_attr = self.edge_encoder(edge_attr)
        #print(x.size(), edge_index.size())
        x = self.layers[0].conv(x)

        for layer in self.layers:
            x = layer(x, edge_index)

        x = self.layers[0].norm(x)
        x = self.layers[0].act(x)
        x = F.dropout(x, p=0.1, training=self.training)
        return self.lin(x)
    

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.lin.reset_parameters()
        self.node_encoder.reset_parameters()
        

class GEN(torch.nn.Module):
    def __init__(self,in_channels, hidden_channels, out_channels, num_layers,dropout=0.5):
        super(GEN, self).__init__()

        self.node_encoder = Linear(in_channels, hidden_channels)
        self.edge_encoder = Linear(in_channels, hidden_channels)

        self.layers = torch.nn.ModuleList()

        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                            t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = AdaGNNLayer(conv, norm, act, dropout=dropout, ckpt_grad=False)
            layer.unfix()
            self.layers.append(layer)
        
        self.lin = Linear(hidden_channels, out_channels)
        self.currenlayer = 1
        # self.layers[0].unfix()
        self.num_layers = num_layers
        

    def get_fixed_layer_cnt(self):
        return self.currenlayer


    def unfix(self, until_num_layer=1):
        for i in range(until_num_layer-self.currenlayer):
            if self.currenlayer > len(self.layers):
                return self.currenlayer
            else:
                self.layers[self.currenlayer].unfix()
                self.currenlayer +=1
        return self.currenlayer
    

    def forward(self, x, edge_index, edge_attr=None):
        x = self.node_encoder(x)
        if edge_attr != None:
            edge_attr = self.edge_encoder(edge_attr)
            x = self.layers[0].conv(x, edge_index, edge_attr)
        else:
            x = self.layers[0].conv(x, edge_index)
        if len(self.layers) > 1:
            for layer in self.layers[1:]:
                if edge_attr == None:
                    x = layer(x, edge_index)
                else:
                    x = layer(x, edge_index, edge_attr)
        x = self.layers[0].norm(x)
        x = self.layers[0].act(x)
        x = F.dropout(x, p=0.1, training=self.training)
        return self.lin(x)
    

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.lin.reset_parameters()
        self.node_encoder.reset_parameters()
        if self.edge_encoder != None:
            self.edge_encoder.reset_parameters()