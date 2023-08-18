#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import Tensor

from torch_geometric.nn import SAGEConv, to_hetero, GATv2Conv
from torch_geometric.nn.pool import knn_graph
import torch.nn.functional as F

import torch.nn as nn

class GCN(nn.Module):
    def __init__(self, n_layers = 3, in_dim = 512, n_heads = 1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for k in range(n_layers):
            self.convs.append(GATv2Conv(in_dim, in_dim // n_heads, n_heads = n_heads, edge_dim = 2))
            self.norms.append(nn.LayerNorm((in_dim, )))

        self.act = nn.ReLU()
    
    def forward(self, x, edge_index, edge_attr):
        for k in range(len(self.convs)):
            x = self.act(self.norms[k](self.convs[k](x, edge_index, edge_attr) + x)) 

        return x
    
class Model(nn.Module):
    def __init__(self, k = 7, in_dim = 512, graph_dim = 512):
        super().__init__()
        self.k = k
        
        self.embedding_inv = nn.Sequential(nn.Linear(in_dim, graph_dim))
                                           
        self.embedding_org = nn.Sequential(nn.Linear(in_dim, graph_dim))
        
        self.gcn = GCN(in_dim = graph_dim)
        self.lin0 = nn.Linear(graph_dim * 2, graph_dim)
        self.lin1 = nn.Linear(graph_dim * 2, graph_dim)
        
    # x0 are the investor features
    # x1 are the org features
    def forward(self, x, edge_index, edge_attr, n_inv, n_unknown):
        x0 = x[:n_inv]
        x1 = x[n_inv:]
        
        x0 = self.embedding_inv(x0)
        x1 = self.embedding_org(x1)
        
        x = torch.cat([x0, x1], 0)
        
        x = self.gcn(x, edge_index, edge_attr)
        
        x0 = self.lin0(torch.cat([x0, x[:n_inv]], -1))
        x1 = self.lin1(torch.cat([x1[-n_unknown:], x[-n_unknown:]], -1))
        
        # inv to org
        x = torch.matmul(x0, x1.T)
        
        return x
    
    
        
        