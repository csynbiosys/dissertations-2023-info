#!/usr/bin/env python3
# -*- coding: utf-8 -*-

 

import warnings
import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.data import  HeteroData
from torch_geometric.nn import SAGEConv, GAE, RGCNConv, Node2Vec, FastRGCNConv, to_hetero


warnings.filterwarnings('ignore')



# class GNN(torch.nn.Module):
#     def __init__(self, hidden_channels):
#         super().__init__()
#         self.conv1 = SAGEConv(hidden_channels, hidden_channels)
#         self.conv2 = SAGEConv(hidden_channels, hidden_channels)
#         self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
# #         self.dropout = torch.nn.Dropout(0.5)
#     def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
# #         x = F.relu(self.conv1(x, edge_index))

# #         x = self.conv2(x, edge_index)

#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)
#         x = F.relu(x)
# #         x = self.dropout(x)

#         x = self.conv2(x, edge_index)

#         return x

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        self.dropout = torch.nn.Dropout(0.1)
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
#         x = F.relu(self.conv1(x, edge_index))

#         x = self.conv2(x, edge_index)

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        # x = self.dropout(x)
        x = self.conv3(x, edge_index)
        # x = self.bn3(x)
        # x = F.relu(x)
        # x = self.conv4(x, edge_index)

        return x

 

class Classifier(torch.nn.Module):
    def forward(self, x_st: Tensor, x_vc: Tensor, edge_label_index: Tensor) -> Tensor:


        edge_feat_st = x_st[edge_label_index[0]]
        edge_feat_vc = x_vc[edge_label_index[1]]

        return (edge_feat_st * edge_feat_vc).sum(dim=-1)

 

class Model(torch.nn.Module):
    def __init__(self, hidden_channels, data):
        super().__init__()
        #data = self.data

        self.st_lin = torch.nn.Linear(2613, hidden_channels)
        self.vc_lin = torch.nn.Linear(4517, hidden_channels)
        self.vc_emb = torch.nn.Embedding(data["investor"].num_nodes, hidden_channels)
        self.st_emb = torch.nn.Embedding(data["startup"].num_nodes, hidden_channels)

        self.gnn = GNN(hidden_channels)

        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()

 

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
          "investor": self.vc_lin(data["investor"].x) +self.vc_emb(data["investor"].node_ids),
          "startup": self.st_lin(data["startup"].x) + self.st_emb(data["startup"].node_ids),
        } 

        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["startup"],
            x_dict["investor"],
            data['startup', 'invested_by', 'investor'].edge_label_index,
        )
        return pred, x_dict  

 

    def topN(self, startup_id, n, data):

        _, x_dict = self.forward(data)
        z_startups, z_investors = x_dict["startup"], x_dict["investor"]
        scores = torch.squeeze(z_startups[startup_id] @ z_investors.t())
        return torch.topk(scores, k=n)