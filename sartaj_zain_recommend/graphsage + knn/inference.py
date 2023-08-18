#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import warnings
import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.data import  HeteroData
from torch_geometric.nn import SAGEConv, GAE, RGCNConv, Node2Vec, FastRGCNConv, to_hetero , knn_graph, knn
import torch_geometric.transforms as T
from torch.nn.functional import cosine_similarity
import pandas as pd
import numpy as np
import json

warnings.filterwarnings('ignore')


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

def precision_at_k(recommended, actual, k):
    """Compute Precision at K."""
    top_k_recommended = recommended[:k]
    relevant_and_recommended = len(set(top_k_recommended).intersection(set(actual)))
    return relevant_and_recommended / k

def recall_at_k(recommended, actual, k):
    """Compute Recall at K."""
    top_k_recommended = recommended[:k]
    relevant_and_recommended = len(set(top_k_recommended).intersection(set(actual)))
    return relevant_and_recommended / len(actual)

def hit_rate(recommended, actual):
    """Compute Hit Rate."""
    hits = len(set(recommended).intersection(set(actual)))
    return 1 if hits > 0 else 0

def average_precision(recommended, actual):
    """Compute average precision for a single list."""
    ap = 0.0
    correct_count = 0
    for i, rec in enumerate(recommended):
        if rec in actual:
            correct_count += 1
            ap += correct_count / (i + 1)
    return ap / len(actual)

def mean_average_precision(all_recommended, all_actual):
    """Compute Mean Average Precision."""
    all_ap = [average_precision(recommended, actual) for recommended, actual in zip(all_recommended, all_actual)]
    if len(all_ap)>0:
        return sum(all_ap) / len(all_ap)
    else:
        return 0
 
def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "data.pt")
    parser.add_argument("--ofile_df", default = "test_startups_r.csv")
    parser.add_argument("--ofile_in", default = "investor_name_dict.json")
    parser.add_argument("--ofile_sn", default = "startup_name_dict.json")
    parser.add_argument("--ofile_sm", default = "startup_map.json")
    parser.add_argument("--ofile_im", default = "investor_map.json")
    
    

    parser.add_argument("--odir", default = "None")
    parser.add_argument("--model_path", default = "model_gsageemb_and_fea__1e-04_1e-05_128.pth")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if args.odir != "None":
        if not os.path.exists(args.odir):
            os.system('mkdir -p {}'.format(args.odir))
            logging.debug('root: made output directory {0}'.format(args.odir))
    # ${odir_del_block}

    return args    
def recommend( st_id, n, data, model, investor_name_dict, startup_name_dict):
    try:
        recommendations = model.topN(st_id, n, data)
    except KeyError:

        recommendations = []
    else:
        recommendations = recommendations.indices.cpu().tolist()
        # recommendations = list(map(lambda x: investor_name_dict[str(x)], recommendations))
    # print("startup: ", startup_name_dict[str(st_id)])
    return recommendations

    
def add_node(data, new_node_features, new_node_id, k):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_nodes = data.x.size(0)
    data.x=data.x.to(device)
    data.x = torch.cat([data.x, new_node_features], dim=0)

    cos_sim = cosine_similarity(data.x, data.x[num_nodes:], dim=-1)
    knn_similarities, knn_indices = torch.topk(cos_sim, k, largest=True)
    knn_indices = knn_indices.to(device)
    knn_similarities = knn_similarities.to(device)

    new_edges = torch.stack([torch.full((k,), num_nodes, dtype=torch.long).to(device), knn_indices.to(device)]).to(device)
    data.edge_index = torch.cat([data.edge_index.to(device), new_edges], dim=1).to(device) 
    data.node_id = torch.cat([data.node_id.to(device), torch.tensor([new_node_id]).to(device)])

    return data, knn_indices, knn_similarities

def main():
    args = parse_args()
    data = torch.load(args.ifile)
    model = Model(hidden_channels=64, data=data)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    f = open(args.ofile_in)
    investor_name_dict = json.load(f)
    
    f = open(args.ofile_sn)
    startup_name_dict = json.load(f)
    
    transform = T.RandomLinkSplit(
    num_val=0.0,
    num_test=0.0,
    disjoint_train_ratio=0.0,
    neg_sampling_ratio=0.0,
    add_negative_train_samples=False,
    edge_types=('startup', 'invested_by', 'investor'),
    rev_edge_types=('investor', 'rev_invested_by', 'startup'), 
    )
    data1, _, _ = transform(data)
    
    knn_data = torch.load('data_st_des_fet_r.pt',map_location=torch.device('cpu'))
    df_test_startups = pd.read_csv(args.ofile_df)
    df_check = df_test_startups.copy()
    df_test_startups.drop('Unnamed: 0',inplace = True, axis = 1)
    startups = np.unique(df_test_startups['org_name'].values)
    df_test_startups.fillna(0,inplace=True)
    
    new_node_id = data['startup'].node_ids.numpy()[-1]
    limit = data['startup'].node_ids.numpy()[-1] +1
    all_recommended_investors = []
    all_actual_investors = []
    
    
    for st in startups[0:100]:
        df_test = df_test_startups[df_test_startups['org_name']==st]
        # print(df_test.columns)
        new_startup_features = df_test.loc[df_test['org_uuid'].drop_duplicates().index].drop(columns=['Unnamed: 0.1','org_name','org_uuid','investor_name','investor_uuid','embeddings','description'])
        new_startup_features = torch.tensor(new_startup_features[:1].values, dtype=torch.int).to(device)
        
        new_node_id += 1
        data2, knn_indices, knn_similarities = add_node(knn_data, new_startup_features, new_node_id, k=5)
    
        # print("New Startup: ", st)
    #     print("Nearest neighbors in order: ", knn_indices.tolist())
    #     print("Corresponding similarities: ", knn_similarities.tolist())
        lst1 = []
        for idx in range(len(knn_indices)):
            neighbor_index = knn_indices[idx].item()
            
            if neighbor_index <= limit:
                lst1.append(recommend(neighbor_index - 1, 10, data1,model, investor_name_dict, startup_name_dict))
    
    #     print(lst1)
        recs = []
        for j in range(10):
            for lst in lst1:
                recs.append(lst[j])
    #         print(recs)
    #         break
        recs = list(set(recs))
        recs = list(map(lambda x: investor_name_dict[str(x)], recs))
    
        # print(" Recommendations: ", recs)
        # print("Actual:", df_check[df_check['org_name']==st]['investor_name'].values)
        
        current_recommendations = set(recs)
        current_actuals = set(df_check[df_check['org_name']==st]['investor_name'].values)
        
        # Find common investors
        common_investors = current_recommendations.intersection(current_actuals)
        
        # Check if there are actual investors before displaying recommendations
        if len(common_investors) > 0:
            print("Startup: ", st)
            print(" Recommendations: ", recs)
            print("Actual:", df_check[df_check['org_name']==st]['investor_name'].values)
            
            print("Common Investors:", common_investors)
            
        
        all_recommended_investors.append(current_recommendations)
        all_actual_investors.append(current_actuals)
        
    precision_at_k_values = []
    recall_at_k_values = []
    hit_rate_values = []
    map_values = []
    
    for recommended, actual in zip(all_recommended_investors, all_actual_investors):
        recommended = list(recommended)
        actual = list(actual)
        precision_at_k_values.append(precision_at_k(recommended, actual, k=20))
        recall_at_k_values.append(recall_at_k(recommended, actual, k=20))
        hit_rate_values.append(hit_rate(recommended, actual))
        map_values.append(mean_average_precision(recommended, actual))
    
    # Average the metric values
    avg_precision_at_k = np.mean(precision_at_k_values)
    avg_recall_at_k = np.mean(recall_at_k_values)
    avg_hit_rate = np.mean(hit_rate_values)
    avg_map = np.mean(map_values)
    
    print(f"Average Precision@K: {avg_precision_at_k}")
    print(f"Average Recall@K: {avg_recall_at_k}")
    print(f"Average Hit Rate: {avg_hit_rate}")
    print(f"Mean Average Precision: {avg_map}")
    
    
if __name__ == '__main__':
     main()  
