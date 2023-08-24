#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import logging

import torch
from layers import Model
from torch_geometric.sampler import BaseSampler
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader, DynamicBatchSampler
from torch_geometric.utils import k_hop_subgraph
from sklearn.metrics import accuracy_score, f1_score
from scipy.special import expit
from torch_geometric.nn.pool import knn_graph
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json
import pandas as pd
import pickle


import random

def chunks(lst, n):
    _ = []    

    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        _.append(lst[i:i + n])
        
    return _

def split_dataset(x0, x1, edge_index, prop = 0.01):
    d = HeteroData()
    d['inv'].x = torch.FloatTensor(x0)
    d['org'].x = torch.FloatTensor(x1)
    
    d['inv', 'ii', 'org'].edge_index = torch.LongTensor(edge_index)
    
    return d

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "None.npz")
    parser.add_argument("--pca", default = "pca_model.pkl")
    parser.add_argument("--lr", default = "0.0001") #4.24330774357976e-05
    parser.add_argument("--weight_decay", default = "0.006331364160485268") #0.03137191337585493
    parser.add_argument("--k", default = "16") #15
    
    parser.add_argument("--model_path", default = "model_best_check.pth")

    parser.add_argument("--odir", default = "None")
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






def process_startup_features(provided_startup_features, args):

    pca = pickle.load(open(args.pca, 'rb'))
    
    
    

    X = provided_startup_features
    
    variance_mask = np.load('variance_mask_x1.npy')  
    Xm = np.load('Xm_x1.npy')  
    
    X = X[:, variance_mask]
    

    X = X - Xm
    
    x0 = np.load(args.ifile)['x0']
    
    pca = pickle.load(open(args.pca, 'rb'))
    
    X_transformed = pca.transform(X)
    
    mean_pca = np.load('mean_pca_x1.npy') 
    std_pca = np.load('std_pca_x1.npy')  
    
    X_normalized = (X_transformed - mean_pca) / std_pca
    
    return torch.tensor(X_normalized, dtype=torch.float32)


def recommend(provided_startup_features):
    args = parse_args()
    device = torch.device('cpu')
    PATH = args.model_path
    batch_size = 16


    model = Model().to(device)
    model.load_state_dict(torch.load(PATH))
    model.eval()


    x0 = np.load(args.ifile)['x0']
    x1 = np.load(args.ifile)['x1']
    edge_index = np.load('data.npz')['edge_index'][::-1, :].copy()

    data = split_dataset(x0, x1, edge_index)
    data = T.ToUndirected()(data)
    edge_index = data['inv', 'ii', 'org'].edge_index
    x_new = process_startup_features(provided_startup_features, args)


    ii = list(range(x0.shape[0]))
    #random.shuffle(ii)
    inv_ii = chunks(list(ii), batch_size)
    Y_pred = []
    for ix, ii in enumerate(inv_ii):
        if ix % 200 == 0:
            logging.info('on batch {}...'.format(ix))

        x_inv = x0[ii]
    
    
    

        _, edge_index_, _, _ = k_hop_subgraph(torch.LongTensor(np.array(ii)), 1, edge_index, directed=False, flow='target_to_source')
        
    
        org_ii_ = list(set(list(edge_index_[1].numpy())))
    
        d_sub = dict()
        d_sub['inv'] = torch.LongTensor(np.array(ii))
        d_sub['org'] = torch.LongTensor(np.array(org_ii_))
    
        batch = data.subgraph(d_sub)
    
        e = batch['inv', 'ii', 'org'].edge_index.clone()
        n_known_edges = e.shape[1]
        e[1] += batch['inv'].x.shape[0]
        
        x = torch.cat([batch['inv'].x, batch['org'].x, x_new], 0).to(device)
    
        knn_indices = knn_graph(torch.cat([batch['org'].x, x_new], 0), int(args.k))
        knn_indices += batch['inv'].x.shape[0]
    
        e = torch.cat([e, knn_indices], 1).to(device)
        
        edge_attr = torch.zeros((e.shape[1], 2))
        edge_attr[-knn_indices.shape[1]:, 1] = 1.
        edge_attr[:n_known_edges, 0] = 1.
        edge_attr = edge_attr.to(device)
    
        with torch.no_grad():
            y_pred = model(x, e, edge_attr, batch['inv'].x.shape[0], x_new.shape[0])

            
        Y_pred.append(y_pred.detach().cpu().numpy().T)
        
    
    

    with open("investor_name_dict.json", "r") as file:
        investor_name_dict = json.load(file)


    Y_pred = Y_pred[:-1]




    Y_pred_array = np.concatenate(Y_pred, axis=0)
    
    
    
    investor_indices_with_high_predictions = np.where(expit(Y_pred_array)>0.7)[0] #add expit

    
    print(investor_indices_with_high_predictions)
    

    sorted_investor_indices = investor_indices_with_high_predictions[np.argsort(-Y_pred_array.flatten()[investor_indices_with_high_predictions])]


    top_10_investors = sorted_investor_indices[:10]
    
    print(top_10_investors)
    
    
    recommended_investors = [investor_name_dict[str(index)] for index in top_10_investors]
    print(recommended_investors)



df_new_startups = pd.read_csv('test_startups_r.csv')
 

df_new_startups.drop('Unnamed: 0', inplace=True, axis=1)
df_new_startups.drop('Unnamed: 0.1', inplace=True, axis=1)
df_new_startups.fillna(0, inplace=True)
 
num_columns = len(df_new_startups.columns)

 
 
 
 
print(df_new_startups[8:9]['org_uuid'])

X = df_new_startups.loc[df_new_startups['org_uuid'].drop_duplicates().index].drop(columns=['org_uuid','description', 'embeddings'])
X = X[8:9].values
provided_startup_features = np.array(X)  
recommend(provided_startup_features)

  
# if __name__ == '__main__':
#     main()






