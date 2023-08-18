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
    # # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "None.npz")
    
    parser.add_argument("--lr", default = "0.00012156299936369672") #4.24330774357976e-05
    parser.add_argument("--weight_decay", default = "0.0021852322399311197") #0.03137191337585493
    parser.add_argument("--k", default = "17") #15
    
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
def precision_at_k(y_true, y_pred, b):
    # Get top-k predicted scores
    top_k_indices = np.argsort(y_pred)[-b:]
    
    # Count the actual positive samples in the top-k predictions
    relevant_items = sum(y_true[top_k_indices])
    
    return relevant_items / b

def recall_at_k(y_true, y_pred, b):
    # Get top-k predicted scores
    top_k_indices = np.argsort(y_pred)[-b:]
    
    # Count the actual positive samples in the top-k predictions
    relevant_items = sum(y_true[top_k_indices])
    
    # Total number of actual positive samples
    total_positive = sum(y_true)
    
    return relevant_items / total_positive

def hit_rate(y_true, y_pred, b):
    # Get top-k predicted scores
    top_k_indices = np.argsort(y_pred)[-b:]
    
    # Check if any of the top-k predictions are actual positive samples
    hit = int(any(y_true[top_k_indices]))
    
    return hit


def main():
    args = parse_args()
    device = torch.device('cpu')
    PATH = args.model_path
    #torch.save(model.state_dict(), PATH)
    x0 = np.load(args.ifile)['x0']
    x1 = np.load(args.ifile)['x1']
    batch_size = 12

        
    with np.load('ts_np.npz', allow_pickle=True) as loaded:
        test_orgs = loaded['test_orgs']
        neg_pos = loaded['neg_pos']
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight = torch.Tensor(np.array([np.mean(neg_pos)])).to(device))
  
    edge_index = np.load('data.npz')['edge_index'][::-1,:].copy()
    
    data = split_dataset(x0, x1, edge_index)
    data = T.ToUndirected()(data)
    
    model  = Model().to(device)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    
    test_losses = []
    test_accs = []
    
    x0 = data['inv'].x
    x1 = data['org'].x

    
    test_orgs = list(test_orgs.item())
    
    edge_index = data['inv', 'ii', 'org'].edge_index
    
    ii = list(range(x0.shape[0]))
    #random.shuffle(ii)
    
    inv_ii = chunks(list(ii), batch_size)
    org_ii = chunks(list(test_orgs), batch_size)
    
    d_sub = dict()
    d_sub['inv'] = torch.LongTensor(np.array(range(x0.shape[0])))
    d_sub['org'] = torch.LongTensor(np.array(test_orgs[:4 * batch_size]))
    
    target = data.subgraph(d_sub)
    
    y = np.zeros((4 * batch_size, x0.shape[0]))
    y[target['inv', 'ii', 'org'].edge_index[1], target['inv', 'ii', 'org'].edge_index[0]] = 1.
    
    logging.info('have {} batches to test against...'.format(len(inv_ii)))
    logging.info('have {} new org batches...'.format(len(org_ii)))
    
    Y_pred = []
    for ix, ii in enumerate(inv_ii):
        if ix % 200 == 0:
            logging.info('on batch {}...'.format(ix))
        
        # get the graph to test against
        x_inv = x0[ii]
        
        # print(ii)
        
        _, edge_index_, _, _ = k_hop_subgraph(torch.LongTensor(np.array(ii)), 1, edge_index, directed = False, flow = 'target_to_source')
        org_ii_ = list(set(list(edge_index_[1].numpy())))
                
        org_ii_ = list(set(org_ii_).difference(test_orgs))
    
        d_sub = dict()
        d_sub['inv'] = torch.LongTensor(np.array(ii))
        d_sub['org'] = torch.LongTensor(np.array(org_ii_))
        
        batch = data.subgraph(d_sub)
        
        _ = []
        for ii_ in org_ii[:4]:
            # print(ii_)
            x_new = x1[ii_]
            #x_new = x1[ii_] New oeganizaiton features
    
    
            # make unified edges and nodes
            e = batch['inv', 'ii', 'org'].edge_index.clone()
    
            n_known_edges = e.shape[1]
    
            e[1] += batch['inv'].x.shape[0]
            
            x = torch.cat([batch['inv'].x, batch['org'].x, x_new], 0).to(device)
    
            knn_indices = knn_graph(torch.cat([batch['org'].x, x_new], 0), int(args.k))
            knn_indices += batch['inv'].x.shape[0]
    
            e = torch.cat([e, knn_indices], 1).to(device)
            
            edge_attr = torch.zeros((e.shape[1], 2))
            edge_attr[-knn_indices.shape[1]:,1] = 1.
            edge_attr[:n_known_edges,0] = 1.
    
            edge_attr = edge_attr.to(device)
    
            with torch.no_grad():
                y_pred = model(x, e, edge_attr, batch['inv'].x.shape[0], x_new.shape[0])
            
            _.extend(y_pred.detach().cpu().numpy().T)
            
        Y_pred.append(np.array(_))
                
    Y_pred = np.concatenate(Y_pred, 1)

    np.savez('test_preds.npz', y_pred = Y_pred, y = y, ii = np.array(test_orgs[:4 * batch_size], dtype = np.int32))
    
    f = f1_score(y.flatten(), np.round(expit(Y_pred).flatten()))
    logging.info('got eval f1 of {}...'.format(f))
    # Evaluate hit rate, precision at k, and recall at k
    b = 10  # or any other value you want to set for 'k'
    hit_rates = []
    precisions = []
    recalls = []
    
    for i in range(Y_pred.shape[0]):
        true_values = y[i]
        predicted_values = Y_pred[i]
        
        hit_rates.append(hit_rate(true_values, predicted_values, b))
        precisions.append(precision_at_k(true_values, predicted_values, b))
        recalls.append(recall_at_k(true_values, predicted_values, b))
    
    avg_hit_rate = np.mean(hit_rates)
    avg_precision_at_k = np.mean(precisions)
    avg_recall_at_k = np.mean(recalls)
    
    logging.info(f'Average Hit Rate at {b}: {avg_hit_rate}')
    logging.info(f'Average Precision at {b}: {avg_precision_at_k}')
    logging.info(f'Average Recall at {b}: {avg_recall_at_k}')
    
    Y_pred = torch.FloatTensor(expit(Y_pred)).to(device)
    y = torch.FloatTensor(y).to(device)
    
    loss = criterion(Y_pred, y).item()
    logging.info('got eval loss of {}...'.format(loss))
    
    # Load the data from the .npz file
    data_pred = np.load('test_preds.npz')
    y_pred = data_pred['y_pred']
    print(y_pred.shape)
    y = data_pred['y']
    print(y.shape)
    original_indices = data_pred['ii']
    print(original_indices)
    
    
    # Load the mapping of indices to investor names from the JSON file
    with open("investor_name_dict.json", "r") as file:
        investor_name_dict = json.load(file)
        
    with open("startup_name_dict.json", "r") as file:
        startup_name_dict = json.load(file)
    
    # Implementing the logic to identify the top 10 investors for each startup based on predictions
    top_investors_for_startups = {}
    
    for startup_index in range(y_pred.shape[0]):  # Loop through each startup
        # Get predictions for the startup and filter out investors with predictions less than 0.7
        investor_indices_with_high_predictions = np.where(expit(y_pred[startup_index]) > 0.5)[0]
    
        # Sort these investors based on the prediction values in descending order
        sorted_investor_indices = investor_indices_with_high_predictions[np.argsort(-y_pred[startup_index][investor_indices_with_high_predictions])]
    
        # Take the top 10 investors
        top_10_investors = sorted_investor_indices[:10]
    
        # Map indices to investor names and store in the dictionary
        original_startup_index = original_indices[startup_index]
        top_investors_for_startups[original_startup_index] = [investor_name_dict[str(index)] for index in top_10_investors]
    
    # Check if the actual investor is present in the recommended investors for each startup
    for startup_index in range(y.shape[0]):
        # Get actual investors for the startup from the y matrix
        actual_investor_indices = np.where(y[startup_index] == 1)[0]
        actual_investor_names = [investor_name_dict[str(index)] for index in actual_investor_indices]
    
        # Get the recommended investors for the startup
        recommended_investor_names = top_investors_for_startups[original_indices[startup_index]]
    
        # Check if any actual investor is present in the list of recommended investors
        common_investors = set(actual_investor_names).intersection(recommended_investor_names)
    
        # Print the result
        startup_name = startup_name_dict[str(original_indices[startup_index])]
        print(f"Startup: {startup_name}")
        print(f"Actual Investors: {actual_investor_names}")
        print(f"Recommended Investors: {recommended_investor_names}")
        if common_investors:
            print(f"Common Investors: {common_investors}")
        else:
            print(f"No common investors between actual and recommended.")
        print("-----")
 
  
if __name__ == '__main__':
    main()



