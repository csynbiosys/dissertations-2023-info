#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import optuna
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


# take out some number from x1 and 
def split_dataset(x0, x1, edge_index, prop = 0.01):
    d = HeteroData()
    d['inv'].x = torch.FloatTensor(x0)
    d['org'].x = torch.FloatTensor(x1)
    
    d['inv', 'ii', 'org'].edge_index = torch.LongTensor(edge_index)
    
    return d

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "None.npz")
    
    parser.add_argument("--lr", default = "0.0001") #4.24330774357976e-05
    parser.add_argument("--weight_decay", default = "0.006331364160485268") #0.03137191337585493
    parser.add_argument("--k", default = "16") #15
    
    parser.add_argument("--model_path", default = "model_best.pth")

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

def run(args):
    # args = parse_args()
    device = torch.device('cpu')
    # Initialize early stopping object
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 5 

    x0 = np.load(args.ifile)['x0']
    x1 = np.load(args.ifile)['x1']
  
    edge_index = np.load('data.npz')['edge_index'][::-1,:].copy()
    
    data = split_dataset(x0, x1, edge_index)
    data = T.ToUndirected()(data)
    
    batch_size = 16
    
    loader = NeighborLoader(
        data,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors = [-1, 0],
        # Use a batch size of 128 for sampling training nodes
        batch_size = batch_size      ,#6
        input_nodes = ('inv', torch.LongTensor(np.array(range(x0.shape[0])))),
        shuffle = True
    )
    
    val_prop = 0.05
    train_prop = 0.05
    test_prop = 0.05
    
    prop_separate = val_prop + train_prop + test_prop
    
    train_nodes = []
    val_nodes = []
    test_nodes = []
    
    batches = []
    
    print(next(iter(loader)))
    
    neg_pos = []
    test_orgs = set()
    
    for k, batch in enumerate(loader):
        l = batch['org'].x.shape[0]
        
        known = np.random.choice(range(l), int(l * prop_separate), replace = False)
        unknown = list(set(range(l)).difference(known))
        random.shuffle(unknown)
        
        train, val, test = chunks(unknown, len(unknown) // 3)[:3]
        
        batches.append((batch, train, val, test))
        test_orgs.update(list(batch['org'].n_id[test].numpy()))
    
        x0 = batch['inv'].x
        x1 = batch['org'].x
        edge_index = batch['org', 'rev_ii', 'inv'].edge_index
        edge_index = torch.stack([edge_index[1], edge_index[0]])
        
        y = torch.zeros((x0.shape[0], x1.shape[0]))

        y[edge_index[0], edge_index[1]] = 1.
        y = y[:,train]
        
        y = y.cpu().numpy()
        
        n_pos = np.sum(y)
        n_neg = y.shape[0] * y.shape[1] - n_pos
        
        if n_pos > 0:
            neg_pos.append(n_neg / n_pos)
    
    logging.info('have {} test orgs...'.format(len(test_orgs)))
    
    # with open("test_orgs.json", "w") as f:
    #     json.dump(list(test_orgs), f)
        
    # with open("neg_pos.json", "w") as f:
    #     json.dump(neg_pos, f)
        
    np.savez('ts_np.npz', test_orgs=test_orgs, neg_pos=neg_pos)

    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay = float(args.weight_decay))
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight = torch.Tensor(np.array([np.mean(neg_pos)])).to(device))

        
    trainplotloss = []
    valplotloss = []
    trainplotf1 = []
    valplotf1 = []

    for ix in range(100):
        model.train()

        losses = []
        accs = []
        random.shuffle(batches)

        for ij, batch in enumerate(batches):
            batch, ii, ii_val, ii_test = batch

            optimizer.zero_grad()

            x0 = batch['inv'].x
            x1 = batch['org'].x
            edge_index = batch['org', 'rev_ii', 'inv'].edge_index
            edge_index = torch.stack([edge_index[1], edge_index[0]])

            y = torch.zeros((x0.shape[0], x1.shape[0]))

            y[edge_index[0], edge_index[1]] = 1.
            y = y[:,ii]

            y = y.to(device)

            ii_known = set(range(x1.shape[0])).difference(ii + ii_val + ii_test)

            d_sub = dict()
            d_sub['inv'] = torch.LongTensor(np.array(range(x0.shape[0])))
            d_sub['org'] = torch.LongTensor(np.array(list(ii_known)))

            batch_ = batch.subgraph(d_sub)

            # make unified edges and nodes
            edge_index = batch_['org', 'rev_ii', 'inv'].edge_index
            edge_index = torch.stack([edge_index[1], edge_index[0]])

            n_known_edges = edge_index.shape[1]

            edge_index[1] += x0.shape[0]
            x = torch.cat([batch_['inv'].x, batch_['org'].x, x1[ii]], 0).to(device)

            knn_indices = knn_graph(torch.cat([batch_['org'].x, x1[ii]], 0), int(args.k))
            knn_indices += x0.shape[0]

            edge_index = torch.cat([edge_index, knn_indices], 1).to(device)
            edge_attr = torch.zeros((edge_index.shape[1], 2))
            edge_attr[-knn_indices.shape[1]:,1] = 1.
            edge_attr[:n_known_edges,0] = 1.

            edge_attr = edge_attr.to(device)

            y_pred = model(x, edge_index, edge_attr, x0.shape[0], len(ii))
            loss = criterion(y_pred, y)


            losses.append(loss.item())
            loss.backward()

            optimizer.step()

            if (ij + 1) % 50 == 0:
                y = y.detach().cpu().numpy().flatten()
                y_pred = y_pred.detach().cpu().numpy().flatten()

                accs.append(f1_score(y, np.round(expit(y_pred))))
                # logging.info('step {}: have loss of {}'.format(ij, np.mean(losses)))


        model.eval()

        logging.info('epoch {}: have loss of {}'.format(ix, np.mean(losses)))
        logging.info('f1 score: {}'.format(np.mean(accs)))

        trainplotloss.append(np.mean(losses))

        trainplotf1.append(np.mean(accs))
        
        val_losses = []
        val_accs = []

        logging.info('validating...')
        for ij, batch in enumerate(batches):
            batch, ii, ii_val, ii_test = batch

            x0 = batch['inv'].x
            x1 = batch['org'].x
            edge_index = batch['org', 'rev_ii', 'inv'].edge_index
            edge_index = torch.stack([edge_index[1], edge_index[0]])

            y = torch.zeros((x0.shape[0], x1.shape[0]))

            y[edge_index[0], edge_index[1]] = 1.
            y = y[:,ii_val]

            y = y.to(device)

            #ii_known = set(range(x1.shape[0])).difference(ii + ii_val + ii_test)
            ii_known = set(range(x1.shape[0])).difference(ii_val + ii_test)

            d_sub = dict()
            d_sub['inv'] = torch.LongTensor(np.array(range(x0.shape[0])))
            d_sub['org'] = torch.LongTensor(np.array(list(ii_known)))

            batch_ = batch.subgraph(d_sub)

            # make unified edges and nodes
            edge_index = batch_['org', 'rev_ii', 'inv'].edge_index
            edge_index = torch.stack([edge_index[1], edge_index[0]])

            n_known_edges = edge_index.shape[1]

            edge_index[1] += x0.shape[0]
            x = torch.cat([batch_['inv'].x, batch_['org'].x, x1[ii_val]], 0).to(device)

            knn_indices = knn_graph(torch.cat([batch_['org'].x, x1[ii_val]], 0), int(args.k))
            knn_indices += x0.shape[0]

            edge_index = torch.cat([edge_index, knn_indices], 1).to(device)
            edge_attr = torch.zeros((edge_index.shape[1], 2))
            edge_attr[-knn_indices.shape[1]:,1] = 1.
            edge_attr[:n_known_edges,0] = 1.

            edge_attr = edge_attr.to(device)

            with torch.no_grad():
                y_pred = model(x, edge_index, edge_attr, x0.shape[0], len(ii_val))
                loss = criterion(y_pred, y)

            val_losses.append(loss.item())

            y = y.detach().cpu().numpy().flatten()
            y_pred = y_pred.detach().cpu().numpy().flatten()

            if ij % 50 == 0:
                val_accs.append(f1_score(y, np.round(expit(y_pred))))

            
        logging.info('epoch {}: have val loss of {}'.format(ix, np.mean(val_losses)))   
        logging.info('Val f1 score: {}'.format(np.mean(val_accs)))

        valplotloss.append(np.mean(val_losses))

        valplotf1.append(np.mean(val_accs))
        
        # Early stopping check
        
        if np.mean(val_losses) < best_val_loss:
            best_val_loss = np.mean(val_losses)
            torch.save(model.state_dict(), 'gat2_embandfeat_best_model.pth')  # Save model parameters
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print("Early stopping")
                break

    
    # Create a figure for the plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Plot training and validation loss per epoch
    ax1.plot(trainplotloss, label='Training Loss')
    ax1.plot(valplotloss, label='Validation Loss')
    ax1.set_title(f'Training & Validation Loss per Epoch\nLR: {args.lr}, Weight Decay: {args.weight_decay}, k: {args.k}, Batch Size: {args.batch_size}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot training and validation accuracy per epoch
    ax2.plot(trainplotf1, label='Training F1')
    ax2.plot(valplotf1, label='Validation F1')
    ax2.set_title(f'Training & Validation F1 per Epoch\nLR: {args.lr}, Weight Decay: {args.weight_decay}, k: {args.k}, Batch Size: {args.batch_size}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # Show the plots
    plt.tight_layout()
  
    plt.savefig(f"plot_at2emb_and_fea_{args.lr}_{args.weight_decay}_{args.k}_{args.batch_size}.png")
    plt.show()

    PATH = f"model_gat2emb_and_fea__{args.lr}_{args.weight_decay}_{args.k}_{args.batch_size}.pth"
    torch.save(model.state_dict(), PATH)
    
    
    model  = Model().to(device)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    test_losses = []
    test_accs = []
    
    x0 = data['inv'].x
    x1 = data['org'].x
    
    test_orgs = list(test_orgs)
    
    edge_index = data['inv', 'ii', 'org'].edge_index

    ii = list(range(x0.shape[0]))
    random.shuffle(ii)

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
        
        _, edge_index_, _, _ = k_hop_subgraph(torch.LongTensor(np.array(ii)), 1, edge_index, directed = False, flow = 'target_to_source')
        org_ii_ = list(set(list(edge_index_[1].numpy())))
                
        org_ii_ = list(set(org_ii_).difference(test_orgs))

        d_sub = dict()
        d_sub['inv'] = torch.LongTensor(np.array(ii))
        d_sub['org'] = torch.LongTensor(np.array(org_ii_))
        
        batch = data.subgraph(d_sub)
        
        _ = []
        for ii_ in org_ii[:4]:
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
    
    f_test = f1_score(y.flatten(), np.round(expit(Y_pred).flatten()))
    logging.info('got eval(test) f1 of {}...'.format(f_test))
    
    Y_pred = torch.FloatTensor(expit(Y_pred)).to(device)
    y = torch.FloatTensor(y).to(device)
    
    loss = criterion(Y_pred, y).item()
    logging.info('got eval(test) loss of {}...'.format(loss))
    
    with open('results.txt', 'a') as f:

        f.write(f"LR: {args.lr} Weight decay: {args.weight_decay} K: {args.k} batch_size :{args.batch_size}.\n")
        f.write(f"got eval(test) f1 of {f_test}...\n")
        f.write(f"got eval(test) loss of {loss}...\n\n")
        
    avg_val_loss = np.mean(valplotloss)
    logging.info(f'Average validation loss after all epochs: {avg_val_loss}')

    return np.mean(val_losses)

    
def objective(trial):
    # Suggest values for the hyperparameters
    lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)

    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
    k = trial.suggest_int('k', 4, 25)
    batch_size = trial.suggest_int('batch_size', 6, 256)

    args = argparse.Namespace(lr=lr, weight_decay=weight_decay, k=k, batch_size=batch_size, ifile="None.npz")
    #args = argparse.Namespace(lr=lr, k=k, batch_size=batch_size, ifile="None.npz")

    return run(args)


def main():
    # Set up the logger
    logging.basicConfig(level=logging.INFO)

    # Create a new study or load an existing study
    study = optuna.create_study(direction="minimize")
    # Optimize the study
    study.optimize(objective, n_trials=30)


if __name__ == '__main__':
    main()


