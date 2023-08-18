#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import matplotlib.pyplot as plt
import warnings
import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, Linear
import torch.nn.functional as F
from torch_geometric.data import Data, GraphSAINTRandomWalkSampler, NeighborSampler, GraphSAINTEdgeSampler, DataLoader, download_url, extract_zip, HeteroData
from torch_geometric.nn import SAGEConv, GAE, RGCNConv, Node2Vec, FastRGCNConv, to_hetero, knn_graph, knn
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, average_precision_score, accuracy_score
from tqdm.notebook import tqdm
import torch_geometric.transforms as T
from model import Model
import tqdm
import optuna
import numpy as np

warnings.filterwarnings('ignore')


def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "data.pt")
    
    parser.add_argument("--lr", default = "1e-04")
    parser.add_argument("--weight_decay", default = "1e-05")
    parser.add_argument("--batch_size", default = "128")
    

    parser.add_argument("--odir", default = "None")
    parser.add_argument("--model_path", default = "gsage.pt")
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
    data = torch.load(args.ifile)
    print(data)
    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=2.0,
        add_negative_train_samples=False,
        edge_types=('startup', 'invested_by', 'investor'),
        rev_edge_types=('investor', 'rev_invested_by', 'startup'), 
    )
    train_data, val_data, test_data = transform(data)
    
    edge_label_index = train_data['startup', 'invested_by', 'investor'].edge_label_index
    edge_label = train_data['startup', 'invested_by', 'investor'].edge_label
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[20, 10],
        neg_sampling_ratio=2.0,
        edge_label_index=(('startup', 'invested_by', 'investor'), edge_label_index),
        edge_label=edge_label,
        batch_size=int(args.batch_size),
        shuffle=True,
    )
    
    edge_label_index = val_data['startup', 'invested_by', 'investor'].edge_label_index
    edge_label = val_data['startup', 'invested_by', 'investor'].edge_label
    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=[20, 10],
        edge_label_index=(('startup', 'invested_by', 'investor'), edge_label_index),
        edge_label=edge_label,
        batch_size=3 * int(args.batch_size),
        shuffle=False,
    )
    
    
    model = Model(hidden_channels=64, data= data)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_val_loss = float('inf')
    patience = 20  
    patience_counter = 0
    
    train_losses, train_accuracies, train_rocs = [], [], []
    val_losses, val_accuracies, val_rocs = [], [], []
    
    train_f1_scores = []
    val_f1_scores = []
    
    for epoch in range(1, 500):
        total_loss = total_examples = correct_preds = 0
        preds = []
        ground_truths = []
    
        model.train()  
        for sampled_data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            sampled_data = sampled_data.to(device)
            pred = model(sampled_data)[0]
             
            ground_truth = sampled_data['startup', 'invested_by', 'investor'].edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            preds.append(pred)
            ground_truths.append(ground_truth)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
             
        pred = torch.cat(preds, dim=0).detach()
        ground_truth = torch.cat(ground_truths, dim=0).detach()

       
        pred = (torch.sigmoid(pred) >= 0.5)
        
    

        
        loss_train = total_loss / total_examples
        train_losses.append(loss_train)

        f1_train = f1_score(ground_truth.cpu(), pred.cpu())
        train_f1_scores.append(f1_train)
    
        print(f"Epoch: {epoch:03d}, Training Loss: {loss_train:.4f},  Training F1 Score: {f1_train:.4f}")
     
        model.eval()  
        total_loss = total_examples = correct_preds = 0
        preds = []
        ground_truths = []
    
        with torch.no_grad():  
            for sampled_data in tqdm.tqdm(val_loader):
                sampled_data = sampled_data.to(device)  
                pred = model(sampled_data)[0]
                   
                ground_truth = sampled_data['startup', 'invested_by', 'investor'].edge_label
                loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
                preds.append(pred)
                ground_truths.append(ground_truth)
                total_loss += float(loss) * pred.numel()
                total_examples += pred.numel()
                
                
            loss_val = total_loss / total_examples
            
        pred = torch.cat(preds, dim=0).detach()
        ground_truth = torch.cat(ground_truths, dim=0).detach()

    #     roc_auc_val = roc_auc_score(ground_truth.numpy(), pred.numpy())
        pred = (torch.sigmoid(pred) >= 0.5)
        
    #     print(np.unique(ground_truth.numpy()), np.unique(pred.numpy()))

        
        val_losses.append(loss_val)

        f1_val = f1_score(ground_truth.cpu(), pred.cpu())
        val_f1_scores.append(f1_val)
    
        print(f"Epoch: {epoch:03d}, Validation Loss: {loss_val:.4f}, Validation F1 Score: {f1_val:.4f}")
        if loss_val < best_val_loss:
                best_val_loss = loss_val
                patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break
            
            
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Plot training and validation loss per epoch
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title(f'Training & Validation Loss per Epoch\nLR: {args.lr}, Weight Decay: {args.weight_decay}, Batch Size: {args.batch_size}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot training and validation accuracy per epoch
    ax2.plot(train_f1_scores, label='Training F1')
    ax2.plot(val_f1_scores, label='Validation F1')
    ax2.set_title(f'Training & Validation F1 per Epoch\nLR: {args.lr}, Weight Decay: {args.weight_decay}, Batch Size: {args.batch_size}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # Show the plots
    plt.tight_layout()
  
    plt.savefig(f"plot_gsageemb_and_fea_{args.lr}_{args.weight_decay}_{args.batch_size}.png")
    plt.show()

    PATH = f"model_gsageemb_and_fea__{args.lr}_{args.weight_decay}_{args.batch_size}.pth"
    torch.save(model.state_dict(), PATH)
    
    
    model  = Model(hidden_channels=64, data= data).to(device)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    
    
    edge_label_index = test_data['startup', 'invested_by', 'investor'].edge_label_index
    edge_label = test_data['startup', 'invested_by', 'investor'].edge_label
    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=[20, 10],
    #     neg_sampling_ratio=2.0,
        edge_label_index=(('startup', 'invested_by', 'investor'), edge_label_index),
        edge_label=edge_label,
        batch_size=3 * int(args.batch_size),
        shuffle=False,
    )
    
    preds = []
    ground_truths = []
    for sampled_data in tqdm.tqdm(test_loader):
        sampled_data = sampled_data.to(device)
        with torch.no_grad():
            pred = model(sampled_data)[0]
            preds.append(pred)
            ground_truths.append(sampled_data['startup', 'invested_by', 'investor'].edge_label)
            loss = F.binary_cross_entropy_with_logits(pred, sampled_data['startup', 'invested_by', 'investor'].edge_label)
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
            
            
    loss_test = total_loss / total_examples
    pred = torch.cat(preds, dim=0).cpu()
    ground_truth = torch.cat(ground_truths, dim=0).cpu()
    
    pred = (torch.sigmoid(pred) >= 0.5)
    
    
    f1_test = f1_score(ground_truth.numpy(), pred.numpy())
    
    print()
    print(f"Test F1 Score: {f1_test:.4f}")
    
    with open('results.txt', 'a') as f:
        
        f.write(f"LR: {args.lr}, Weight Decay: {args.weight_decay},  Batch Size: {args.batch_size}\n")
        f.write(f"got eval(test) f1 of {f1_test}...\n")
        f.write(f"got eval(test) loss of {loss_test}...\n\n")
        
    return np.mean(val_losses)
            
def objective(trial):
    # Suggest values for the hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-1)
    
    batch_size = trial.suggest_int('batch_size', 32, 256)



    args = argparse.Namespace(lr=lr, weight_decay=weight_decay,  batch_size=batch_size, ifile="data.pt")
    #args = argparse.Namespace(lr=lr, k=k, batch_size=batch_size, ifile="None.npz")



    return run(args)


def main():
    # Set up the logger
    logging.basicConfig(level=logging.INFO)
    # Create a new study or load an existing study
    study = optuna.create_study(direction="minimize")
    # Optimize the study
    study.optimize(objective, n_trials=15) 
        
if __name__ == '__main__':
    main()     
        
        