#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import pandas as pd
import numpy as np
import pickle

from sklearn.decomposition import PCA

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--k", default = "512")
    parser.add_argument("--n_sample", default = "25000")
    
    parser.add_argument("--ofile", default = "None")

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

def main():
    args = parse_args()
    
    X = np.load('data.npz')['x0']

    ii = np.where(np.var(X, axis = 0) > 0)[0]
    X = X[:,ii]
    Xm = np.mean(X, axis = 0).reshape(1, -1)
    
    iis = np.random.choice(range(X.shape[0]), 25000, replace = False)
    X = X[iis, :]

    X = X - Xm

    pca = PCA(int(args.k))
    pca.fit(X)
    logging.info('at dim of {} have explained variance of {}...'.format(args.k, np.sum(pca.explained_variance_ratio_)))

    X = np.load('data.npz')['x0'][:,ii]
    X = X - Xm
    
    # transform
    X = pca.transform(X)

    # normalize
    x0 = (X - np.mean(X, axis = 0).reshape(1, -1)) / (np.std(X, axis = 0).reshape(1, -1))
    
    
    X = np.load('data.npz')['x1']
    ii = np.where(np.var(X, axis = 0) > 0)[0]
    np.save('variance_mask_x1.npy', ii)
    X = X[:,ii]
    Xm = np.mean(X, axis = 0).reshape(1, -1)
    np.save('Xm_x1.npy', Xm)
    
    iis = np.random.choice(range(X.shape[0]), 25000, replace = False)
    X = X[iis, :]
    
    X = X - Xm

    pca = PCA(int(args.k))
    pca.fit(X)
    logging.info('at dim of {} have explained variance of {}...'.format(args.k, np.sum(pca.explained_variance_ratio_)))

    X = np.load('data.npz')['x1'][:,ii]

    X = X - Xm
    
    # transform
    X = pca.transform(X)
    np.save('mean_pca_x1.npy', np.mean(X, axis=0))
    np.save('std_pca_x1.npy', np.std(X, axis=0))
    
    # normalize
    x1 = (X - np.mean(X, axis = 0).reshape(1, -1)) / (np.std(X, axis = 0).reshape(1, -1))
    
    # save
    np.savez(args.ofile, x0 = x0, x1 = x1)
    # Save the model to disk
    
    filename = 'pca_model.pkl'
    pickle.dump(pca, open(filename, 'wb'))
    
    # ${code_blocks}

if __name__ == '__main__':
    main()


