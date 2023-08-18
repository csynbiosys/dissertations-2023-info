#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import torch
import json

# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ofile", default = "data.npz")
    parser.add_argument("--ofile_df", default = "test_startups_r.csv")
    parser.add_argument("--ofile_in", default = "investor_name_dict.json")
    parser.add_argument("--ofile_sn", default = "startup_name_dict.json")
    parser.add_argument("--ofile_sm", default = "startup_map.json")
    parser.add_argument("--ofile_im", default = "investor_map.json")

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


    df = pd.read_csv('investor_startup_rel_freq.csv')
    df_investor = pd.read_csv('investor_features_n_embedding.csv')
    df_startup = pd.read_csv('startup_features_n_text_embedding.csv')

    gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
    
    # Split data based on the org_uuid group
    train_idx, test_idx = next(gss.split(df, groups=df['org_uuid']))
    
    df_train = df.iloc[train_idx]
    df_test_startups = df.iloc[test_idx]
    
    # Get unique investor_uuids in each set
    investor_train = set(df_train['investor_uuid'])
    investor_test = set(df_test_startups['investor_uuid'])
    
    # Find exclusive investors in df_test_startups
    exclusive_investors_test = investor_test - investor_train
    
    # Move the exclusive investors from df_test_startups to df_train
    for investor in exclusive_investors_test:
        move_rows = df_test_startups[df_test_startups['investor_uuid'] == investor]
        df_train = pd.concat([df_train, move_rows])
        df_test_startups = df_test_startups.drop(move_rows.index)

    df_test_startups = pd.merge(df_test_startups[['org_uuid','org_name','investor_uuid','investor_name']],df_startup,how='left', left_on='org_uuid',right_on='org_uuid')
    df_test_startups.to_csv(args.ofile_df)
    
    df_investor.drop('Unnamed: 0',inplace=True,axis=1)
    df_startup.drop('Unnamed: 0',inplace=True,axis=1)

    df_investor = pd.merge(df['investor_uuid'],df_investor,how='left', left_on='investor_uuid',right_on='uuid')
    df_startup = pd.merge(df['org_uuid'],df_startup,how='left', left_on='org_uuid',right_on='org_uuid')



    investor_features_df = df_investor.loc[df_investor['investor_uuid'].drop_duplicates().index].drop(columns=['investor_uuid','uuid'])
    org_features_df = df_startup.loc[df_startup['org_uuid'].drop_duplicates().index].drop(columns=['org_uuid','description', 'embeddings'])

    investor_features_df.fillna(0,inplace=True)
    org_features_df.fillna(0,inplace=True)

    x0 = np.array(investor_features_df.values, dtype = np.float32)
    x1 = np.array(org_features_df.values, dtype = np.float32)

    unique_startup_id = df['org_uuid'].unique()
    unique_startup_id = pd.DataFrame(data={
        'startupId': unique_startup_id,
        'mappedID': pd.RangeIndex(len(unique_startup_id)),
    })
    print("Mapping of startups to consecutive values:")
    print("==========================================")
    print(unique_startup_id.head())
    print()
    unique_investor_id = df['investor_uuid'].unique()
    unique_investor_id = pd.DataFrame(data={
        'investorId': unique_investor_id,
        'mappedID': pd.RangeIndex(len(unique_investor_id)),
    })
    print("Mapping of investors to consecutive values:")
    print("===========================================")
    print(unique_investor_id.head())
    investment_startup_id = pd.merge(df['org_uuid'], unique_startup_id,
                                left_on='org_uuid', right_on='startupId', how='left')
    investment_startup_id = torch.from_numpy(investment_startup_id['mappedID'].values)
    investment_vc_id = pd.merge(df['investor_uuid'], unique_investor_id,
                                left_on='investor_uuid', right_on='investorId', how='left')
    investment_vc_id = torch.from_numpy(investment_vc_id['mappedID'].values)
    edge_index_investor_to_startup = torch.stack([investment_startup_id,investment_vc_id], dim=0)

    #assert edge_index_investor_to_startup.size() == (2, 354648)
    print()
    print("Final edge indices pointing from investors to startups:")
    print("=================================================")
    print(edge_index_investor_to_startup)

    startup_map = unique_startup_id.set_index('startupId')['mappedID'].to_dict()
    investor_map = unique_investor_id.set_index('investorId')['mappedID'].to_dict()
    
    org_name_dict = dict(zip(df_train['org_uuid'].values, df_train['org_name'].values))
    investor_name_dict = dict(zip(df_train['investor_uuid'].values, df_train['investor_name'].values))
    
    startup_name_dict = unique_startup_id['startupId'].map(org_name_dict).to_dict()
    investor_name_dict = unique_investor_id['investorId'].map(investor_name_dict).to_dict()

    with open(args.ofile_in, 'w') as f:
        
        json.dump(investor_name_dict, f)
    
    with open(args.ofile_sn , 'w') as f:
        
        json.dump(startup_name_dict, f)
    
    
    with open(args.ofile_im , 'w') as f:
        
        json.dump(investor_map, f)
    
    
    with open(args.ofile_sm , 'w') as f:
        
        json.dump(startup_map, f)

    np.savez(args.ofile, x0 = x0, x1 = x1, edge_index = edge_index_investor_to_startup.numpy().astype(np.int32))
    # ${code_blocks}

if __name__ == '__main__':
    main()


