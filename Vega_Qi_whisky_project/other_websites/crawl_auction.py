import requests
from bs4 import BeautifulSoup
import csv
import os
import json 
import threading 
import pandas as pd 
from multiprocessing.pool import ThreadPool
import argparse
import numpy as np
# Define the base URL
user_agent = {'User-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OSX 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0Safari/537.36',
              "Cookie":'SSESSada4ea5ba2305e4ea206db7f05c8d739=O6y%2C-o9cZhhu1jwNCEnbq%2COnBxCXzT6dXEimznjC6E4IUmwR; cookie-agreed-version=1.0.0; cookie-agreed=2'}
total_num = 490000 # 492268后面都是空的  
KEYS = ['hist_ids','hist_times','hist_prices','hist_codes'] + \
['condition description', 'series', 'ean', 'back label code', 'category', 'alcohol strength', 'packing type', 'addons', 'front label code', 'description', 'auction category', 'bottled for', 'handfilled date', 'age', 'bottom code', 'lot number', 'filling code', 'vp sort description', 'distillery', 'finishing', 'abnumber', 'maturing', 'importer', 'tax banderole code', 'bottler label', 'bottle number', 'distilled date', 'owner', 'name', 'filling level', 'bottled date', 'cask no.', 'content quantity', 'count of bottle', 'brandname', 'release number', 'bottle code', 'whisky type label']
# print(len(KEYS))
POOL_NUM = 20 
DATA_DIR = 'data'
step_len = 300

def get_ids(i:int):
    all_ids = []
    r = requests.post('https://whiskyauction.com/wac/whiskyBrowserData', 
        data={'start':i,'length':step_len,'currentsort':'auction_prose','currentview':'table'},
        headers=user_agent)
    if r.status_code != 200:
        print('error with',str(i))
        return None 
    all_ids += [{'id':dd['id'],'legacy_id':dd['legacy_id'],'item_id':dd['item_id'],'legacy_item_name':dd['legacy_item_name']} for dd in json.loads(r.text)['data']]    
    print('now processing for',i,len(all_ids))
    return all_ids 

def get_all_ids():
    pool = ThreadPool(POOL_NUM)
    i_s = list(np.arange(0,total_num,step_len))
    col_ids = [v for v in pool.map(get_ids,i_s) if v is not None]
    all_ids = []
    for ids in col_ids:
        all_ids += ids 
    print('collected ids num is',len(all_ids))
    pool.close()
    json.dump(sorted(all_ids,key=lambda x:int(x['id'])),open('all_ids.json','w'))

def parse_obj(id:str):
    try:
        r = requests.get(f'https://whiskyauction.com/item/{id}',headers=user_agent)
    except Exception as e:
        print(e)
        return None 
    if r.status_code != 200: return None
    soup = BeautifulSoup(r.content, 'html.parser')
    aa = soup.find('table',class_='detailpage-detaildata')
    data = {}
    data['id'] = id 
    for tr in aa.find_all('tr'):
        v = tr.find_all('td')
        kk = v[0].text.strip().lower().replace(':','')
        vv = v[1].text.strip().lower().replace('\n','')
        data[kk] = vv
    try:
        r = requests.get(f'https://whiskyauction.com/wac/rest/history/{id}?_format=json&action=bh',headers=user_agent)
    except Exception as e:
        print(e)
        return None 
    if r.status_code != 200: return None 
    rr = json.loads(r.text)
    hists = []
    for v in rr['ResultObject']['BottleResults']:
        val = {}
        val['id'] = v['Id']
        val['time'] = v['AuctionNumber']
        val['price'] = v['Result']
        # val['bottlecode'] = v['BottleCode']['BottleCode'] if 'BottleCode' in v else None
        hists.append(val)
    data['hist_ids'] = [v['id'] for v in hists]
    data['hist_times'] = [v['time'] for v in hists]
    data['hist_prices'] = [v['price'] for v in hists]
    # data['hist_codes'] = [v['bottlecode'] for v in hists]
    print('now processing finished for',id)
    # data['history'] = hists
    return data 

def get_key(id:str):
    r = requests.get(f'https://whiskyauction.com/item/{id}',headers=user_agent)
    print(f'https://whiskyauction.com/item/{id}')
    if r.status_code != 200: 
        print(id)
        return None
    soup = BeautifulSoup(r.content, 'html.parser')
    aa = soup.find('table',class_='detailpage-detaildata')
    keys = []
    for tr in aa.find_all('tr'):
        v = tr.find_all('td')
        kk = v[0].text.strip().lower().replace(':','')
        keys.append(kk)

    # data['history'] = hists
    return keys 


def crawl_all(st,ed):
    if os.path.exists(f'data/obj_{st}_{ed}.json'):
        return 
    ids = json.load(open('all_ids.json','r'))
    ids = sorted([int(v['item_id']) for v in ids[st:ed]],reverse=True)
    pool = ThreadPool(POOL_NUM)
    infos = [v for v in pool.map(parse_obj,ids) if v is not None]
    infos = sorted(infos,key=lambda x:int(x['id']))
    print(len(infos))
    json.dump(infos,open(os.path.join(DATA_DIR,f'obj_{st}_{ed}.json'),'w'),indent=4)
    pool.close()


STEP_SIZE = 200 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',choices=['crawl','parse','getids'])
    parser.add_argument('--st',type=int)
    parser.add_argument('--ed',type=int)
    args = parser.parse_args()
   
    if args.run == 'crawl':
        assert (args.ed - args.st) % 200 == 0
        for st in range(args.st,args.ed,STEP_SIZE):
            print('start for',st)
            crawl_all(st,st+STEP_SIZE)
    elif args.run == 'getids':
        # get ids
        get_all_ids()
    elif args.run == 'parse':
        all_recs = []
        fnames = sorted([fname for fname in os.listdir('./data') if (fname.endswith('json') and fname.startswith('obj'))],key=lambda x:int(x.split('_')[1]))[args.st:args.ed]
        print('the number of fnames are',len(fnames))
        for ii,fname in enumerate(fnames):
            print(fname)
            try:
                recs = json.load(open(os.path.join('data',fname),'r'))
            except:
                print('error in read', fname)
                continue
            recs = [{k:(v.replace('\r',' ').replace('\n',' ').strip().lower() if isinstance(v,str) else v) for k,v in rec.items() if 'hist' not in k} for rec in recs]    
            all_recs += recs 
            print(f'now len of recs are {len(all_recs)} for {ii} in {args.ed-args.st}')
        df = pd.DataFrame(all_recs)
        df.to_csv(f'data/data_whiskyauction_{args.st}_{args.ed}.csv',index=False)

        
    
    # get cols 
    # ids = [int(v['item_id']) for v in json.load(open('all_ids.json','r'))]
    # ids = sorted(ids,reverse=True)[:1000]
    # # print(len(ids),len(np.unique(ids)),np.asarray(ids).min(),np.asarray(ids).max(),ids[:100],ids[-100:])
    # all_keys = set()
    # pool = ThreadPool(POOL_NUM)
    # infos = [v for v in pool.map(get_key,ids) if v is not None]
    # for info in infos:
    #     all_keys = all_keys | set(info)
    # pool.close()


