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
BASE_URL = 'https://www.scotchwhiskyauctions.com/'
FILTER_WORD = ['Error 502','Bad Gateway']
KEYS = ['distillery','strength','his_bid_prices','his_bid_times','name','url','auction']
parsed_keys = {'bottle no', 'causeway collection', 'bottle number', 'distiled', 'edition', 'committee release', 'secret distillery #1batch', 'barrel', 'blended', 'ck', 'td-s', 'cask', 'travel exclusive edition', 'farm', 'no', 'easter elchies', 'batch', 'fine oak', 'cask strength', 'creation of a classic', 'rare cask', 'cask toast level', 'recasked', 'edition no', 'catch type', 'appleton', 'single barrel bottle number', 'dumped', 'serial number', "founder's reserve", 'batch number', 'classic cut', 'release number', 'kit number', 'punch bowl', 'barrel number', 'barley type', 'dalbeallie dram no', 'moved to sherry casks', 'bottlled', 'barreled', 'kylver release', 'pact', 'renaissance series no', 'distilled', 'certificate number', 'lot no', 'selection', 'concept no', 'compendium', 'artist no', 'bottled number', 'the renaissance', "master blender's edition", 'the home collection', 'distillery', 'botted', 'small batch', 'harmony collection', '18 year old', "balvenie founder's reserve", 'set number', 'lalique', 'trilogy part', 'the harmony collection', 'no corners to hide', 'cask numbers', 'cask selection', 'please note', 'cask no', 'lot number', 'released', 'edition series', 'feis 2021 release', 'bottles number', 'white stag', "whiskymaker's reserve no", 'cask number', 'cask type', 'home collection', 'single barrel', 'the boutique collection', 'age', 'batch no', 'warehouse', 'casktype', 'barrelled', 'quest collection', 'balvenie doublewood', 'traigh bhan', 'in cask date', 'chichibu', 'bottle numbers', 'tree number', 'winning bid', 'warehouse number', 'selection number', 'bottled', 'mc', 'doublewood', 'dream cask', 'grape', 'botttled', 'solar sytem'}
KEYS = set(KEYS) | parsed_keys
# print(len(KEYS))
fix_keys = ['bidinfo won','lotno']
RUN_MODE = 'crawl'

# {'daftmill', 'dumbarton', 'allt-a-bhainne', 'annandale', 'hanyu', 'longmorn', 'braeval', 'st|magdalene', 'millburn', 'cardhu', 'kininvie', 'ladyburn', 'hakushu', 'yamazaki', 'north|port', 'glenglassaugh', 'miyashita|shuzo', 'glenallachie', 'glenrothes', 'glen|grant', 'glenlivet', 'laphroaig', 'macduff', 'ardnamurchan', 'cameronbridge', 'glasgow', 'blair|athol', 'tormore', 'glenfarclas', 'karuizawa', 'speyburn', 'ballindalloch', 'imperial', 'invergordon', 'glentauchers', 'balmenach', 'ailsa|bay*', 'tobermory', 'glenlochy', 'harris', 'north|of|scotland/strathmore', 'teaninich', 'springbank', 'glen|albyn', 'chichibu', 'bunnahabhain', 'linkwood', 'tullibardine', 'tomatin', 'cragganmore', 'port|ellen', 'north|british', 'scapa', 'glenlossie', 'strathclyde', 'dallas|dhu', 'ardbeg', 'eden|mill', 'white|oak', 'kilchoman', 'knockando', 'glen|elgin', 'lagavulin', 'aberfeldy', 'ardmore', 'lone|wolf', 'carsebridge', 'glendullan', 'dalwhinnie', 'oban', 'fuji|gotemba', 'convalmore', 'lochside', 'tamnavulin', 'dalmunach*', 'glenfiddich', 'miltonduff', 'pulteney', 'bladnoch', 'abhainn|dearg', 'balvenie', 'glen|flagler', 'girvan', 'inverleven', 'mortlach', 'clynelish', 'caledonian', 'yoichi', 'glenury|royal', 'bowmore', 'ben|nevis', 'glen|spey', 'tamdhu', 'torabhaig', 'caol|ila', 'deanston', 'garnheath/moffat', 'strathisla', 'inchdairnie', 'strathearn', 'wolfburn', 'starlaw', 'isle|of|jura', 'inchgower', 'glencadam', 'dailuaine', 'banff', 'fettercairn', 'glen|scotia', 'benrinnes', 'glen|garioch', 'coleburn', 'auchentoshan', 'mannochmore', 'speyside', 'glenesk', 'benromach', 'glen|moray', 'glen|ord', 'auchroisk', 'glendronach', 'glenkinchie', 'aberlour', 'kinclaith', 'edradour', 'bruichladdich', 'arbikie', 'glengyle', 'strathmill', 'glen|keith', 'talisker', 'arran', 'glengoyne', 'littlemill', 'macallan', 'dornoch', 'dufftown', 'glenburgie', 'miyagikyo', 'tomintoul', 'knockdhu', 'highland|park', 'glenmorangie', 'loch|lomond', 'glenugie', 'glenturret', 'dalmore', 'balblair', 'cambus', 'port|dundas', 'rosebank', 'kingsbarns', 'benriach', 'shinshu', 'caperdonich', 'glen|mhor', 'pittyvaich', 'roseisle*', 'aultmore', 'royal|brackla', 'royal|lochnagar', 'brora', 'craigellachie'}

# Define function to get auction details
def get_detailed_pages(base_url:str):
    url = base_url + 'auctions'
    response = requests.get(url)
    assert response.status_code == 200,'network error!'
    
    soup = BeautifulSoup(response.content, 'html.parser')
    items = soup.find_all('div', class_="auctions")[0]
    pages = [base_url + item['href'] for item in items.find_all('a',class_='auction')]
    return pages

def get_sub_pages(auction_url:str):
    if not auction_url.endswith('/'): 
        auction_url+='/'
    response = requests.get(auction_url)
    if response.status_code != 200:
        return []
    soup = BeautifulSoup(response.content, 'html.parser')
    options = soup.find_all('select',{'id':'choosepage'})[0].find_all('option')
    subpages = [auction_url + f'?mode=&page={option.text.split(" ")[1]}' for option in options]
    # subpages = sorted(subpages)
    return subpages

def get_object_pages(sub_page:str):
    response = requests.get(sub_page)
    if response.status_code != 200:
        return []
    soup = BeautifulSoup(response.content, 'html.parser')
    res = soup.find_all('div',{'id':'lots'})[0].find_all('a',class_='lot')
    return [BASE_URL+aval['href'] for aval in res]

def parse_obj(object_page:str):
    info = {}
    response = requests.get(object_page)
    if response.status_code != 200:
        return None
    soup = BeautifulSoup(response.content, 'html.parser')
    info['url'] = object_page
    info['name'] = soup.select('#contentstart > h1')[0].text
    lotinfo = soup.find_all('div',class_='lotinfo')[0]
    for pval in lotinfo.find_all('p')[:2]:
        if pval.has_attr('class'):
            info[' '.join(pval['class'])] = pval.text 
    info['details'] = [v.text for v in lotinfo.find('div',class_='descr').find_all('p')] 
    if len(soup.find_all('div',class_='chartwrap'))!=0:
        info['his_bid_times'] = soup.find('input',{'id':'chartx'})['value']
        info['his_bid_prices'] = soup.find('input',{'id':'charty'})['value']
        # print(info)
        # exit(0)
    else:
        info['his_bid_times'] = []
        info['his_bid_prices'] = []
    
    return info 



def parse_json_info(data_dir,dist_names):
    print('start parsing',data_dir)
    file_path = os.path.join('data',f'auctions_{data_dir.split("-")[0].split("/")[-1]}.csv')
    if os.path.isfile(file_path):
        print('has existed for',file_path)
        return 
    all_res = []
    for f in os.listdir(data_dir):
        if not f.endswith('.json'): continue
        org_lst = json.load(open(os.path.join(data_dir,f)))
        for org in org_lst:
            dd = {k: ([] if k=='details' else None) for k in KEYS}
            dd['auction'] = org['url'].split('auctions/')[1].split('/')[0]
            dd['url'] = org['url']
            dd['name'] = org['name']
            dd['his_bid_times'] = [float(v) for v in org['his_bid_times'].split(',')] if len(org['his_bid_times'])!=0 else []
            dd['his_bid_prices'] = [float(v) for v in org['his_bid_prices'].split(',')] if len(org['his_bid_times'])!=0 else []
            dd['url_id'] = int(dd['url'].split('/')[-2 if dd['url'][-1]=='/' else -1].split('-')[0])
            assert len(dd['his_bid_prices']) == len(dd['his_bid_times'])
            for k in fix_keys:
                if k in org:
                    if ': ' not in org[k]: continue 
                    dd[org[k].split(': ')[0]] = org[k].split(': ')[1].replace('\u00a3','')
            words = org['url'].split('-')
            for det in org['details']:
                if ": " in det:
                    kk = det.split(': ')[0].replace('\n',' ').strip().lower()
                    if kk in KEYS:
                        dd[kk] = ': '.join(det.split(': ')[1:])
                if '%' in det and 'cl' in det and '/' in det and len(det.split(' '))<20:
                    dd['strength'] = det
                if 'distillery' in det.lower() and len(det.split(' '))<20 and all([wd in words for wd in det.strip().lower().split()]):
                    dd['distillery'] = det.strip().lower()                    
                    # dd['details'].append(det)
            for k,v in dd.items():
                if v is str and any([wd in v for wd in FILTER_WORD]):
                    dd[k] = None
            all_res.append(dd)
        # break 
    tmpdf = pd.DataFrame(all_res)
    print(tmpdf.shape,len(all_res))
    extract_distillery_from_name(tmpdf,dist_names)
    tmpdf.to_csv(file_path,index=False)
    print('finished parsing',file_path)

def get_cand_cols(data_dir):
    res = set()
    for f in os.listdir(data_dir):
        if not f.endswith('.json'): continue
        org_lst = json.load(open(os.path.join(data_dir,f)))
        for org in org_lst:
            for k in fix_keys + ['details']:
                if k not in org: continue 
                if k == 'details':
                    for v in org[k]:
                        if ': ' in v and len(v.split(' '))<20 and len(v.split(': ')[0].split(' '))<5:
                            kk = v.split(': ')[0].replace('\n',' ').strip().lower()
                            if kk not in res:
                                print(v)
                            res.add(kk)
                elif ': ' in org[k]:
                    kk = org[k].split(': ')[0].lower()
                    if kk not in res:
                        print(org[k])
                    res.add(kk)
                    
        # print(res)
    return res 

def get_dist_names():
	url = 'https://whiskymate.net/the-distillery-list/'
	response = requests.get(url)
	assert response.status_code == 200,'network error!'
	soup = BeautifulSoup(response.content, 'html.parser')
	items = soup.find_all('div', class_="entry-content")[0].find_all('p')
	print(len(items))
	names = set()
	for item in items:
		txt = item.text
		if ') ' in txt and txt.split(') ')[0][0] in [str(i) for i in range(10)] and 'Name' not in txt:
			names.add('|'.join(txt.split(' (')[0].split(') ')[-1].strip().replace('-',' ').lower().split(' ')))
		elif 'Name: ' in txt and ') ' in txt:
			names.add('|'.join(txt.split('Name: ')[-1].strip().replace('-',' ').lower().split(' ')))
	print('collected names len is',len(names))
	return names 

def extract_distillery_from_name(org_csv,dist_names):
	# org_csv = pd.read_csv(file_path)
	org_names = org_csv['name'] #[:100]
	res_dists = [None]*len(org_names)
	for ii,name in enumerate(org_names):
		wds = name.strip(' ').replace('-',' ').lower().split(' ')
		suc = False 
		for i in range(len(wds)):
			for dist in dist_names:
				if wds[i] == dist:
					res_dists[ii] = dist 
					suc = True 
					break 
				elif wds[i] in dist:
					j = i+1
					now_name = wds[i] 
					while j < len(wds):
						now_name = '|'.join([now_name,wds[j]])
						if now_name != dist:
							suc = False 
							break 
						elif now_name == dist:
							res_dists[ii] = now_name
							suc = True
							break 
						else:
							continue
				if suc:
					break 
			if suc: 
				break 
		if suc:
			assert res_dists[ii] is not None, name 
	extra_names = [org_names[i] for i in range(len(res_dists)) if res_dists[i] is None]
	print(len(org_names),len([v for v in res_dists if v is not None]),len(extra_names),len(np.unique([v for v in res_dists if v is not None])))
	# print(res_dists,extra_names,sep='\n')
	org_csv['extracted_distillery'] = res_dists
	# print(org_csv['extracted_distillery'])
	# org_csv.to_csv(file_path.split('.csv')[0]+'_v1'+'.csv')
	return res_dists,extra_names



NEED_DIRS = {90, 91, 92, 119, 120, 141, 142, 143, 144, 151}
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',choices=['crawl','parse','ex-name'])
    parser.add_argument('--st',type=int)
    parser.add_argument('--ed',type=int)
    args = parser.parse_args()

    if args.run == 'crawl':
        pages = get_detailed_pages(BASE_URL)[args.st:args.ed]
        # print(sorted([int(pg.split('-')[0].split('/')[-1]) for pg in pages]))
        # exit(0)
        print('the number of auctions are',len(pages),f'from {args.st} to {args.ed}')
        data_dir = './data'
        pool = ThreadPool(20)

        for i,auction_page in enumerate(pages):
            if NEED_DIRS is not None and int(auction_page.split('-')[0].split('/')[-1]) not in NEED_DIRS: continue
            print(f'current auction page is {auction_page}, and progress is {i}:{len(pages)}')
            subpages = get_sub_pages(auction_page)
            print(f'the auction with {len(subpages)} subpages')
            auction_name = auction_page.split('/')[-1] if not auction_page.endswith('/') else auction_page.split('/')[-2]
            # if os.path.exists(os.path.join(data_dir,auction_name)): continue 
            os.makedirs(os.path.join(data_dir,auction_name),exist_ok=True)
            for j,sub_page in enumerate(subpages):
                fname = os.path.join(data_dir,auction_name,f'{j}.json')
                print(fname)
                if os.path.exists(fname): continue
                # if j < args.st or j>= args.ed: continue
                print(f'current sub page progress is {j}:{len(subpages)}')
                object_pages = get_object_pages(sub_page)
                print('object nums of cur page is',len(object_pages))
                infos = [v for v in pool.map(parse_obj,object_pages) if v is not None]
                json.dump(infos,open(fname,'w'),indent=4)
                print('----------')
            # break # only test one auction 
        pool.close()
    elif args.run == 'parse':
        # print(get_cand_cols('data/189-the-144th-auction/'))
        dist_names = json.load(open('dist_names.json','r'))
        dirnames = sorted([v for v in os.listdir('data') if (not v.startswith('.')) and (os.path.isdir(os.path.join('data',v)))])
        print(dirnames)
        for ii,dd in enumerate(dirnames):
            if NEED_DIRS and (int(dd.split('-')[0]) not in NEED_DIRS): continue
            if ii < args.st or ii >= args.ed: continue
            parse_json_info(os.path.join('data',dd),dist_names)

    elif args.run == 'ex-name':
        dist_names = get_dist_names()
        json.dump(list(dist_names),open('dist_names.json','w'))
        dist_names = json.load(open('dist_names.json','r'))
        print(len(dist_names))
    	# print(dist_names)
        extract_distillery_from_name('data/189-the-144th-auction/auctions.csv',dist_names)





