# Auction Price Prediction

## Data Collection
### Environment 
```bash pip3 install -r requirements.txt ```
### Crawl Data from Different Websites 
1. ***scotchwhiskyauctions.com***
- run codes
	- run `mkdir data; mkdir data_csv` in command line
	- get distillery names `dist_names.json` from `https://whiskymate.net/the-distillery-list/`: `python3 crawl_auction.py --run ex-name`
	- multi-thread crawl auction records from {st=0} and {ed=100} and get json file: `python3 crawl_auction.py --run crawl --st 0 --ed 100`
		- st or ed stand for one auction (totally 147 auctions and **60w+** object records)
		- one dir includes multiple object json files(20 object per page(file))
	- multi-process&thread crawl(run multiple crawl programs separately simultaneously): `bash crawl.sh`
	- convert single json file to csv: `python3 crawl_auction.py --run parse --st 0 --ed 100`
	- merge all csv files and post data processing using regex, statistics: `run parse_data.ipynb and post_process_1.ipynb in order` and we get total processed file `data_all_v2.csv`
		- remove non-meaningful columns
		- extract value from original text
		- remove rows with multiple nan value
		- fill values using other columns e.g. `age = bottled - distilled`
		- remove and clip extreme values
- crawled data demo

> other statistics and visualization for dataset in output of above Jupyter notebook.

<img src="/Users/gmm/Library/Application Support/typora-user-images/image-20230720192053804.png" alt="image-20230720192053804" style="zoom:50%;" />



2. ***whiskyauction.com***

- run codes
	- run `cd other_websites; mkdir data` in command line
	- using `dist_names.json` from above process
	- multi-thread crawl auction records from {st=0} and {ed=10000} and get json file: `python3 crawl_auction.py --st 0 --ed 10000`
		- st or ed constraint the range of object records (totally **49w+** object records)
		- maybe reset connection by peer (ban IP or other anti-crawl in target website)
	- multi-process&thread crawl(run multiple crawl programs separately simultaneously): `bash crawl.sh`
	- convert single json file to csv: `python3 crawl_auction.py --run parse --st 0 --ed 10000`
	- TODO: post process for csv files similar to crawling process of above website.

- crawled data demo 

