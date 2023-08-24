import news_company_scrape
import website_company_scrape
import company_search
import data_preprocessing
import org_submit
import evaluation

import os
import fnmatch
import pandas as pd
import pickle
import requests

import time
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger


def find_path(dir):
    fileList = []
    # 遍历项目中的所有文件和文件夹
    for root, dirs, files in os.walk(dir):
        for file in files:
            # 使用fnmatch模块的fnmatch函数进行文件名匹配
            if fnmatch.fnmatch(file, '*info*'):
                # 打印匹配到的文件路径
                file_path = os.path.join(root, file)
                # file_path.replace('/', '//')
                print(file_path)
                fileList.append(file_path)
    return fileList

def merge_info(direct):
    fileList = find_path(direct)
    columns = data_preprocessing.attributes
    data = pd.DataFrame([], columns=columns)
    for file in fileList:
        df = pd.read_csv(file)
        data = pd.concat([data, df])

    data_unique = data.drop_duplicates(subset=['name'])
    data_unique.to_csv('./data/org_info.csv', encoding='utf-8', index=False)

def spider():
    # extract org entity from the news website
    news_company_scrape.extract_techcrunch_article_dynamic()
    news_company_scrape.extract_techcrunch('./data/news_company/techcrunch_links.csv', './data/news_company'
                                                                                       '/techcrunch_articles.csv')
    news_company_scrape.extract_orgs('./data/news_company/techcrunch_articles.csv')

    # extract org entity from the orgs website
    website_company_scrape.extract_stratupers_dynamic()

    merge_info('/Users/northarbour/Desktop/Bishe/Jiayan_Liu_find/data')
    company_search.search_UK_org()
    org_submit.submit_org()
    pass

scheduler = BackgroundScheduler()
# 每隔一定时间执行一次爬虫，可以使用 cron 表达式或者间隔秒数来设置
scheduler.add_job(spider, IntervalTrigger(hours=1))
# 启动定时任务
scheduler.start()

while True:
    time.sleep(1)
    # 防止程序退出
    try:
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        scheduler.shutdown()
