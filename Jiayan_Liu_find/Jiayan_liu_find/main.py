# import attribute_scrape
import news_company_scrape
import website_company_scrape
import company_search
import data_preprocessing
import org_submit
import evaluation
import data_preprocessing

import os
import fnmatch
import pandas as pd
import pickle
import requests


if __name__ == '__main__':
    # extract org entity from the news website
    # news_company_scrape.extract_techcrunch_article()
    # news_company_scrape.extract_techcrunch('./data/news_company/techcrunch_links.csv', './data/news_company'
    #                                                                                    '/techcrunch_articles.csv')
    # news_company_scrape.extract_newsCrunchbase_article()
    # news_company_scrape.extract_newsCrunchbase('./data/news_company/newsCrunchbase_links.csv', './data/news_company'
    #                                                                                            '/newsCrunchbase_articles.csv')
    # news_company_scrape.extract_thenextweb_article()
    # news_company_scrape.extract_thenextweb('./data/news_company/thenextweb_links.csv', './data/news_company'
    #                                                                                    '/thenextweb_articles.csv')
    #
    # news_company_scrape.extract_orgs('./data/news_company/techcrunch_articles.csv')
    # news_company_scrape.extract_orgs('./data/news_company/newsCrunchbase_articles.csv')
    # news_company_scrape.extract_orgs('./data/news_company/thenextweb_articles.csv')

    # extract org entity from the orgs website
    # website_company_scrape.extract_stratupers()
    # website_company_scrape.extract_startus('united-kingdom')

    data_preprocessing.merge_info('/Users/northarbour/Desktop/Bishe/Jiayan_Liu_find/data')
    # company_search.search_UK_org()
    # org_submit.submit_org()


    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # df = pd.read_csv('./data/org_info_searched.csv')
    # unique_elements = df['location'].unique()
    # print(unique_elements)

    # df1 = pd.read_csv('/Users/northarbour/Downloads/bulk_export/organizations.csv')
    # unique_elements1 = df1['country_code'].unique()
    # print(unique_elements1)

    # df = pd.read_csv('data/org_info_searched.csv')
    # unique_elements = df['name']
    # print(unique_elements)
    # print(len(unique_elements))

    # df = pd.read_csv('./data/website_company/startUs_united-states_info.csv')
    # unique_elements = df['name']
    # print(unique_elements)
    # df = df.drop_duplicates()
    # # 保存去重后的数据回到文件
    # df.to_csv('./data/news_company/newsCrunchbase_articles.csv', index=False)


    # df = pd.read_csv('data/org_info_searched.csv')
    # target_value = 'None'
    # # 遍历每一列并统计特定值出现的次数
    # for column in df.columns:
    #     count = (df[column] != target_value).sum()
    #     # count = (df[column] == target_value).sum()
    #     print(f"在列 {column} 中特定值 {count/len(df)}")
    # evaluation.data_preprocessing()
    # evaluation.evaluation()


