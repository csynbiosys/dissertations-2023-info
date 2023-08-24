from bs4 import BeautifulSoup
from datetime import datetime

import pandas as pd
import time
import requests
import pickle

import data_preprocessing
import text_similarity_model


# 创建文件，添加表头
columns = ['name', 'location', 'stage', 'type', 'category', 'close_time',
           'create_time', 'employees']

def filter_UK_org(dataframe):
    companies = []
    # 遍历每一行
    count = 0
    for row in dataframe.itertuples(index=False):
        # print(row[4])
        # if 'United Kingdom' in row[4][1]:
        companies.append([count, row[0], row[4]])
        count = count + 1
    return companies


def find_target_company(search_org, companies, model):
    max_score = 0
    index = 0
    for i in range(0, len(companies)):
        name = companies[i].find('a').text
        name = name.replace('LIMITED', '')
        name = name.replace('LTD', '')
        name = name.strip()
        score = text_similarity_model.calculate_similarity(search_org, name, model)
        if score > max_score:
            max_score = score
            index = i
    return companies[index]


def search_UK_org():
    df = pd.read_csv('./data/org_info.csv')
    companies = filter_UK_org(df)
    model = text_similarity_model.load_model('GBR')
    for company in companies:
        url_keyword = company[1].replace(' ', '+')

        url = "https://find-and-update.company-information.service.gov.uk/search/companies?q=" + url_keyword
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        searched_companies = soup.find_all('li', 'type-company')
        if len(searched_companies) > 0:
            searched_org = find_target_company(company[1], searched_companies, model)
            company_url = 'https://find-and-update.company-information.service.gov.uk' + searched_org.find('a')['href']
            print(searched_org.find('a').text.strip())
            page = requests.get(company_url)
            soup = BeautifulSoup(page.content, 'html.parser')
            info = soup.find_all('dd', 'text data')
            if len(info) >= 3:
                address = info[0].text.strip()
                status = info[1].text.strip()
                company_type = info[2].text.strip()
            elif len(info) >= 2:
                address = info[0].text.strip()
                status = info[1].text.strip()
            elif len(info) >= 1:
                address = info[0].text.strip()


            spans = soup.find_all('span', id=lambda value: value and 'sic' in value)
            categories = []
            for span in spans:
                categories.append(span.text.strip())

            cessation_date = soup.find(id='cessation-date-label')
            cessation_date = cessation_date.text.strip() if cessation_date is not None else "None"
            creation_date = soup.find(id='company-creation-date')
            creation_date = creation_date.text.strip() if creation_date is not None else "None"

            # People
            page = requests.get(company_url + '/officers')
            soup = BeautifulSoup(page.content, 'html.parser')
            people_list = soup.find_all(class_=lambda value: value and 'appointment-' in value)
            officers = []
            for people in people_list:
                # Role
                role = people.find(id=lambda value: value and 'officer-role-' in value)
                role = role.text.strip() if role else 'None'
                # Birthdate
                birth_date = people.find(id=lambda value: value and 'officer-date-of-birth-' in value)
                birth_date = birth_date.text.strip() if birth_date else 'None'
                # Appointed time
                appointed_time = people.find(id=lambda value: value and 'oofficer-appointed-on-' in value)
                appointed_time = appointed_time.text.strip() if appointed_time else 'None'
                # Resigned time
                resigned_time = people.find(id=lambda value: value and 'officer-resigned-on-' in value)
                resigned_time = resigned_time.text.strip() if resigned_time else 'None'
                # Country
                country = people.find(id=lambda value: value and 'officer-country-of-residence-' in value)
                country = country.text.strip() if country else 'None'
                # Occupation
                occupation = people.find(id=lambda value: value and 'officer-occupation-' in value)
                occupation = occupation.text.strip() if occupation else 'None'
                officers.append([role, birth_date, appointed_time, resigned_time, country, occupation])
            org = pd.Series([company[1], address, status, company_type, categories,
                             cessation_date, creation_date, officers], columns)
            with open('./attributes.pickle', 'rb') as file:
                attributes = pickle.load(file)
            df.iloc[company[0]] = data_preprocessing.search_merge(df.iloc[company[0]], org, 3, attributes)
            df.to_csv('./data/org_info_searched.csv', encoding='utf-8', index=False)
