from bs4 import BeautifulSoup
from datetime import datetime

import pandas as pd
import time
import requests


# 创建文件，添加表头
columns = ['original_org', 'searched_org', 'address', 'status', 'company_type', 'categories', 'cessation_date', 'creation_date', 'officers']
data = pd.DataFrame([], columns=columns)
data.to_csv('companyHouse_info.csv', encoding='utf-8', index=False)

df = pd.read_csv('../news_company_scrape/startUs_info.csv')
goal_companies = []
# 遍历每一行
for row in df.itertuples(index=False):
    if 'United Kingdom' in row[7]:
        goal_companies.append([row[0], row[7]])
for i in range(0, len(goal_companies)):
    data = pd.DataFrame([], columns=columns)
    url_keyword = goal_companies[i][0].replace(' ', '+')
    print(url_keyword)

    url = "https://find-and-update.company-information.service.gov.uk/search/companies?q=" + url_keyword
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    companies = soup.find_all('li', 'type-company')
    for j in range(0, min(5, len(companies))):
        company_url = 'https://find-and-update.company-information.service.gov.uk' + companies[j].find('a')['href']
        page = requests.get(company_url)
        print(company_url)
        soup = BeautifulSoup(page.content, 'html.parser')
        searched_org=soup.find('p', 'heading-xlarge').text
        info = soup.find_all('dd', 'text data')
        if(len(info)<3):
            continue
        address = info[0].text.strip()
        status = info[1].text.strip()
        company_type = info[2].text.strip()
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

        company = pd.DataFrame([[goal_companies[i][0], searched_org, address, status, company_type, categories, cessation_date, creation_date, officers]], columns=columns)
        data = pd.concat([data, company], ignore_index=True)
        data.to_csv('companyHouse_info.csv', encoding='utf-8', index=False, mode='a', header=False)
