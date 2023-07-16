import pandas as pd
import re
import selenium
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from datetime import datetime


def extract_kickstarter(link):
    ua = UserAgent()
    header = {'User-Agent': str(ua.chrome)}

    page = requests.get(link, headers=header)
    soup = BeautifulSoup(page.content, 'html.parser')
    print(soup.text)
    product_name = soup.find("h2", class_="type-28 type-24-md soft-black mb1 project-name").text.strip()
    fund=1
    fund_goal=1
    supporter_num=1
    industry=1
    location=1
    message_num=1
    columns = ['product_name', 'fund', 'fund_goal', 'supporter_num', 'industry', 'location', 'message_num']
    product = pd.DataFrame([[product_name, fund, fund_goal, supporter_num, industry, location, message_num]], columns=columns)
