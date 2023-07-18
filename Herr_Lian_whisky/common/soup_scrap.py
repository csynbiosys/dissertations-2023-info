import requests
from bs4 import BeautifulSoup
import pandas as pd


# 定义一个函数，用于获取网页的HTML内容
def get_html(url):
    response = requests.get(url)

    return response.text


# 定义一个函数，用于解析HTML内容，并提取威士忌拍卖的成交情况
def parse_html(html):
    soup = BeautifulSoup(html, 'html.parser')

    # 在这里，你需要根据whiskyhammer的网页结构，来提取威士忌拍卖的成交情况
    # 例如，你可能需要找到包含威士忌拍卖信息的HTML元素，然后提取这些元素的文本或属性
    # 你可能需要使用soup.find_all或soup.select等函数来找到这些元素
    # 你可能需要使用element.text或element['attribute']等表达式来提取这些元素的文本或属性
    elements = soup.find_all('div', class_='inner')
    results = []

    temp = {'Id': Id, 'Distillery': 'not show', 'Age': 'not show', 'Country': 'not show',
            'Bottles Produced': 'not show',
            'Region': 'not show', 'Staus': 'not show', 'Whisky Type': 'not show', 'Size': 'not show',
            'Strength': 'not show'}
    attr_name = ["Distillery", "Age", "Bottles Produced", "Whisky Type", "Size", "Region", "Country", 'Strength']
    # 假设你已经找到了包含威士忌拍卖信息的HTML元素，并将它们存储在一个名为elements的列表中
    for element in elements:
        result = {
            'Region': element.find('div', {'class': 'region'}).text,
            'Distillery': element.find('div', {'class': 'distillery'}).text,
            'Distillery Status': element.find('div', {'class': 'distillery-status'}).text,
            'Type': element.find('div', {'class': 'type'}).text,
            'Size': element.find('div', {'class': 'size'}).text,
            'Delivery Weight': element.find('div', {'class': 'delivery-weight'}).text,
            'Strength': element.find('div', {'class': 'strength'}).text,
        }
        results.append(result)

    return results


# 定义一个函数，用于获取威士忌拍卖的成交情况
def get_whisky_auction_results(url):
    html = get_html(url)
    results = parse_html(html)
    return results


# 定义一个函数，用于将结果保存到Excel文件
def save_to_excel(results, filename):
    df = pd.DataFrame(results)
    df.to_excel(filename)


# 获取威士忌拍卖的成交情况，并保存到Excel文件
results = []
auction_urls = [
    'https://www.whiskyhammer.com/auction/past/auc-92/'
    # 在这里，你需要将每个月份的拍卖URL添加到这个列表中
    # 例如：'https://www.whiskyhammer.com/auction/past/auc-92/',
    # 'https://www.whiskyhammer.com/auction/past/auc-91/',
    # ...
]
for url in auction_urls:
    html = get_html(url)
    soup = BeautifulSoup(html, 'html.parser')
    print(soup)
    lot_urls = [a['href'] for a in soup.find_all('a', {'class': 'buttonAlt'}) if 'item' in a['href']]
    for lot_url in lot_urls:
     results.extend(get_whisky_auction_results(lot_url))
save_to_excel(results, 'whisky_auction_results.xlsx')
