import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from datetime import datetime


def extract_techcrunch(linkfile, articlefile):
    links = pd.read_csv(linkfile, index_col=False)
    columns = ['title', 'author', 'date', 'body']
    article = []
    article = pd.DataFrame(article, columns=columns)
    article.to_csv(articlefile, encoding='utf-8', index=False)
    data = pd.DataFrame(columns=columns)

    for i in range(0, len(links)):
        try:
            url = links.iloc[i][0]
            page = requests.get(url)
            soup = BeautifulSoup(page.content, 'html.parser')

            title = soup.find("h1", class_="article__title").text
            author = soup.find("div", class_="article__byline")
            author = author.find("a").text.strip()
            date = re.findall(r'[0-9]{4}/[0-9]{2}/[0-9]{2}', url)[0]
            body = soup.find("div", class_="article-content")

            # remove all links to other articles which are part of body
            other_link = body.find("div", class_="embed breakout")
            while other_link != None:
                body.find("div", class_="embed breakout").decompose()
                other_link = body.find("div", class_="embed breakout")
            body = body.text

            article = pd.DataFrame([[title, author, date, body]], columns=columns)
            data = pd.concat([data, article], ignore_index=True)
            if i % 10 == 0:
                data.to_csv(articlefile, encoding='utf-8', index=False, mode='a', header=False)
                data = pd.DataFrame(columns=columns)
        except:
            print("error extract turn is " + str(i))
            pass


def extract_thenextweb(linkfile, articlefile):
    ua = UserAgent()
    header = {'User-Agent': str(ua.chrome)}
    links = pd.read_csv(linkfile, index_col=False)
    columns = ['title', 'author', 'date', 'body']
    article = []
    article = pd.DataFrame(article, columns=columns)
    article.to_csv(articlefile, encoding='utf-8', index=False)
    data = pd.DataFrame(columns=columns)

    for i in range(0, len(links)):
        try:
            url = links.iloc[i][0]
            page = requests.get(url, headers=header)
            soup = BeautifulSoup(page.content, 'html.parser')

            title = soup.find("h1", class_="c-header__heading").text.strip()
            author = soup.find("span", class_="c-article__authorName latest").text.strip()

            date = soup.find('time')['datetime'].split()
            month = datetime.strptime(date[0], "%B").month
            if month < 10:
                month = '0' + str(month)
            else:
                month = str(month)
            date = date[2] + '/' + month + '/' + date[1][:-1]

            if soup.find("div", class_="inarticle-wrapper"):
                soup.find("div", class_="inarticle-wrapper").extract()
            body = soup.find("div", class_="c-richText c-richText--large").text

            article = pd.DataFrame([[title, author, date, body]], columns=columns)
            data = pd.concat([data, article], ignore_index=True)
            if i % 10 == 0:
                data.to_csv(articlefile, encoding='utf-8', index=False, mode='a', header=False)
                data = pd.DataFrame(columns=columns)
        except:
            print("error extract turn is " + str(i))
            pass


def extract_newsCrunchbase(linkfile, articlefile):
    ua = UserAgent()
    header = {'User-Agent': str(ua.chrome)}
    links = pd.read_csv(linkfile, index_col=False)
    columns = ['title', 'author', 'date', 'body']
    article = []
    article = pd.DataFrame(article, columns=columns)
    article.to_csv(articlefile, encoding='utf-8', index=False)
    data = pd.DataFrame(columns=columns)

    for i in range(0, len(links)):
        try:
            url = links.iloc[i][0]
            page = requests.get(url, headers=header)
            soup = BeautifulSoup(page.content, 'html.parser')

            title = soup.find("h1", class_="entry-title h1").text.strip()
            author = soup.find("a", class_="herald-author-name").text.strip()

            date = soup.find("span", class_="updated").text.split()
            month = datetime.strptime(date[0], "%B").month
            if month < 10:
                month = '0' + str(month)
            else:
                month = str(month)
            date = date[2] + '/' + month + '/' + date[1][:-1]

            if soup.find("div", class_="ss-inline-share-wrapper"):
                soup.find("div", class_="ss-inline-share-wrapper").extract()
            if soup.find("div", class_="wp-block-columns"):
                soup.find("div", class_="wp-block-columns").extract()
            if soup.find("div", class_="footnotes"):
                soup.find("div", class_="footnotes").extract()
            soup.find("div", class_="wp-block-cover_ blog-footer-cta").extract()
            soup.find("div", class_="herald-ad").extract()
            body = soup.find("div", class_="entry-content herald-entry-content").text

            article = pd.DataFrame([[title, author, date, body]], columns=columns)
            data = pd.concat([data, article], ignore_index=True)
            if i % 10 == 0:
                data.to_csv(articlefile, encoding='utf-8', index=False, mode='a', header=False)
                data = pd.DataFrame(columns=columns)
        except:
            print("error extract turn is " + str(i))
            pass



