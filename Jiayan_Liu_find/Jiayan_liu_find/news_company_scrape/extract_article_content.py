import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

import pandas as pd
import re
from datetime import datetime
import os


def extract_techcrunch(linkfile, articlefile):
    """
        Extracts news articles from the provided links and stores them in a CSV file.

        Args:
            linkfile (str): Path to the CSV file containing the links to the news articles.
            articlefile (str): Path to the CSV file where the extracted articles will be stored.

        Returns:
            None
    """


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
    """
    Extracts content from The Next Web articles using provided links and saves it to the specified article file.

    Parameters:
    linkfile (str): Path to the CSV file containing The Next Web article links.
    articlefile (str): Path to the CSV file to save the article content.
    """
    ua = UserAgent()
    header = {'User-Agent': str(ua.chrome)}
    links = pd.read_csv(linkfile, index_col=False)
    columns = ['title', 'author', 'date', 'body']
    article = []

    if not os.path.exists(articlefile):
        # Create an empty DataFrame to store article content and save it to the file
        article = pd.DataFrame(article, columns=columns)
        article.to_csv(articlefile, encoding='utf-8', index=False)
    data = pd.DataFrame(columns=columns)

    # Iterate through the links and extract article content
    for i in range(0, 100):
        try:
            url = links.iloc[i][0]
            print(str(i)+": "+url)
            # Send a request with the specified user agent
            page = requests.get(url, headers=header)
            soup = BeautifulSoup(page.content, 'html.parser')

            # Extract article title, author, and date
            title = soup.find("h1", class_="c-header__heading").text.strip()
            author = soup.find("span", class_="c-article__authorName latest").text.strip()

            date = soup.find('time')['datetime'].split()
            month = datetime.strptime(date[0], "%B").month
            if month < 10:
                month = '0' + str(month)
            else:
                month = str(month)
            date = date[2] + '/' + month + '/' + date[1][:-1]

            # Remove certain content from the body
            if soup.find("div", class_="inarticle-wrapper"):
                soup.find("div", class_="inarticle-wrapper").extract()
            body = soup.find("div", class_="c-richText c-richText--large").text

            # Create a DataFrame to store the extracted article content
            article = pd.DataFrame([[title, author, date, body]], columns=columns)
            data = pd.concat([data, article], ignore_index=True)
            print(len(data))

            # Write data to the file and reset the DataFrame every 10 processed links
            if (i+1) % 10 == 0:
                links = links.drop(links.index[:10])
                links.to_csv(linkfile, index=False)
                data.to_csv(articlefile, encoding='utf-8', index=False, mode='a', header=False)
                data = pd.DataFrame(columns=columns)
        except:
            print("error extract turn is " + str(i))
            pass


def extract_newsCrunchbase(linkfile, articlefile):
    """
        Extracts news articles from the provided links (Crunchbase) and stores them in a CSV file.

        Args:
            linkfile (str): Path to the CSV file containing the links to the news articles.
            articlefile (str): Path to the CSV file where the extracted articles will be stored.

        Returns:
            None
    """


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
            if soup.find("div", class_="meta-tags"):
                soup.find("div", class_="meta-tags").extract()
            if soup.find("div", class_="wp-block-cover _blog-footer-cta"):
                soup.find("div", class_="wp-block-cover _blog-footer-cta").extract()

            soup.find("div", class_="herald-ad").extract()
            body = soup.find("div", class_="entry-content herald-entry-content").text
            article = pd.DataFrame([[title, author, date, body]], columns=columns)
            data = pd.concat([data, article], ignore_index=True)
            if i % 10 == 0:
                data.to_csv(articlefile, encoding='utf-8', index=False, mode='a', header=False)
                data = pd.DataFrame(columns=columns)
            print('Turn'+str(i))
        except:
            print("error extract turn is " + str(i))
            pass


def extract_venturebeat(linkfile, articlefile):
    """
    Extracts VentureBeat article content from the provided link file and saves it to the specified article file.

    Parameters:
    linkfile (str): Path to the CSV file containing VentureBeat article links.
    articlefile (str): Path to the CSV file to save the article content.
    """
    # Read the link file
    links = pd.read_csv(linkfile, index_col=False)
    columns = ['title', 'author', 'date', 'body']
    article = []

    # Create an empty DataFrame to store article content
    article = pd.DataFrame(article, columns=columns)
    article.to_csv(articlefile, encoding='utf-8', index=False)
    data = pd.DataFrame(columns=columns)

    # Iterate through the links and extract article content
    for i in range(0, len(links)):
        try:
            url = links.iloc[i][0]
            page = requests.get(url)
            soup = BeautifulSoup(page.content, 'html.parser')

            # Extract article title, author, date, and body
            title = soup.find("h1", class_="article-title").text
            author = soup.find("div", class_="article__byline")
            author = author.find("a").text.strip()
            date = re.findall(r'[0-9]{4}/[0-9]{2}/[0-9]{2}', url)[0]
            body = soup.find("div", class_="article-content")

            # Remove other article links from the body text
            other_link = body.find("div", class_="embed breakout")
            while other_link != None:
                body.find("div", class_="embed breakout").decompose()
                other_link = body.find("div", class_="embed breakout")
            body = body.text

            # Create a DataFrame to store the extracted article content
            article = pd.DataFrame([[title, author, date, body]], columns=columns)
            data = pd.concat([data, article], ignore_index=True)

            # Write data to the file and reset the DataFrame every 10 processed links
            if i % 10 == 0:
                data.to_csv(articlefile, encoding='utf-8', index=False, mode='a', header=False)
                data = pd.DataFrame(columns=columns)
        except:
            print("error extract turn is " + str(i))
            pass



