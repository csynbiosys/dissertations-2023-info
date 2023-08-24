from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import pandas as pd
import time

import configuration


def extract_thenextweb_article():
    """
    Scrapes the latest news links from "theNextWeb" using 'Selenium' to simulate browser actions
    and saves them to a CSV file named 'thenextweb_links.csv'.

    Args:
        None

    Returns:
        None
    """


    # The desired number of links to crawl from theNextWeb.
    number_of_links = configuration.thenextweb_crawl_num

    # Open site
    chrome_driver = "C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe"
    driver = webdriver.Chrome(executable_path=chrome_driver)
    driver.maximize_window()

    # Create CSV file
    data = []
    data = pd.DataFrame(data, columns=['links'])
    data.to_csv('./data/news_company/thenextweb_links.csv', encoding='utf-8', index=False)

    # Scrape news links
    for turn in range(1, number_of_links // 7 + 1):
        try:
            url = "https://thenextweb.com/startups-technology/page/" + str(turn)
            driver.get(url)

            print("Turn " + str(turn), end=" ")
            data = []
            elements = driver.find_elements(By.CLASS_NAME, "c-listArticle__heading")
            for element in elements:
                a_element = element.find_element(By.XPATH, './*')
                data.append(a_element.get_attribute('href'))
            data = pd.DataFrame(data, columns=['links'])
            data.to_csv('./data/news_company/thenextweb_links.csv', encoding='utf-8', index=False, mode='a', header=False)

        except Exception:
            print("error turn is " + str(turn))
            pass

    driver.implicitly_wait(5)
    driver.close()
