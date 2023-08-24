from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import pandas as pd
import time

import configuration


def extract_venturebeat_article():
    """
    Scrapes the latest news links from "Venturebeat" using 'Selenium' to simulate browser actions
    and saves them to a CSV file named 'venturebeat_links.csv'.

    Args:
        None

    Returns:
        None
    """

    # The desired number of links to crawl from Venturebeat.
    number_of_links = configuration.venturebeat_crawl_num

    # Open Site
    chrome_driver = "C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe"
    driver = webdriver.Chrome(executable_path=chrome_driver)
    driver.maximize_window()

    driver.get("https://venturebeat.com/category/enterprise-analytics/page/1")
    driver.implicitly_wait(5)
    parent_element = driver.find_elements(By.CLASS_NAME, "qc-cmp2-summary-buttons")[0]
    elements = parent_element.find_elements(By.XPATH, './/*')
    element = elements[4]
    ActionChains(driver).click(element).perform()

    # Create CSV file
    data = []
    data = pd.DataFrame(data, columns=['links'])
    data.to_csv('./data/news_company/venturebeat_links.csv', encoding='utf-8', index=False)

    # Scrape news links
    for turn in range(1, number_of_links // 7 + 1):
        try:
            url = "https://venturebeat.com/category/enterprise-analytics/page/" + str(turn)
            driver.get(url)

            print("Turn " + str(turn), end=" ")
            data = []
            elements = driver.find_elements(By.CLASS_NAME, "ArticleListing__title")
            for element in elements:
                a_element = element.find_element(By.XPATH, './*')
                data.append(a_element.get_attribute('href'))
            data = pd.DataFrame(data, columns=['links'])
            data.to_csv('./data/news_company/venturebeat_links.csv', encoding='utf-8', index=False, mode='a', header=False)

        except Exception:
            print("error turn is " + str(turn))
            pass

    driver.implicitly_wait(5)
    driver.close()
