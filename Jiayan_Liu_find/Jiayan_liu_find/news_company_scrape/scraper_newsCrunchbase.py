from selenium import webdriver
from selenium.webdriver.common.by import By

import pandas as pd

import configuration


def extract_newsCrunchbase_article():
    """
    Scrapes the latest news links from "Crunchbase" using 'Selenium' to simulate browser actions
    and saves them to a CSV file named 'newsCrunchbase_links.csv'.

    Args:
        None

    Returns:
        None
    """

    # The desired number of links to crawl from Crunchbase.
    number_of_links = configuration.crunchbase_crawl_num

    # Open site
    chrome_driver = "C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe"
    driver = webdriver.Chrome(executable_path=chrome_driver)
    driver.maximize_window()

    # Create CSV file
    data = []
    data = pd.DataFrame(data, columns=['links'])
    data.to_csv('./data/news_company/newsCrunchbase_links.csv', encoding='utf-8', index=False)

    # Scrape news links
    for turn in range(1, number_of_links // 10 + 1):
        try:
            url = "https://news.crunchbase.com/page/" + str(turn)
            driver.get(url)

            print("Turn " + str(turn), end=" ")
            data = []
            elements = driver.find_elements(By.CLASS_NAME, "entry-title.h3")
            for element in elements:
                a_element = element.find_element(By.XPATH, './*')
                data.append(a_element.get_attribute('href'))
            data = pd.DataFrame(data, columns=['links'])
            data.to_csv('./data/news_company/newsCrunchbase_links.csv', encoding='utf-8', index=False, mode='a',
                        header=False)

        except Exception:
            print("error turn is " + str(turn))
            pass

    driver.implicitly_wait(5)
    driver.close()
