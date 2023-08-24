from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By

import pandas as pd
import os

import configuration


def extract_techcrunch_article():
    """
    Scrapes the latest news links from "TechCrunch" using 'Selenium' to simulate browser actions
    and saves them to a CSV file named 'techcrunch_links.csv'.

    Args:
        None

    Returns:
        None
    """

    # The desired number of links to crawl from TechCrunch.
    number_of_links = configuration.techcrunch_crawl_num

    # The current count of loaded links.
    number_of_loaded = 0

    # The current turn or iteration count.
    turn = 1

    # The interval in which to store the crawled links.
    storeinterval = 5

    # Open site
    chrome_driver = "C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe"
    driver = webdriver.Chrome(executable_path=chrome_driver)
    driver.maximize_window()
    driver.get("https://techcrunch.com/startups/")

    # Load site and accept Cookie
    driver.implicitly_wait(5)
    element = driver.find_element(By.NAME, "agree")
    ActionChains(driver).click(element).perform()
    driver.implicitly_wait(3)

    # Create CSV file
    data = []
    data = pd.DataFrame(data, columns=['links'])
    data.to_csv('./data/news_company/techcrunch_links.csv', encoding='utf-8', index=False)

    # Scrape news links
    while number_of_loaded < number_of_links:
        try:
            # Click the "Load More" button to load links
            print("Turn " + str(turn), end=" ")
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            element = driver.find_element(By.CLASS_NAME, "load-more ")
            actions = ActionChains(driver)
            actions.move_to_element(element).perform()
            ActionChains(driver).click(element).perform()
            driver.implicitly_wait(2)

            # Storing temporarily loaded links
            if turn % storeinterval == 0:
                data = []
                elements = driver.find_elements(By.CLASS_NAME, "post-block__title__link")
                number_of_loaded += len(elements)
                print("The loaded page number is " + str(number_of_loaded))
                for element in elements:
                    data.append(element.get_attribute('href'))
                data = pd.DataFrame(data, columns=['links'])
                data.to_csv('./data/news_company/techcrunch_links.csv', encoding='utf-8', index=False, mode='a',
                            header=False)
            turn += 1
        except Exception:
            print("error turn is " + str(turn))
            pass

    driver.implicitly_wait(5)
    driver.close()


def extract_techcrunch_article_dynamic():
    # The desired number of links to crawl from TechCrunch.
    number_of_links = 20

    # The current count of loaded links.
    number_of_loaded = 0

    # The current turn or iteration count.
    turn = 1

    # The interval in which to store the crawled links.
    storeinterval = 5

    # Open site
    chrome_driver = "C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe"
    driver = webdriver.Chrome(executable_path=chrome_driver)
    driver.maximize_window()
    driver.get("https://techcrunch.com/startups/")

    # Load site and accept Cookie
    driver.implicitly_wait(5)
    element = driver.find_element(By.NAME, "agree")
    ActionChains(driver).click(element).perform()
    driver.implicitly_wait(3)

    # Create CSV file
    data = []
    data = pd.DataFrame(data, columns=['links'])
    data.to_csv('./data/news_company/techcrunch_links.csv', encoding='utf-8', index=False)

    if not os.path.exists('./sign_links.csv'):
        # Create an empty DataFrame to store article content and save it to the file
        df = pd.DataFrame([], columns=['source', 'last_link'])
        df.to_csv('./sign_links.csv', encoding='utf-8', index=False)
    sign_links=pd.read_csv('./sign_links.csv')

    if 'techcrunch' in sign_links['source'].values:
        sign_row=sign_links[sign_links['source'] == 'techcrunch'].iloc[0]
        last_link=sign_row[1]
        sign=True

        # Scrape news links
        while sign:
            try:
                # Click the "Load More" button to load links
                print("Turn " + str(turn), end=" ")
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                element = driver.find_element(By.CLASS_NAME, "load-more ")
                actions = ActionChains(driver)
                actions.move_to_element(element).perform()
                ActionChains(driver).click(element).perform()
                driver.implicitly_wait(2)

                # Storing temporarily loaded links
                if turn % storeinterval == 0:
                    data = []
                    elements = driver.find_elements(By.CLASS_NAME, "post-block__title__link")
                    number_of_loaded += len(elements)
                    print("The loaded page number is " + str(number_of_loaded))
                    for element in elements:
                        if element.get_attribute('href') != last_link:
                            data.append(element.get_attribute('href'))
                        else:
                            sign_links.loc['techcrunch'] = ['source', data[0]]
                            sign=False
                            break
                    data = pd.DataFrame(data, columns=['links'])
                    data.to_csv('./data/news_company/techcrunch_links.csv', encoding='utf-8', index=False, mode='a',
                                header=False)
                turn += 1
            except Exception:
                print("error turn is " + str(turn))
                pass
    else:
        # Scrape news links
        while number_of_loaded < number_of_links:
            try:
                # Click the "Load More" button to load links
                print("Turn " + str(turn), end=" ")
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                element = driver.find_element(By.CLASS_NAME, "load-more ")
                actions = ActionChains(driver)
                actions.move_to_element(element).perform()
                ActionChains(driver).click(element).perform()
                driver.implicitly_wait(2)

                # Storing temporarily loaded links
                if turn % storeinterval == 0:
                    data = []
                    elements = driver.find_elements(By.CLASS_NAME, "post-block__title__link")
                    number_of_loaded += len(elements)
                    print("The loaded page number is " + str(number_of_loaded))
                    for element in elements:
                        data.append(element.get_attribute('href'))
                    sign_links.loc['techcrunch']=['source', data[-1]]
                    data = pd.DataFrame(data, columns=['links'])
                    data.to_csv('./data/news_company/techcrunch_links.csv', encoding='utf-8', index=False, mode='a',
                                header=False)
                turn += 1
            except Exception:
                print("error turn is " + str(turn))
                pass

    sign_links.to_csv('./sign_links.csv', encoding='utf-8', index=False)

    driver.implicitly_wait(5)
    driver.close()
