from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium_stealth import stealth

import pandas as pd
import time
import requests


def extract_wellfound():
    """
    Extracts startup links from the Wellfound website.

    Returns:
        None
    """

    # Obtain proxy IP information from API
    porxyUrl = "http://api.proxy.ipidea.io/getBalanceProxyIp?num=100&return_type=json&lb=1&sb=0&flow=1&regions=gb&protocol=http"
    ipInfo = requests.get(porxyUrl)
    info = ipInfo.json()["data"]
    ip = info[0]["ip"]
    port = info[0]["port"]

    # Configure Chrome webdriver with proxy settings
    options = webdriver.ChromeOptions()
    options.add_argument(f'--proxy-server=http://{ip}:{port}')

    # Initialize Chrome driver
    driver = webdriver.Chrome(chrome_options=options)
    driver.maximize_window()
    driver.get("https://wellfound.com/discover/startups")

    # Create a DataFrame to store links and save as CSV
    data = []
    data = pd.DataFrame(data, columns=['links'])
    data.to_csv('wellfound_links.csv', encoding='utf-8', index=False)

    # Define the total number of links to load and the current loaded count
    number_of_links = 1000  # change this for more articles
    number_of_loaded = 0
    turn = 1
    storeInterval = 5

    # Loop until the required number of links are loaded
    while number_of_loaded < number_of_links:
        print("Turn " + str(turn), end=" ")

        # Scroll to the "Load More" element and click it
        parentElement = driver.find_element(By.CLASS_NAME, "mt-4.text-center")
        element = parentElement.find_element(By.XPATH, "./*[1]")
        driver.execute_script("arguments[0].scrollIntoView();", element)
        driver.implicitly_wait(5)
        element.click()
        driver.implicitly_wait(5)

        # Uncomment the following block if you want to store links at specific intervals
        # # Store loaded links at intervals
        # if turn % storeInterval == 0:
        #     data = []  # empty list to contain URLs to individual pages
        #     elements = driver.find_elements(By.CLASS_NAME, "post-block__title__link")
        #     number_of_loaded += len(elements)
        #     print("The loaded page number is " + str(number_of_loaded))
        #     for element in elements:
        #         data.append(element.get_attribute('href'))
        #     data = pd.DataFrame(data, columns=['links'])
        #     data.to_csv('wellfound_links.csv', encoding='utf-8', index=False, mode='a', header=False)
        # turn += 1

    # Close the Chrome driver
    driver.implicitly_wait(5)
    driver.close()
