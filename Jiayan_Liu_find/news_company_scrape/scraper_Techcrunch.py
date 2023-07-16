from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium_stealth import stealth
import pandas as pd
import time
import undetected_chromedriver as uc

import configuration
import extract_article_content

# 打开网页
chrome_driver = "C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe"
driver = webdriver.Chrome(executable_path=chrome_driver)
driver.maximize_window()
driver.get("https://techcrunch.com/startups/")

# 加载网页，同意Cookie
driver.implicitly_wait(5)
element = driver.find_element(By.NAME, "agree")
ActionChains(driver).click(element).perform()
driver.implicitly_wait(3)

# 创建文件，添加表头
data = []
data = pd.DataFrame(data, columns=['links'])
data.to_csv('techcrunch_links.csv', encoding='utf-8', index=False)

# 总链接数，加载回合数
number_of_links = configuration.techcrunch_crawl_num
number_of_loaded = 0
turn = 1
storeInterval = 5
while number_of_loaded < number_of_links:
    try:
        # 点击"Load More"，加载链接
        print("Turn " + str(turn), end=" ")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        element = driver.find_element(By.CLASS_NAME, "load-more ")
        actions = ActionChains(driver)
        actions.move_to_element(element).perform()
        ActionChains(driver).click(element).perform()
        driver.implicitly_wait(2)

        # 暂存加载出的链接
        if turn % storeInterval == 0:
            data = []  # empty list; will contain URLs to individual pages

            # extract links
            elements = driver.find_elements(By.CLASS_NAME, "post-block__title__link")
            number_of_loaded += len(elements)
            print("The loaded page number is " + str(number_of_loaded))
            for element in elements:
                # print(element.get_attribute('href')) # prints the link for each article's <a>
                data.append(element.get_attribute('href'))

            # store links locally
            data = pd.DataFrame(data, columns=['links'])
            data.to_csv('techcrunch_links.csv', encoding='utf-8', index=False, mode='a', header=False)
        turn += 1
    except Exception:
        print("error turn is " + str(turn))
        pass

driver.implicitly_wait(5)
driver.close()

extract_article_content.extract_techcrunch('techcrunch_links.csv', 'techcrunch_articles.csv')
