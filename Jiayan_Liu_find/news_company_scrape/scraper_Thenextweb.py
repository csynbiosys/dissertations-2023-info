from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import pandas as pd
import time

import configuration
import extract_article_content

# 打开网页
chrome_driver = "C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe"
driver = webdriver.Chrome(executable_path=chrome_driver)
driver.maximize_window()

# 创建文件，添加表头
data = []
data = pd.DataFrame(data, columns=['links'])
data.to_csv('thenextweb_links.csv', encoding='utf-8', index=False)

# 总链接数，加载回合数
number_of_links = configuration.thenextweb_crawl_num
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
        data.to_csv('thenextweb_links.csv', encoding='utf-8', index=False, mode='a', header=False)

    except Exception:
        print("error turn is " + str(turn))
        pass

driver.implicitly_wait(5)
driver.close()

extract_article_content.extract_thenextweb('thenextweb_links.csv', 'thenextweb_articles.csv')
