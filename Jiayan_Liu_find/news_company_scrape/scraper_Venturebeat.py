from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import pandas as pd
import time

import configuration

# 打开网页
chrome_driver = "C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe"
driver = webdriver.Chrome(executable_path=chrome_driver)
driver.maximize_window()

driver.get("https://venturebeat.com/category/enterprise-analytics/page/1")
driver.implicitly_wait(5)
parent_element = driver.find_elements(By.CLASS_NAME, "qc-cmp2-summary-buttons")[0]
elements = parent_element.find_elements(By.XPATH, './/*')
element = elements[4]
ActionChains(driver).click(element).perform()

# 创建文件，添加表头
data = []
data = pd.DataFrame(data, columns=['links'])
data.to_csv('venturebeat_links.csv', encoding='utf-8', index=False)

# 总链接数，加载回合数
number_of_links = configuration.venturebeat_crawl_num
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
        data.to_csv('venturebeat_links.csv', encoding='utf-8', index=False, mode='a', header=False)

    except Exception:
        print("error turn is " + str(turn))
        pass

driver.implicitly_wait(5)
driver.close()
