from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium_stealth import stealth

import pandas as pd
import time
import requests


# 打开网页
porxyUrl = "http://api.proxy.ipidea.io/getBalanceProxyIp?num=100&return_type=json&lb=1&sb=0&flow=1&regions=gb&protocol=http"
ipInfo = requests.get(porxyUrl)
info = ipInfo.json()["data"]
ip = info[0]["ip"]
port = info[0]["port"]

options=webdriver.ChromeOptions()
options.add_argument(f'--proxy-server=http://{ip}:{port}')

driver=webdriver.Chrome(chrome_options=options)
driver.maximize_window()
driver.get("https://wellfound.com/discover/startups")

#创建文件，添加表头
data=[]
data = pd.DataFrame(data, columns=['links'])
data.to_csv('wellfound_links.csv', encoding='utf-8', index=False)

# 总链接数，加载回合数
number_of_links = 1000 # change this for more articles
number_of_loaded=0
turn=1
storeInterval=5
while number_of_loaded<number_of_links:
    # try:
    #点击"Load More"，加载链接
    print("Turn "+str(turn), end=" ")
    # driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    parentElement = driver.find_element(By.CLASS_NAME, "mt-4.text-center")
    element=parentElement.find_element(By.XPATH, "./*[1]")
    driver.execute_script("arguments[0].scrollIntoView();", element)
    driver.implicitly_wait(5)
    element.click()
    # actions = ActionChains(driver)
    # actions.move_to_element(element).perform()
    # ActionChains(driver).click(element).perform()
    driver.implicitly_wait(5)

        # #暂存加载出的链接
        # if turn%storeInterval==0:
        #     data = []  # empty list; will contain URLs to individual pages
        #
        #     # extract links
        #     elements = driver.find_elements(By.CLASS_NAME, "post-block__title__link")
        #     number_of_loaded+=len(elements)
        #     print("The loaded page number is "+str(number_of_loaded))
        #     for element in elements:
        #         # print(element.get_attribute('href')) # prints the link for each article's <a>
        #         data.append(element.get_attribute('href'))
        #
        #     # store links locally
        #     data = pd.DataFrame(data, columns=['links'])
        #     data.to_csv('techcrunch_links.csv', encoding='utf-8', index=False, mode='a', header=False)
        # turn+=1
    # except Exception:
    #     print("last turn is "+str(turn))
    #     pass

driver.implicitly_wait(5)
driver.close()