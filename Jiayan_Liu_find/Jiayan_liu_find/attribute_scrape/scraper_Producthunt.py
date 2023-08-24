from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium_stealth import stealth
from bs4 import BeautifulSoup
import undetected_chromedriver as uc

import pandas as pd
import time

# # 打开网页
# porxyUrl = "http://api.proxy.ipidea.io/getBalanceProxyIp?num=100&return_type=json&lb=1&sb=0&flow=1&regions=gb&protocol=http"
# ipInfo = requests.get(porxyUrl)
# info = ipInfo.json()["data"]
# ip = info[0]["ip"]
# port = info[0]["port"]
#
# options=webdriver.ChromeOptions()
# options.add_argument(f'--proxy-server=http://{ip}:{port}')
#
# driver=webdriver.Chrome(chrome_options=options)
driver = uc.Chrome()
driver.maximize_window()
driver.get("https://www.producthunt.com/")

# 创建文件，添加表头
columns = ['name', 'date']
data = pd.DataFrame([], columns=columns)
data.to_csv('producthunt_info.csv', encoding='utf-8', index=False)

# 总链接数，加载回合数
number_of_links = 1000  # change this for more articles
number_of_loaded = 0
turn = 1
while number_of_loaded < number_of_links:
    # try:
    last_height = driver.execute_script("return document.body.scrollHeight")

    # 模拟下拉操作，直到滑动到底部
    while True:
        data = pd.DataFrame([], columns=columns)
        # 模拟下拉操作
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        buttons = driver.find_elements(By.CLASS_NAME, 'styles_reset__opz7w styles_button__zKntg styles_full__mfdC2')
        for button in buttons:
            actions = ActionChains(driver)
            actions.move_to_element(button).perform()
            ActionChains(driver).click(button).perform()

        time_elements = driver.find_elements(By.CLASS_NAME, 'pt-desktop-6.pt-mobile-0.pt-tablet-0.pt-widescreen-6')
        print('time', len(time_elements))
        for time_element in time_elements:
            print(time_element.text)
            date = time_element.text
            parent_element = time_element.find_element(By.XPATH, "..")
            link_elements = parent_element.find_elements(By.CLASS_NAME, 'styles_title__jWi91')
            print('link', len(link_elements))
            for link_element in link_elements:
                link = 'https://www.producthunt.com' + link_element.get_attribute('href')
                print(link)
                # time.sleep(1)
                # page = requests.get(link)
                # soup = BeautifulSoup(page.content, 'html.parser')
        #         name=soup.find('h1', 'color-darker-grey fontSize-24 fontWeight-700 noOfLines-undefined '
        #                              'styles_title__vct6Q')
        #         product = pd.DataFrame([[name, date]], columns=columns)
        #         print(product)
        #         data = pd.concat([data, product], ignore_index=True)
        # data.to_csv('producthunt_info.csv', encoding='utf-8', index=False, mode='a', header=False)
        # 等待页面加载
        time.sleep(2)
        # 获取当前页面的高度
        new_height = driver.execute_script("return document.body.scrollHeight")
        # 判断是否已经到达页面底部
        if new_height == last_height:
            break
        # 继续下拉操作
        last_height = new_height

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
