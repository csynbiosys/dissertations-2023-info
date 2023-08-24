from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium_stealth import stealth
import pandas as pd
import time
import undetected_chromedriver as uc

import extract_content


#创建文件，添加表头
data = pd.DataFrame([], columns=['product_name', 'fund', 'fund_goal', 'supporter_num', 'industry', 'location', 'message_num'])
data.to_csv('kickstarter_attributes.csv', encoding='utf-8', index=False)

number_of_links = 1000 # change this for more articles
number_of_loaded=0
turn=1
storeInterval=5
lastpage=1

# 打开网页
driver=uc.Chrome()
driver.maximize_window()
driver.get("https://www.kickstarter.com/discover/advanced?sort=newest")
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
driver.implicitly_wait(5)
element = driver.find_element(By.CLASS_NAME, "bttn bttn-primary")
actions = ActionChains(driver)
actions.move_to_element(element).perform()
ActionChains(driver).click(element).perform()
driver.implicitly_wait(5)

while number_of_loaded<number_of_links:
    try:
        #点击"Load More"，加载链接
        print("Turn "+str(turn), end=" ")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        driver.implicitly_wait(1)
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
        turn+=1
    except Exception:
        print("last turn is "+str(turn))
        pass

driver.implicitly_wait(5)
driver.close()


# 总链接数，加载回合数
number_of_links = 900 # change this for more articles
for turn in range(1,number_of_links//18+1):
    try:
        url="https://www.kickstarter.com/discover/advanced?woe_id=0&sort=newest&page="+str(turn)
        driver.get(url)
        print("Turn "+str(turn))

        data = pd.DataFrame([], columns=['product_name', 'fund', 'fund_goal', 'supporter_num', 'industry', 'location', 'message_num'])
        elements = driver.find_elements(By.CLASS_NAME, "soft-black mb3")[6:]
        for element in elements:
            product=extract_content.extract_kickstarter(element.get_attribute('href'))
            data = pd.concat([data, product], ignore_index=True)
        print(data)
        data.to_csv('kickstarter_attributes.csv', encoding='utf-8', index=False, mode='a', header=False)
        driver.implicitly_wait(5)

    except Exception:
        print("last turn is "+str(turn))
        pass

driver.implicitly_wait(10)
driver.close()