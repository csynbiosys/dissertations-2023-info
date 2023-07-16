from selenium.webdriver.common.by import By
import undetected_chromedriver as uc

import time
import pandas as pd

import data_preprocessing

# 打开网页
driver = uc.Chrome()
driver.maximize_window()
driver.get("https://www.startupers.com/")

# 创建文件，添加表头
columns = data_preprocessing.attributes
data = pd.DataFrame([], columns=columns)
data.to_csv('startupers_info.csv', encoding='utf-8', index=False)

last_height = 0
warn = 0
while True:
    data = pd.DataFrame([], columns=columns)
    # 模拟下拉操作
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    elements = driver.find_elements(By.CLASS_NAME, 'Card_card__QH1mK')
    print('time', len(elements))

    for element in elements:
        name = element.find_element(By.CLASS_NAME, 'Card_cardStartup__ejFuP').text
        location = element.find_element(By.CLASS_NAME, 'Card_cardBadge__0lfBD').text
        location = location.replace('-', '').replace(',', '').replace('Remote', '')
        location = location.strip()
        # page = requests.get(element.get_attribute('href'))
        # soup = BeautifulSoup(page.content, 'html.parser')
        #
        # div=soup.select_one('div.Post_greyBox__7MB7_.Post_border__ACaet')
        # print(div.text)
        # name = soup.select_one('p.Post_tweetDataName__XoMGK').text
        # # name = soup.find('p', class_='').text
        # location = soup.find('span', class_='Post_tweetDataText__MGCB2').text[4:]
        # description = soup.find('p', class_='Post_tweetDescription__gYyry').text
        attributes = {'name': name, 'location': location}
        organisation = []
        for column in columns:
            if column in attributes.keys():
                organisation.append(['Log1', attributes[column]])
            else:
                organisation.append('None')
        organisation = pd.DataFrame([organisation], columns=columns)
        data = pd.concat([data, organisation], ignore_index=True)
    data.to_csv('startupers_info.csv', encoding='utf-8', index=False, mode='a', header=False)
    # 等待页面加载
    time.sleep(1)

    # 获取当前页面的高度
    new_height = driver.execute_script("return document.body.scrollHeight")
    print('sign', new_height, last_height)
    # 判断是否已经到达页面底部
    if new_height == last_height:
        warn += 1
    if warn == 7:
        break
    # 继续下拉操作
    last_height = new_height

driver.implicitly_wait(5)
driver.close()
