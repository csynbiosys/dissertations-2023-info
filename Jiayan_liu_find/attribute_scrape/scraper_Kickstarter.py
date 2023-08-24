from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium_stealth import stealth
import pandas as pd
import time
import undetected_chromedriver as uc
from attribute_scrape.extract_content import extract_kickstarter

# 打开网页
# chrome_driver = "C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe"
# driver = webdriver.Chrome(executable_path=chrome_driver)
# driver.maximize_window()

driver = uc.Chrome()

# options = webdriver.ChromeOptions()
# options.add_argument("start-maximized")
# # options.add_argument("--headless")
# options.add_experimental_option("excludeSwitches", ["enable-automation"])
# options.add_experimental_option('useAutomationExtension', False)
# driver = webdriver.Chrome(options=options, executable_path=r"C:\Users\DIPRAJ\Programming\adclick_bot\chromedriver.exe")
#
# stealth(driver,
#         languages=["en-US", "en"],
#         vendor="Google Inc.",
#         platform="Win32",
#         webgl_vendor="Intel Inc.",
#         renderer="Intel Iris OpenGL Engine",
#         fix_hairline=True,
#         )


# 创建文件，添加表头
data = pd.DataFrame([], columns=['product_name', 'fund', 'fund_goal', 'supporter_num', 'industry', 'location',
                                 'message_num'])
data.to_csv('kickstarter_attributes.csv', encoding='utf-8', index=False)

# 总链接数，加载回合数
number_of_links = 900  # change this for more articles
for turn in range(1, number_of_links // 18 + 1):
    try:
        url = "https://www.kickstarter.com/discover/advanced?woe_id=0&sort=newest&page=" + str(turn)
        driver.get(url)
        print("Turn " + str(turn))

        data = pd.DataFrame([], columns=['product_name', 'fund', 'fund_goal', 'supporter_num', 'industry', 'location',
                                         'message_num'])
        elements = driver.find_elements(By.CLASS_NAME, "soft-black mb3")[6:]
        for element in elements:
            product = extract_kickstarter(element.get_attribute('href'))
            data = pd.concat([data, product], ignore_index=True)
        print(data)
        data.to_csv('kickstarter_attributes.csv', encoding='utf-8', index=False, mode='a', header=False)
        driver.implicitly_wait(5)

    except Exception:
        print("last turn is " + str(turn))
        pass

driver.implicitly_wait(10)
driver.close()
