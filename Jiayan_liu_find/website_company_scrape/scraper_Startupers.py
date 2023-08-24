from selenium.webdriver.common.by import By
import undetected_chromedriver as uc

import time
import pandas as pd
import os

import data_preprocessing


def extract_stratupers():
    """
        Extracts company information from “Startupers” website using Selenium.

        This function opens the website, simulates scrolling to load more content,
        and extracts information about startup companies. The extracted information
        is then saved to a CSV file.

        Returns:
        None
    """

    # Open the website using a web driver
    driver = uc.Chrome()
    driver.maximize_window()
    driver.get("https://www.startupers.com/")

    # Create a CSV file and add headers
    columns = data_preprocessing.attributes
    data = pd.DataFrame([], columns=columns)
    data.to_csv('./data/website_company/startupers_info.csv', encoding='utf-8', index=False)

    last_height = 0
    warn = 0
    while True:
        data = pd.DataFrame([], columns=columns)
        # Simulate scrolling to load more content
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
            source = ['0'] * len(columns)
            for index, column in enumerate(columns[:-1]):
                if column in attributes.keys():
                    organisation.append(attributes[column])
                    source[index] = '2'
                else:
                    organisation.append('None')
            organisation.append(' '.join(source))
            organisation = pd.DataFrame([organisation], columns=columns)
            data = pd.concat([data, organisation], ignore_index=True)
        data.to_csv('./data/website_company/startupers_info.csv', encoding='utf-8', index=False, mode='a', header=False)
        # 等待页面加载
        time.sleep(1)

        # Get the current page's height
        new_height = driver.execute_script("return document.body.scrollHeight")
        print('sign', new_height, last_height)
        # Check if the page has reached the bottom
        if new_height == last_height:
            warn += 1
        if warn == 5:
            break
        last_height = new_height

    driver.implicitly_wait(5)
    driver.close()


def extract_stratupers_dynamic():
    """
        Extracts company information from “Startupers” website using Selenium.

        This function opens the website, simulates scrolling to load more content,
        and extracts information about startup companies. The extracted information
        is then saved to a CSV file.

        Returns:
        None
    """

    # Open the website using a web driver
    driver = uc.Chrome()
    driver.maximize_window()
    driver.get("https://www.startupers.com/")

    # Create a CSV file and add headers
    columns = data_preprocessing.attributes
    data = pd.DataFrame([], columns=columns)
    data.to_csv('./data/website_company/startupers_info.csv', encoding='utf-8', index=False)

    if not os.path.exists('./sign_links.csv'):
        # Create an empty DataFrame to store article content and save it to the file
        df = pd.DataFrame([], columns=['source', 'last_link'])
        df.to_csv('./sign_links.csv', encoding='utf-8', index=False)
    sign_links = pd.read_csv('./sign_links.csv')

    last_height = 0
    warn = 0
    if 'startupers' in sign_links['source'].values:
        sign_row=sign_links[sign_links['source'] == 'startupers'].iloc[0]
        last_link=sign_row[1]
        sign = True
        while sign:
            data = pd.DataFrame([], columns=columns)
            # Simulate scrolling to load more content
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            elements = driver.find_elements(By.CLASS_NAME, 'Card_card__QH1mK')
            print('time', len(elements))

            for element in elements:
                if element.get_attribute('href') == last_link:
                    sign_links.loc['startupers'] = ['source', data[0]]
                    sign = False
                    break
                else:
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
                    source = ['0'] * len(columns)
                    for index, column in enumerate(columns[:-1]):
                        if column in attributes.keys():
                            organisation.append(attributes[column])
                            source[index] = '2'
                        else:
                            organisation.append('None')
                    organisation.append(' '.join(source))
                    organisation = pd.DataFrame([organisation], columns=columns)
                    data = pd.concat([data, organisation], ignore_index=True)
            data.to_csv('./data/website_company/startupers_info.csv', encoding='utf-8', index=False, mode='a',
                        header=False)
            # 等待页面加载
            time.sleep(1)

            # Get the current page's height
            new_height = driver.execute_script("return document.body.scrollHeight")
            print('sign', new_height, last_height)
            # Check if the page has reached the bottom
            if new_height == last_height:
                warn += 1
            if warn == 5:
                break
            last_height = new_height
    else:
        data = pd.DataFrame([], columns=columns)
        # Simulate scrolling to load more content
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        elements = driver.find_elements(By.CLASS_NAME, 'Card_card__QH1mK')
        print('time', len(elements))
        href = ''
        for element in elements:
            href = element.get_attribute('href')
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
            source = ['0'] * len(columns)
            for index, column in enumerate(columns[:-1]):
                if column in attributes.keys():
                    organisation.append(attributes[column])
                    source[index] = '2'
                else:
                    organisation.append('None')
            organisation.append(' '.join(source))
            organisation = pd.DataFrame([organisation], columns=columns)
            data = pd.concat([data, organisation], ignore_index=True)
        sign_links.loc['startupers'] = ['source', href]
        data.to_csv('./data/website_company/startupers_info.csv', encoding='utf-8', index=False, mode='a',
                    header=False)

    sign_links.to_csv('./sign_links.csv', encoding='utf-8', index=False)

    driver.implicitly_wait(5)
    driver.close()
