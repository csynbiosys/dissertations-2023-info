from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import spacy

# Start the WebDrivers
driver = webdriver.Firefox()
# html_list_test
html_list = ['https://www.whiskyhammer.com/auction/past/auc-92/']

# html_list = ['https://www.highlandwhiskyauctions.com/may-2023/180-per-page' ,
#              'https://www.highlandwhiskyauctions.com/may-2023/180-per-page/page-2',
#              'https://www.highlandwhiskyauctions.com/may-2023/180-per-page/page-3',
#              'https://www.highlandwhiskyauctions.com/may-2023/180-per-page/page-4',
#              'https://www.highlandwhiskyauctions.com/may-2023/180-per-page/page-5',
#              'https://www.highlandwhiskyauctions.com/may-2023/180-per-page/page-6']

data = []
count = 0
count_attr = {'Distillery': 0, 'Age': 0, 'Country': 0, 'Bottles Produced': 0,
              'Region': 0, 'Staus': 0, 'Whisky Type': 0, 'Size': 0,
              'Strength': 0}
for htmls in html_list:
    # Go to the website
    # driver.get('https://www.highlandwhiskyauctions.com/may-2023')
    driver.get(htmls)
    # Get the page source
    html = driver.page_source

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')

    # Find all lot URLs on the page
    # lot_urls = [a['href'] for a in soup.find_all('a', {'class': 'buttonAlt'}) if 'item' in a['href']]
    lot_urls = ['https://www.whiskyhammer.com/item/152092/Bowmore/Bowmore---50-Year-Old-1969-Vault-Collection.html']
    # print(lot_urls)
    # lot_urls = [
    # 'https://www.highlandwhiskyauctions.com/lot-160861/macallan-james-bond-60th-anniversary-release-decade-i-vi-skyfall-lodge-print/auction-16']

    # Initialize an empty list to store the data for each lot

    # For each lot URL
    for lot_url in lot_urls:
        # Go to the lot page
        driver.get(lot_url)

        # Get the page source
        html = driver.page_source

        # Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        # description = soup.find('div', {'class': 'innerText pageContent'}).text
        # print(soup)
        # Extract the required information

        result_head = lot_url.split('item/')
        result_head = result_head[1].split('/')
        Id = result_head[0]
        Distillery = result_head[1]

        result = soup.find_all('div', {'class': 'inner'})
        attr_list = []
        for tag in result:
            # print(tag.text.strip())
            # print(tag)
            # print('----------------------------------------')
            attr_list.append(tag.text.strip())
            print(attr_list)

        temp = {'Id': Id, 'Distillery': Distillery, 'Age': 'not show', 'Country': 'not show',
                'Bottles Produced': 'not show',
                'Region': 'not show', 'Staus': 'not show', 'Whisky Type': 'not show', 'Size': 'not show',
                'Strength': 'not show'}

        attr_name = ["Age", "Bottles Produced", "Whisky Type", "Size", "Region", "Country", 'Strength']
        for attr in attr_list:
            for name in attr_name:
                # print(attr)
                if "Distillery Staus" in attr:
                    temp['Staus'] = attr
                    count_attr['Staus'] = count_attr['Staus'] + 1
                    break
                if name in attr:
                    temp[name] = attr
                    count_attr[name] = count_attr[name] + 1
                    break
        print(temp)
        count = count + 1
        # distillery = temp["Distillery"].replace("Distillery", "")
        vintage = temp["Age"].replace("Age", "")
        NumBottle = temp['Bottles Produced'].replace("Bottles Produced", "")
        Type = temp['Whisky Type'].replace("Whisky Type", "")
        Amount = temp['Size'].replace("Size", "")
        region = temp['Region'].replace("Region", "")
        country = temp['Country'].replace("Country\n", "")
        Staus = temp['Staus'].replace("Distillery Staus\n", "")
        Strength = temp['Strength'].replace("Strength", "")

        # ... continue for the other fields

        # Add the data for this lot to the list
        data.append([Id, Distillery, Staus, vintage, NumBottle, Type, Amount, Strength, region, country])
print(count)
print(count_attr)
# Convert the list to a pandas DataFrame
df = pd.DataFrame(data, columns=['Id', 'distillery', 'Distillery Staus', 'vintage', 'Bottles Produced', 'type',
                                 'amount',
                                 'Strength',
                                 'region', 'country'])

# Save the DataFrame to an Excel file
df.to_excel('auction_data_new.xlsx', index=False)

# Close the WebDriver
driver.quit()
