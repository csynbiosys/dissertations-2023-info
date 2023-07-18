from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import spacy

# Start the WebDrivers
nlp = spacy.load('en_core_web_sm')
driver = webdriver.Firefox()
# html_list_test
html_list = ['https://www.highlandwhiskyauctions.com/may-2023/']

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
              'Strength': 0, "age_estimate": 0}
for htmls in html_list:
    # Go to the website
    # driver.get('https://www.highlandwhiskyauctions.com/may-2023')
    driver.get(htmls)
    # Get the page source
    html = driver.page_source

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')

    # Find all lot URLs on the page
    lot_urls = [a['href'] for a in soup.find_all('a', {'class': 'buttonAlt fullWidth'}) if 'lot-' in a['href']]
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
        description = soup.find('div', {'class': 'innerText pageContent'}).text
        doc = nlp(description)
        years = [ent.text for ent in doc.ents if ent.label_ == 'DATE' and ent.text.isdigit() and len(ent.text) == 4]
        if years:
            age_estimate = min(years)
            count_attr["age_estimate"] = count_attr["age_estimate"] + 1
        else:
            age_estimate = 'Unknown'

        # Extract the required information

        result_head = soup.find('h2')
        Id = result_head.text.strip(' <h2>Lot #')

        result = soup.find_all('div', {'class': 'inner'})
        attr_list = []
        for tag in result:
            # print(tag.text.strip())
            attr_list.append(tag.text.strip())

        temp = {'Id': Id, 'Distillery': 'not show', 'Age': 'not show', 'Country': 'not show',
                'Bottles Produced': 'not show',
                'Region': 'not show', 'Staus': 'not show', 'Whisky Type': 'not show', 'Size': 'not show',
                'Strength': 'not show'}

        attr_name = ["Distillery", "Age", "Bottles Produced", "Whisky Type", "Size", "Region", "Country", 'Strength']
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
        distillery = temp["Distillery"].replace("Distillery", "")
        vintage = temp["Age"].replace("Age", "")
        NumBottle = temp['Bottles Produced'].replace("Bottles Produced", "")
        Type = temp['Whisky Type'].replace("Whisky Type", "")
        Amount = temp['Size'].replace("Size", "")
        region = temp['Region'].replace("Region", "")
        country = temp['Country'].replace("Country", "")
        Staus = temp['Staus'].replace("Distillery Staus", "")
        Strength = temp['Strength'].replace("Strength", "")

        # ... continue for the other fields

        # Add the data for this lot to the list
        data.append([Id, distillery, Staus, vintage, NumBottle, Type, Amount, Strength, region, country, age_estimate])
print(count)
print(count_attr)
# Convert the list to a pandas DataFrame
df = pd.DataFrame(data, columns=['Id', 'distillery', 'Distillery Staus', 'vintage', 'Bottles Produced', 'type',
                                 'amount',
                                 'Strength',
                                 'region', 'country', 'age_estimate'])

df.loc[len(df.index)] = [count, count_attr['Distillery'], count_attr['Staus'], count_attr['Age'],
                         count_attr['Bottles Produced'], count_attr['Whisky Type'], count_attr['Size'],
                         count_attr['Strength'], count_attr['Region'], count_attr['Country'],
                         count_attr["age_estimate"]]
df.loc[len(df.index)] = [(count / count * 100), (count_attr['Distillery'] / count * 100),
                         (count_attr['Staus'] / count * 100), (count_attr['Age'] / count * 100),
                         (count_attr['Bottles Produced'] / count * 100), (count_attr['Whisky Type'] / count * 100),
                         (count_attr['Size'] / count * 100),
                         (count_attr['Strength'] / count * 100), (count_attr['Region'] / count * 100),
                         (count_attr['Country'] / count * 100), count_attr["age_estimate"]/count* 100]
df.loc[len(df.index)] = ['%.2f%%' % (count / count * 100), '%.2f%%' % (count_attr['Distillery'] / count * 100),
                         '%.2f%%' % (count_attr['Staus'] / count * 100), '%.2f%%' % (count_attr['Age'] / count * 100),
                         '%.2f%%' % (count_attr['Bottles Produced'] / count * 100),
                         '%.2f%%' % (count_attr['Whisky Type'] / count * 100),
                         '%.2f%%' % (count_attr['Size'] / count * 100),
                         '%.2f%%' % (count_attr['Strength'] / count * 100),
                         '%.2f%%' % (count_attr['Region'] / count * 100),
                         '%.2f%%' % (count_attr['Country'] / count * 100), '%.2f%%' % (count_attr["age_estimate"]/count * 100)]

# Save the DataFrame to an Excel file
df.to_excel('auction_data_new.xlsx', index=False)

# Close the WebDriver
driver.quit()
