import requests
from bs4 import BeautifulSoup

from datetime import datetime
import time
import pandas as pd

import data_preprocessing


def extract_startus(region):
    """
        Extracts company information from “StartUs” website for a specific region.

        Args:
            region (str): The region of interest.

        Returns:
            None
    """

    # The desired number of links to crawl from StartUs.
    number_of_links = 100

    path = './data/website_company/startUs_' + region + '_info.csv'
    columns = data_preprocessing.attributes
    data = pd.DataFrame([], columns=columns)
    data.to_csv(path, encoding='utf-8', index=False)

    for turn in range(1, number_of_links // 10 + 1):
        try:
            data = pd.DataFrame([], columns=columns)
            url = "https://www.startus.cc/companies/startup/" + region + "?page=" + str(turn)
            page = requests.get(url)
            soup = BeautifulSoup(page.content, 'html.parser')
            print("Turn " + str(turn))

            elements = soup.findAll("h2", class_="company-profile__name")
            for element in elements:
                page = requests.get('https://www.startus.cc' + element.find('a')['href'])
                soup = BeautifulSoup(page.content, 'html.parser')
                org_name = soup.find('div', class_='field field--name-field-company-name field--type-text '
                                                   'field--label-hidden').text.strip()
                description = soup.find('div', class_='field-expander field-expander-0').text.strip()
                href_div = soup.find('div', class_='field field--name-field-company-homepage field--type-link-field '
                                                   'field--label-hidden')
                href = href_div.find('a')['href']

                # Category
                category = soup.find('div',
                                     'field field--name-field-company-industry field--type-taxonomy-term-reference '
                                     'field--label-inline clearfix')
                if category is not None:
                    category = category.text.strip()
                    categories = category[category.find(':') + 2:].split(',')
                    for i in range(len(categories)):
                        categories[i] = categories[i].strip()
                        if ' - ' in categories[i]:
                            categoryItem = categories[i].split(' ')
                            categories[i] = categoryItem[0] + '.' + categoryItem[2]
                else:
                    categories = 'None'

                # Found time
                found_time = soup.find('div', 'field field--name-field-company-founded-when field--type-datetime '
                                              'field--label-inline clearfix')
                if found_time is not None:
                    found_time = found_time.text.strip()
                    found_time = found_time[found_time.find(':') + 2:].split(' ')
                    month = datetime.strptime(found_time[2][:-1], "%B").month
                    if month < 10:
                        month = '0' + str(month)
                    else:
                        month = str(month)
                    found_time = found_time[3] + '/' + month + '/' + found_time[1]
                else:
                    found_time = 'None'

                # Stage
                stage = soup.find('div',
                                  'field field--name-field-company-startup-stage field--type-taxonomy-term-reference '
                                  'field--label-inline clearfix')
                if stage is not None:
                    stage = stage.text.strip()
                    stage = stage[stage.find(':') + 2:]
                    stage = stage[:stage.find('stage') - 1]
                else:
                    stage = 'None'

                # Employee
                employee = soup.find('div',
                                     'field field--name-field-company-number-employees field--type-number-integer '
                                     'field--label-inline clearfix')
                if employee is not None:
                    employee = employee.text.strip()
                    employee = employee[employee.find(':') + 2:]
                else:
                    employee = 'None'

                # Products
                products = soup.select(
                    'div.entity.entity-field-collection-item.field-collection-item-field-company-profile'
                    '-products')
                if products is not None:
                    intros = []
                    for product in products:
                        intro = []
                        content = product.select_one('div.content').find_all()
                        intro.append(content[0].text)
                        if len(content[1].text.split()) > 1:
                            intro.append(content[1].text.split()[1].strip())
                        else:
                            intro.append(content[1].text.strip())
                        intro.append(content[2].text.strip())
                        intros.append(intro)
                else:
                    intros = 'None'

                # Address
                address = soup.find('span', "country").text
                city = soup.select_one('div.addressfield-container-inline.locality-block')
                if city is not None:
                    address = address + ', ' + city.text
                else:
                    address = 'None'

                attributes = {'name': org_name, 'description': description, 'category': categories, 'URL': href,
                              'create_time': found_time, 'stage': stage, 'employees': employee, 'location': address}
                organisation = []
                source = ['0'] * len(columns)
                for index, column in enumerate(columns[:-1]):
                    if column in attributes.keys():
                        organisation.append(attributes[column])
                        source[index] = '1'
                    else:
                        organisation.append('None')
                organisation.append(' '.join(source))
                organisation = pd.DataFrame([organisation], columns=columns)
                data = pd.concat([data, organisation], ignore_index=True)
            data.to_csv(path, encoding='utf-8', index=False, mode='a', header=False)
            time.sleep(1)
        except Exception:
            print("error turn is " + str(turn))
            pass
