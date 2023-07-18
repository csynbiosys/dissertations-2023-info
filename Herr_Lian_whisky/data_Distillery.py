import requests
from bs4 import BeautifulSoup
import pandas as pd
# Send a GET request to the website
response = requests.get('https://whiskymate.net/the-distillery-list/')

# Parse the HTML content of the page with BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Find the table on the page
table = soup.find(attrs={'class': 'entry-content'})

# Initialize an empty list to store the distillery data
distillery_data = []

# For each row in the table (except for the first header row)
for row in table.find_all('p'):
    # Get the columns in the row
    if 'Established' in row.text and 'Region' in row.text:
        print(row.text)

        # attr_list.append(row.text.strip('()'))
        attr = row.text.split(')')
        attr_name = attr[1].split('(')
        name = attr_name[0].strip(' ')
        attr_tmp = attr[1].split(':')
        attr_establish = attr_tmp[1].strip(' ')
        attr_establish = attr_establish.split('|')
        year = attr_establish[0]
        attr_region = attr_tmp[2].strip(' ')
        attr_region = attr_region.split('|')
        region = attr_region[0]
        attr_status = attr_tmp[-1].strip(' ')
        attr_status = attr_status.split('|')
        status = attr_status[0]

        print(status)
        distillery_data.append([name, year, region, status])

# Convert the list to a pandas DataFrame
distillery_df = pd.DataFrame(distillery_data, columns=['Name', 'Year', 'Region', 'Status'])

# Save the DataFrame to an Excel file
distillery_df.to_excel('distillery_data.xlsx', index=False)
