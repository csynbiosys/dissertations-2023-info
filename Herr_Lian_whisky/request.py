import requests
from bs4 import BeautifulSoup

# Visit the main page and get the links for each "View Lot"
main_url = "https://www.highlandwhiskyauctions.com/may-2023"
response = requests.get(main_url)
print(response)
soup = BeautifulSoup(response.text, 'html.parser')
print(soup)
# for link in soup.find_all('a'):
#     print(link)
#     print(link.get('href'))

lot_links = [a['href'] for a in soup.select('a[href^="buttonAlt fullWidth"]')]

print(lot_links)

# Scrape the content from the first "View Lot" link
lot_url = lot_links[0]  # Use the first link as an example
response = requests.get(lot_url)
soup = BeautifulSoup(response.text, 'html.parser')

# Extract the required information
winery = soup.select_one('div.winery').text
vintage = soup.select_one('div.vintage').text
num_bottles = soup.select_one('div.num_bottles').text
# ... continue for the rest of the fields

# Print the information
print(f"Winery: {winery}")
print(f"Vintage: {vintage}")
print(f"Number of bottles originally produced: {num_bottles}")
# ... continue for the rest of the fields
