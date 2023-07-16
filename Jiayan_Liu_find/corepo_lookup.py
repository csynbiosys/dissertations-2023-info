import requests


query = {'q': 'insurance'}

resp = requests.get('https://corepo.org/api/v1/search/', params=query)
print(resp.text)