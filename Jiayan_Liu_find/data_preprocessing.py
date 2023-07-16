import pandas as pd
import time
import requests

columns = ['name', 'type', 'priority']
general_company_attributes = ['location', 'category', 'stage', 'social_media', 'age',
                              'create_time', 'close_time', 'employees', 'type']
meta_information_attributes = ['name', 'URL', 'update_time', 'description']

data = pd.DataFrame([], columns=columns)

# Org1: StartUs, Org2: Startupers, Log1: Company House
for attribute in general_company_attributes:
    element = pd.DataFrame([[attribute, 'General Company', ['Log1', 'Org1', 'Org2']]], columns=columns)
    data = pd.concat([data, element], ignore_index=True)
for attribute in meta_information_attributes:
    element = pd.DataFrame([[attribute, 'Meta Information', ['Log1', 'Org1', 'Org2']]], columns=columns)
    data = pd.concat([data, element], ignore_index=True)

data.to_csv('attributes_info.csv', encoding='utf-8', index=False)

attributes = meta_information_attributes + general_company_attributes


