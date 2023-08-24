import pandas as pd
import requests
import execjs
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient import errors
from googleapiclient.discovery import build
import os.path

SCOPES = ['https://www.googleapis.com/auth/script.projects']


def similar_org_search():
    df = pd.read_csv('./data/org_info_searched.csv')
    for i in range(len(df)):
        row = df.iloc[i]
        row['competitors']=similar_company_g2(row['name'])
    df.to_csv('./data/org_info_searched.csv', index=False)


def similar_company_g2(company_name):
    script_id = '1vJAUvNfzRkaVqAqfa2N6UZWkenMd-Lz1sxFcCRaMyXG6l83wImpJMITN'

    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        print('success')
        service = build('script', 'v1', credentials=creds)
        request = {
            "function": "SimilarCompanyG2",
            "parameters": [
                company_name
            ], }
        response = service.scripts().run(scriptId=script_id, body=request).execute()
        return response['response']
    except errors.HttpError as error:
        # The API encountered a problem.
        print(error.content)
