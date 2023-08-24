import os
import fnmatch
import pandas as pd

import text_similarity_model


def find_org_by_URL(url, cb_orgs):
    if url.endswith("/"):
        url = url.rstrip("/")
    url = url.split("://")[1]
    url1 = "://" + url
    url2 = "." + url
    url3 = url
    if url.endswith(".com"):
        url3 = url3.rstrip(".com")
    if url.startswith("www."):
        url3 = url3.lstrip("www.")
    # to avoid partial matches, make sure match url has www.url or http(s)://url
    temp = cb_orgs[
        cb_orgs['homepage_url'].str.contains(url, na=False, regex=False)
        | cb_orgs['homepage_url'].str.contains(url1, na=False, regex=False)
        | cb_orgs['homepage_url'].str.contains(url2, na=False, regex=False)
        | cb_orgs['homepage_url'].str.contains(url3, na=False, regex=False)]
    return temp


def get_similarity(text, org, model):
    return text_similarity_model.calculate_similarity(text, org, model)


def find_org_by_Name(org, cb_orgs, model):
    cb_orgs['similarity'] = cb_orgs['name'].apply(get_similarity, args=(org, model))
    temp = cb_orgs.nlargest(10, 'similarity')
    if temp.iloc[0, temp.columns.get_loc('similarity')] < 0.8:
        temp = pd.DataFrame()
    return temp


def convert_employees(employees):
    if employees == 'None':
        return 'None'
    else:
        employees = int(employees)
        if employees <= 10:
            return '1-10'
        elif employees <= 50:
            return '11-50'
        elif employees <= 100:
            return '51-100'
        elif employees <= 250:
            return '101-250'
        elif employees <= 500:
            return '251-500'
        elif employees <= 1000:
            return '501-1000'
        elif employees <= 5000:
            return '1001-5000'
        elif employees <= 10000:
            return '5001-10000'
        else:
            return '10000+'


def data_preprocessing():
    result_orgs = pd.read_csv('./data/org_info_final.csv')
    cb_orgs = pd.read_csv('/Users/northarbour/Downloads/bulk_export/organizations.csv')

    result_orgs['create_year'] = 'None'
    result_orgs['create_month'] = 'None'
    result_orgs['create_date'] = 'None'
    result_orgs['country_code']='None'
    result_orgs['city']='None'
    for i in range(len(result_orgs)):
        org = result_orgs.iloc[i]
        # create_time
        if org['create_time'] != 'None':
            create_time = org['create_time'].split('/')
            org['create_year'] = create_time[0].lstrip('0')
            org['create_month'] = create_time[1].lstrip('0')
            org['create_date'] = create_time[2].lstrip('0')
        else:
            org['create_year'] = 'None'
            org['create_month'] = 'None'
            org['create_date'] = 'None'

        # location
        if org['location'] is not None:
            print(org['location'])
            location = org['location'].split(', ')
            if len(location) >= 2:
                org['country_code'] = location[0]
                org['city'] = location[1]
            else:
                org['country_code'] = 'None'
                org['city'] = 'None'
        else:
            org['country_code'] = 'None'
            org['city'] = 'None'
        # employees
        org['employees'] = convert_employees(org['employees'].replace(' ', ''))
    result_orgs.drop('create_time', axis=1)
    result_orgs.drop('location', axis=1)
    result_orgs.drop('description', axis=1)




    result_orgs.to_csv('./data/final_org_prepossed.csv', encoding='utf-8', index=False)

    cb_orgs_not_none = cb_orgs[cb_orgs['created_at'] != 'None']
    if cb_orgs_not_none['created_at'].str is not None:
        cb_orgs_not_none['created_at'] = cb_orgs_not_none['created_at'].str.split(' ')[0].split('-')
    new_columns = cb_orgs_not_none['created_at'].apply(pd.Series)
    new_columns.columns = ['create_year', 'create_month', 'create_date']

    new_columns['create_month'] = new_columns['create_month'].str.lstrip('0')
    new_columns['create_date'] = new_columns['create_date'].str.lstrip('0')
    cb_orgs_not_none = pd.concat([cb_orgs_not_none, new_columns], axis=1)
    cb_orgs_not_none.drop('created_at', axis=1)
    print(cb_orgs_not_none.columns)

    cb_orgs_none = cb_orgs[cb_orgs['created_at'] != 'None']
    cb_orgs_none['create_year'] = 'None'
    cb_orgs_none['create_month'] = 'None'
    cb_orgs_none['create_date'] = 'None'
    cb_orgs_none.drop('created_at', axis=1)
    print(cb_orgs_none.columns)

    cb_orgs = pd.concat([cb_orgs_not_none, cb_orgs_none], ignore_index=True)
    cb_orgs['employee_count'] = cb_orgs['employee_count'].replace('unknown', 'None')
    cb_orgs.drop('created_at', axis=1)
    cb_orgs.drop('short_description', axis=1)
    cb_orgs.to_csv('./data/cb_org_prepossed.csv', encoding='utf-8', index=False)

    # for i in range(len(cb_orgs)):
    #     org = cb_orgs.iloc[i]
    #     # create_time
    #     if org['created_at'] != 'None':
    #         create_time = org['created_at'].split(' ')[0].split('-')
    #         org['create_year'] = create_time[0].lstrip('0')
    #         org['create_month'] = create_time[1].lstrip('0')
    #         org['create_date'] = create_time[2].lstrip('0')
    #     else:
    #         org['create_year'] = 'None'
    #         org['create_month'] = 'None'
    #         org['create_date'] = 'None'
    #     # employees
    #     if org['employee_count'] == 'unknown':
    #         org['employee_count'] = 'None'
    # cb_orgs.drop('created_at', axis=1)
    # cb_orgs.drop('short_description', axis=1)
    # cb_orgs.to_csv('./data/cb_org_prepossed.csv', encoding='utf-8', index=False)


def evaluation():
    no_match_count = 0
    result_orgs = pd.read_csv('./data/final_org_prepossed.csv')
    cb_orgs = pd.read_csv('./data/cb_org_prepossed.csv')
    result_orgs['accuracy'] = 0
    accuracy=0
    for i in range(len(result_orgs)):
        print('aim org: ' + str(i))
        org = result_orgs.iloc[i]
        # 先进行URL检验，无存在则执行名称匹配
        url = str(org['URL'])
        if url != 'None':
            temp = find_org_by_URL(url, cb_orgs)
            if temp.empty:
                no_match_count += 1
                continue
        else:
            no_match_count += 1
            continue
        # else:
        #     name=str(org['name'])
        #     model=text_similarity_model.load_model('USA')
        #     temp=find_org_by_Name(name, cb_orgs, model)
        #     if temp.empty:
        #         no_match_count += 1
        #         continue
        print('temp length: ' + str(len(temp)))

        accuracies = []
        for index, row in temp.iterrows():
            accuracy = 0
            non_empty_cols = 0
            # 1 compare company name
            if str(row['name']) != 'nan':
                print('Name: ' + row['name'] + org['name'])
                non_empty_cols += 1
                if str(row['name']).lower() in str(org['name']).lower() or str(
                        org['name']).lower() in str(row['name']).lower():
                    accuracy += 1

            # 2 compare employee count
            if str(org['employees']) == 'None':
                if str(row['employee_count']) == 'None':
                    non_empty_cols += 1
                    accuracy += 1
            else:
                if str(row['employee_count']) != 'None':
                    non_empty_cols += 1
                    if str(row['employee_count']) == str(org['employees']):
                        accuracy += 1

            # 4 compare country
            if str(row['country_code']) != 'None':
                non_empty_cols += 1
                actual = row['country_code']
                scraped = org['country_code']
                if actual == scraped:
                    accuracy += 1
            # 5 compare city
            if str(row['city']) != 'None':
                non_empty_cols += 1
                actual = row['city']
                scraped = str(org['city'])
                if str(actual).lower().strip() == str(scraped).lower().strip():
                    accuracy += 1

            # 6 compare founding year
            if row['create_year'] != 'None':
                non_empty_cols += 1
                if row['create_year'] == org['create_year']:
                    accuracy += 1

            # # 7 compare linkedin URL using identifier
            # if str(row['linkedin_identifier']) != 'None':
            #     non_empty_cols += 1
            #     actual = row['linkedin_identifier']
            #     scraped = str(org['linkedin_identifier'])
            #     if actual.lower().strip() == scraped.lower().strip():
            #         accuracy += 1

            accuracies.append(float(accuracy / non_empty_cols))
            print(str(index) + ' accuracy:' + str(accuracy) + ' nec: ' + str(non_empty_cols))
        print(max(accuracies), end=' ')
        # store accuracy for the best match
        result_orgs['accuracy'].iloc[i] = max(accuracies)
        accuracy+=max(accuracies)
    print('average accuracy: '+str(accuracy/len(result_orgs)-no_match_count))
