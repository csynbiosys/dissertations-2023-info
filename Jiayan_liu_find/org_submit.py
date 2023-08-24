import pandas as pd

from data_preprocessing import attributes


def submit_org():
    statistic = {'name': 0, 'location': 0, 'category': 0, 'stage': 0, 'create_time': 0, 'employees': 0}
    columns = attributes
    df = pd.read_csv('./data/org_info_searched.csv')
    attrs_required = ['name', 'category']
    attrs_index = []
    for attr in attrs_required:
        attrs_index.append(columns.index(attr))
    print(attrs_index)

    org_info_final = []
    org_info_wait = []
    for org in df.itertuples(index=False):
        count = 0
        for index in attrs_index:
            if org[index] == 'None':
                # statistic[attrs_required[attrs_index.index(index)]] = statistic[ attrs_required[attrs_index.index(
                # index)]] + 1
                break
            count = count + 1
        if count == len(attrs_required):
            org_info_final.append(org)
        else:
            org_info_wait.append(org)

    org_info_final = pd.DataFrame(org_info_final, columns=columns)
    org_info_final.to_csv('./data/org_info_final.csv', encoding='utf-8', index=False)
    org_info_wait = pd.DataFrame(org_info_wait, columns=columns)
    org_info_wait.to_csv('./data/org_info_wait.csv', encoding='utf-8', index=False)
