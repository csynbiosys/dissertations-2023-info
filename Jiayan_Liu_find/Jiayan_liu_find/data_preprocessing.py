import pandas as pd
import pickle
import os
import fnmatch

columns = ['name', 'type', 'priority']

general_company_attributes = ['location', 'category', 'stage', 'social_media', 'age',
                              'create_time', 'close_time', 'employees', 'type']
meta_information_attributes = ['name', 'URL', 'update_time', 'description']

data = {}

# 1: StartUs, 2: Startupers, 3: Company House, 4: News Website
for attribute in general_company_attributes:
    data[attribute] = ['General Company', [4, 1, 3, 2]]
for attribute in meta_information_attributes:
    data[attribute] = ['Meta Information', [4, 1, 3, 2]]

with open('attributes.pickle', 'wb') as file:
    pickle.dump(data, file)

attributes = meta_information_attributes + general_company_attributes
attributes.append('source')


def search_merge(old, new, new_priority, attributes):
    source = old['source'].replace(' ', '')
    for index in new.index:
        if index in old.index:
            old_priority = int(source[old.index.get_loc(index)])
            priority = attributes[index]
            if old_priority == 0 or priority[1].index(new_priority) < priority[1].index(old_priority):
                old[index] = new[index]
    return old

def find_path(dir):
    fileList = []
    # 遍历项目中的所有文件和文件夹
    for root, dirs, files in os.walk(dir):
        for file in files:
            # 使用fnmatch模块的fnmatch函数进行文件名匹配
            if fnmatch.fnmatch(file, '*info*'):
                # 打印匹配到的文件路径
                file_path = os.path.join(root, file)
                # file_path.replace('/', '//')
                print(file_path)
                fileList.append(file_path)
    return fileList


def merge_info(direct):
    fileList = find_path(direct)
    columns = attributes
    data = pd.DataFrame([], columns=columns)
    for file in fileList:
        df = pd.read_csv(file)
        data = pd.concat([data, df])

    data_unique = data.drop_duplicates(subset=['name'])
    data_unique.to_csv('./data/org_info.csv', encoding='utf-8', index=False)