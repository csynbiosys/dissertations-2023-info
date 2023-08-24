import numpy as np
import pandas as pd
import re

from gensim.models import FastText
from gensim.models import KeyedVectors

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')


def wordavg(model_ft, words):
    return np.mean([model_ft.wv[word] for word in words], axis=0)


def sentence_preprocessing(sentence):
    words = []
    stopwords_list = stopwords.words('english')
    pattern = r'[^a-zA-Z0-9\s]'

    sentence = re.sub(pattern, '', sentence).lower()
    tokens = word_tokenize(sentence)
    for token in tokens:
        if token not in stopwords_list:
            words.append(token)
    words = [word for word in words if not word.isdigit()]
    return words


def fasttext_train(country_code, num):
    address = filter_by_country(country_code, num)
    model_ft = FastText(address, vector_size=100, window=5, min_count=5, workers=4, sg=1)
    model_ft.save('vectors' + country_code + '.bin')


def filter_by_country(country_code, num):
    address = []
    df = pd.read_csv('/Users/northarbour/Downloads/bulk_export/organizations.csv')
    df = df.dropna(subset=['address'])
    # country = df['country_code'].unique()
    # count = df['country_code'].value_counts()
    # pd.set_option('display.max_rows', None)
    if country_code != 'WORLD':
        df = df[df['country_code'] == country_code]
    selected_address = df[['state_code', 'region', 'city', 'address']]
    for row in selected_address[:num].itertuples(index=False):
        words = str(row[0]) + ' ' + str(row[1]) + ' ' + str(row[2]) + ' ' + str(row[3])
        address.append(sentence_preprocessing(words))
    print(len(address))
    return address
    # USA    221971 200000 6000
    #
    # GBR     40562 39000 1170
    # DEU     13937 12000 360
    # FRA     12048 10000 300
    # ESP     9236  8000 240
    #
    # NLD      6373 5000 150
    # BRA      5924 5000 150
    # CHE      5647 5000 150
    # SWE      5466 5000 150
    #
    # ISR      5014 4000 120
    # ITA      4575 4000 120
    # IRL      3981 3000 90
    # 300000 9000


def load_model(country_code='WORLD'):
    return FastText.load('vectors' + country_code + '.bin')


def calculate_similarity(s1, s2, model):
    token1 = sentence_preprocessing(str(s1))
    token2 = sentence_preprocessing(str(s2))

    # token1 = wordavg(model, token1)
    # token2 = wordavg(model, token2)

    # 计算两个文本之间的 WMD; 输出相似度得分
    # wmdistance = model.wv.wmdistance(token1, token2)
    if len(token1) == 0 or len(token2) == 0:
        return 0
    else:
        return model.wv.n_similarity(token1, token2)
