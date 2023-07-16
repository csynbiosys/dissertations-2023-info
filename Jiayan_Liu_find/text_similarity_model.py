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

    sentence = re.sub(pattern, '', sentence)
    tokens = word_tokenize(sentence)
    for token in tokens:
        if token not in stopwords_list:
            words.append(token)
    return words


def fasttext_train(file_path):
    address = []

    df = pd.read_csv(file_path)
    for row in df.itertuples(index=False):
        address.append(sentence_preprocessing(row[7]))

    df = pd.read_csv('/Users/northarbour/Downloads/bulk_export/organizations.csv')
    for row in df.itertuples(index=False):
        words = str(row[12]) + ' ' + str(row[13]) + ' ' + str(row[14]) + ' ' + str(row[15]) + ' ' + str(
            row[16]) + ' ' + str(row[17])
        address.append(sentence_preprocessing(words))

    model_ft = FastText(address, vector_size=100, window=5, min_count=5, workers=4, sg=1)
    model_ft.save('vectors.bin')


def load_model():
    return KeyedVectors.load('vectors.bin')


def calculate_similarity(s1, s2, model):
    token1 = sentence_preprocessing(s1)
    token2 = sentence_preprocessing(s2)

    token1 = wordavg(model, token1)
    token2 = wordavg(model, token2)

    # 计算两个文本之间的 WMD; 输出相似度得分
    wmdistance = model.wv.wmdistance(token1, token2)
    return 1 - wmdistance
