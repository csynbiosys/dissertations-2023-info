import spacy
import pandas as pd
import re
import math
import statistics
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim import corpora, models

import data_preprocessing

stopwords_list = stopwords.words('english')
pattern = r'[^a-zA-Z0-9\s]'

nlp = spacy.load("en_core_web_sm")


def build_dictionary(corpus):
    dictionary = corpora.Dictionary(corpus)
    corpus_bow = [dictionary.doc2bow(doc) for doc in corpus]
    corpora.MmCorpus.serialize('corpus.mm', corpus_bow)
    mm_corpus = corpora.MmCorpus('corpus.mm')

    word_counts = {}
    for doc in mm_corpus:
        for word_id, count in doc:
            word = dictionary[word_id]
            word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts


def tf_idf(orgs, dictionary, total_doc):
    tfidf = {}
    total = 0
    for value in orgs.values():
        total = total + value
    if total > 0:
        for key, value in orgs.items():
            tf = value / total
            count = dictionary.get(key, 0)
            idf = math.log(total_doc / (1 + count))
            tfidf[key] = tf * idf
        max_tfidf = max(tfidf.values())
        return tfidf, max_tfidf
    else:
        return tfidf, 0


def data_prepocessing(phrase):
    output = ''
    phrase = re.sub(pattern, '', phrase)
    phrase = re.sub(r"('s|’s)$", "", phrase)
    phrase = re.sub(r"(s'|s’)$", "s", phrase)
    tokens = word_tokenize(phrase)
    for token in tokens:
        if token not in stopwords_list:
            output = output + token + ' '
    return output.strip()


def extract_orgs(filepath):
    organisations = []
    columns = data_preprocessing.attributes
    filename = filepath.split('_')[0] + '_info.csv'
    df = pd.read_csv(filepath)

    corpus = []
    total_doc = 0
    for row in df.itertuples(index=False):
        text = data_prepocessing(row[3])
        corpus.append(word_tokenize(text))
        total_doc = total_doc + 1
    dictionary = build_dictionary(corpus)

    maxlist = []
    total = 0
    success = 0
    for row in df.itertuples(index=False):
        doc = nlp(row[3])
        orgs = {}
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                org = data_prepocessing(ent.text)
                if org in orgs.keys():
                    orgs[org] = orgs[org] + 1
                else:
                    orgs[org] = 1
        tfidf, max_tfidf = tf_idf(orgs, dictionary, total_doc)
        maxlist.append(max_tfidf)

        title = data_prepocessing(row[0])
        goal_orgs = ''
        max_count = 0
        for key, value in orgs.items():
            if key in title:
                if value > max_count:
                    goal_orgs = key

        if goal_orgs:
            success = success + 1
            organisations.append(['News', goal_orgs])
        else:
            if max_tfidf > 3.5:
                success = success + 1
                keys = [key for key, value in tfidf.items() if value == max_tfidf]
                organisations.append(['News', keys[0]])
        total = total + 1

    df = pd.DataFrame({columns[0]: organisations})
    df_expanded = df.reindex(columns=[*df.columns, *columns[1:]])
    df_expanded.fillna('None', inplace=True)
    df_expanded.to_csv(filename, encoding='utf-8', index=False)
    print(sum(maxlist), sum(maxlist) / len(maxlist))
    print(success / total)


def evaluation():
    df = pd.read_csv('/Users/northarbour/Downloads/bulk_export/organization_descriptions.csv', nrows=1000)

    corpus = []
    total_doc = 0
    for row in df.itertuples(index=False):
        text = data_prepocessing(row[8])
        corpus.append(word_tokenize(text))
        total_doc = total_doc + 1
    dictionary = build_dictionary(corpus)

    faillist = []
    successlist = []
    total = 0
    detection = 0
    success = 0
    for row in df.itertuples(index=False):
        output = ''
        doc = nlp(row[8])
        orgs = {}
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                org = data_prepocessing(ent.text)
                if org in orgs.keys():
                    orgs[org] = orgs[org] + 1
                else:
                    orgs[org] = 1
        tfidf, max_tfidf = tf_idf(orgs, dictionary, total_doc)

        title = data_prepocessing(row[0])
        goal_orgs = ''
        max_count = 0
        for key, value in orgs.items():
            if key in title:
                if value > max_count:
                    goal_orgs = key

        if goal_orgs:
            output = goal_orgs
        else:
            if max_tfidf > 3.5:
                keys = [key for key, value in tfidf.items() if value == max_tfidf]
                output = keys[0]
        if output:
            detection = detection + 1
            output = output.lower()
            label = row[1].lower()
            if output in label or label in output:
                success = success + 1
                successlist.append(max_tfidf)
            else:
                print(output + '    ' + row[1])
                print(max_tfidf)
                faillist.append(max_tfidf)
        total = total + 1
    if len(faillist) > 0:
        print(len(faillist), sum(faillist) / len(faillist), statistics.median(faillist))
    print(success, detection, total, success / detection)

    # 绘制第一个图表
    plt.subplot(2, 1, 1)  # 创建2行1列的子图布局，并选择第一个子图
    plt.hist(faillist, bins=50, edgecolor='black')  # 设置bins数量和边界颜色
    plt.xlabel('Value')  # 设置x轴标签
    plt.ylabel('TF-IDF')  # 设置y轴标签
    plt.title('Histogram of Fail TF-IDF')  # 设置标题

    # 绘制第二个图表
    plt.subplot(2, 1, 2)  # 创建2行1列的子图布局，并选择第二个子图
    plt.hist(successlist, bins=50, edgecolor='black')  # 设置bins数量和边界颜色
    plt.xlabel('Value')  # 设置x轴标签
    plt.ylabel('TF-IDF')  # 设置y轴标签
    plt.title('Histogram of Success TF-IDF')  # 设置标题

    # 调整子图之间的间距
    plt.subplots_adjust(hspace=0.5)

    # 显示图表
    plt.show()


extract_orgs('techcrunch_articles.csv')
# evaluation()
