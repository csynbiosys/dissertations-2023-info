import spacy
import pandas as pd
import re
import math
import statistics
import matplotlib.pyplot as plt
from datetime import datetime
import csv

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim import corpora

import data_preprocessing

# Load the list of stopwords in English
stopwords_list = stopwords.words('english')

# Regular expression pattern to match non-alphanumeric characters
pattern = r'[^a-zA-Z0-9\s]'

# Load the English language model from spaCy
nlp = spacy.load("en_core_web_sm")


def build_dictionary(corpus):
    """
       Build a dictionary from the given corpus and calculate word counts.

       Parameters:
           corpus (A list of list of str): A list of documents where each document is represented as a list of words.

       Returns:
           dict: A dictionary containing word counts for each word in the corpus.
       """
    # Build a dictionary from the given corpus
    dictionary = corpora.Dictionary(corpus)
    # Convert the corpus to Bag-of-Words representation
    corpus_bow = [dictionary.doc2bow(doc) for doc in corpus]
    # Serialize the corpus in MmCorpus format
    corpora.MmCorpus.serialize('corpus.mm', corpus_bow)
    # Load the serialized corpus
    mm_corpus = corpora.MmCorpus('corpus.mm')

    # Count the occurrences of each word in the corpus
    word_counts = {}
    for doc in mm_corpus:
        for word_id, count in doc:
            word = dictionary[word_id]
            word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts


def tf_idf(orgs, dictionary, total_doc):
    """
        Calculate the TF-IDF scores for all organizations extracted form one news in a given dictionary.

        Parameters:
            orgs (dict): A dictionary containing organization names as keys and their occurrence counts as values.
            dictionary (dict): A dictionary containing word counts for each word in the corpus.
            total_doc (int): Total number of documents in the corpus.

        Returns: tuple: A tuple containing two elements - a dictionary with TF-IDF scores for organizations and the
        maximum TF-IDF score.
    """

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


def bm_25(orgs, dictionary, doc_length, avg_doc_length, total_doc, k1=1.5, b=0.75):
    """
        Calculate the BM25 scores for all organizations extracted form one news based on their occurrence in the corpus.

        Parameters:
            orgs (dict): A dictionary containing organization names as keys and their occurrence counts as values.
            dictionary (dict): A dictionary containing word counts for each word in the corpus.
            doc_length (int): Length of the document.
            avg_doc_length (float): Average length of documents in the corpus.
            total_doc (int): Total number of documents in the corpus.
            k1 (float, optional): BM25 parameter k1. Default is 1.5.
            b (float, optional): BM25 parameter b. Default is 0.75.

        Returns: tuple: A tuple containing two elements - a dictionary with BM25 scores for organizations and the
        maximum BM25 score.
    """

    bm25 = {}
    for key, value in orgs.items():
        if key in dictionary:
            idf = math.log((total_doc - dictionary[key] + 0.5) / (dictionary[key] + 0.5) + 1.0)
            numerator = (value * (k1 + 1))
            denominator = (value + k1 * (1 - b + b * (doc_length / avg_doc_length)))
            bm25[key] = idf * (numerator / denominator)
    if bm25:
        max_bm25 = max(bm25.values())
        return bm25, max_bm25
    else:
        return bm25, 0


def org_in_text(org, text):
    """
        Check if the given organization name exists in the text using similarity comparison.

        Parameters:
            org (str): Organization name to be checked.
            text (str): Text where the organization name will be searched.

        Returns:
            bool: True if the organization name is found with high similarity, False otherwise.
    """

    org = nlp(org)
    text = text.split(' ')
    similarity = 0
    length = len(org)
    for i in range(len(text)):
        # 内部循环，遍历接下来的三个元素
        if i + length - 1 < len(text):
            subtext = nlp(''.join(text[i:i + length]))
            similarity_score = org.similarity(subtext)
            if similarity_score > similarity:
                similarity = similarity_score
        else:
            break
    if similarity > 0.8:
        return True
    else:
        return False


def data_prepocessing(phrase):
    """
        Preprocess the input phrase by removing special characters, possessive forms, stopwords, and tokenizing.

        Parameters:
            phrase (str): Input text to be preprocessed.

        Returns:
            str: Preprocessed text after removing special characters, stopwords, and tokenizing.
    """

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
    """
        Extract and save organization entities from a given file contains news body.
        This function extracts organization entities using various methods and saves the results in a CSV file.

        Parameters:
            filepath (str): Path to the input news body file.

    """

    # A list of organisation names
    organisations = []

    # Attributes names of the extracted dataset
    columns = data_preprocessing.attributes

    # Load Dataset
    filename = filepath.split('_')[0] + '_' + filepath.split('_')[1] + '_info.csv'
    df = pd.read_csv(filepath)

    # Build Dictionary
    corpus = []
    total_doc = 0
    for row in df.itertuples(index=False):
        text = data_prepocessing(row[3])
        corpus.append(word_tokenize(text))
        total_doc = total_doc + 1
    dictionary = build_dictionary(corpus)

    # Get a list of each doc's length
    doc_lengths = []
    for row in df.itertuples(index=False):
        doc_lengths.append(len(nlp(data_prepocessing(row[3]))))
    avg_doc_length = sum(doc_lengths) / len(doc_lengths)

    # Extract org names from each news
    # Total number of cases.
    total = 0
    threshold = 6

    for row in df.itertuples(index=False):
        output = ''
        doc = nlp(data_prepocessing(row[3]))
        orgs = {}
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                org = data_prepocessing(ent.text)
                if org in orgs.keys():
                    orgs[org] = orgs[org] + 1
                else:
                    orgs[org] = 1
        doc_length = len(doc)
        bm25, max_bm25 = bm_25(orgs, dictionary, doc_length, avg_doc_length, total_doc)

        title = data_prepocessing(row[0])
        # method 1
        # key = max(orgs, key=orgs.get, default='')
        # if key:
        #     max_count = orgs[key]
        #     goal_orgs = next((key for key, value in orgs.items() if key in title and value == max_count), '')
        # else:
        #     goal_orgs = ''
        goal_orgs = ''
        max_count = -1
        for key, value in orgs.items():
            if org_in_text(key, title):
                if value > max_count:
                    goal_orgs = key
                    max_count = value

        if goal_orgs:
            organisations.append(goal_orgs)
        else:
            if max_bm25 > threshold:
                keys = [key for key, value in bm25.items() if value == max_bm25]
                organisations.append(keys[0])
        total = total + 1

    # Store org names
    source = ['0'] * len(columns)
    source[0] = '4'
    source = ' '.join(source)
    df = pd.DataFrame({columns[0]: organisations})
    df_expanded = df.reindex(columns=[*df.columns, *columns[1:]])
    df_expanded.fillna('None', inplace=True)
    df_expanded['source'] = source
    df_expanded.to_csv(filename, encoding='utf-8', index=False)


def evaluation_frequency():
    """
        Evaluate the extraction results using frequency-based approach.

        This function evaluates the extraction results by comparing the detected organizations with the ground truth
        using frequency-based approach.
    """

    df = pd.read_csv('/Users/northarbour/Downloads/bulk_export/organization_descriptions.csv', nrows=100000)
    df = df.sample(n=5000, random_state=1)
    print('load success')
    total = 0
    detection = 0
    success = 0
    for row in df.itertuples(index=False):
        doc = nlp(data_prepocessing(row[8]))
        orgs = {}
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                org = data_prepocessing(ent.text)
                if org in orgs.keys():
                    orgs[org] = orgs[org] + 1
                else:
                    orgs[org] = 1
        output = max(orgs, key=orgs.get, default='')
        if output:
            detection = detection + 1
            output = output.lower()
            label = row[1].lower()
            if output in label or label in output:
                success = success + 1
        total = total + 1
    save_result(success, detection, total, 'frequency')


def evaluation_title_tfidf():
    """
        Evaluate the extraction results using title and TF-IDF approach.

        This function evaluates the extraction results by comparing the detected organizations in titles with the
        ground truth using TF-IDF approach.
    """

    df = pd.read_csv('/Users/northarbour/Downloads/bulk_export/organization_descriptions.csv', nrows=10000)

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
    threshold = 3.5
    for row in df.itertuples(index=False):
        output = ''
        doc = nlp(data_prepocessing(row[8]))
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
        # method 1
        # key = max(orgs, key=orgs.get, default='')
        # if key:
        #     max_count = orgs[key]
        #     goal_orgs = next((key for key, value in orgs.items() if key in title and value == max_count), '')
        # else:
        #     goal_orgs = ''
        # method 2
        goal_orgs = ''
        max_count = -1
        for key, value in orgs.items():
            if key in title:
                if value > max_count:
                    goal_orgs = key
                    max_count = value

        if goal_orgs:
            output = goal_orgs
        else:
            # #method 1
            # if max_tfidf > threshold:
            #     keys = [key for key, value in tfidf.items() if value == max_tfidf]
            #     output = keys[0]
            # method 2
            keys = [key for key, value in tfidf.items() if value == max_tfidf]
            if keys:
                output = keys[0]

        if output:
            detection = detection + 1
            output = output.lower()
            label = row[1].lower()
            if output in label or label in output:
                success = success + 1
                successlist.append(max_tfidf)
            else:
                faillist.append(max_tfidf)
        total = total + 1

    if len(faillist) > 0:
        print(len(faillist), sum(faillist) / len(faillist), statistics.median(faillist))
    # show_distribution(faillist, successlist)

    save_result(success, detection, total, 'title+tfidf')


def evaluation_title_bm25():
    """
        Evaluate the extraction results using title and BM25 approach.

        This function evaluates the extraction results by comparing the detected organizations in titles with the
        ground truth using BM25 approach.
    """

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
    threshold = 6

    doc_lengths = []
    for row in df.itertuples(index=False):
        doc_lengths.append(len(nlp(data_prepocessing(row[8]))))
    avg_doc_length = sum(doc_lengths) / len(doc_lengths)

    for row in df.itertuples(index=False):
        output = ''
        doc = nlp(data_prepocessing(row[8]))
        orgs = {}
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                org = data_prepocessing(ent.text)
                if org in orgs.keys():
                    orgs[org] = orgs[org] + 1
                else:
                    orgs[org] = 1
        doc_length = len(doc)
        bm25, max_bm25 = bm_25(orgs, dictionary, doc_length, avg_doc_length, total_doc)

        title = data_prepocessing(row[0])
        # method 1
        # key = max(orgs, key=orgs.get, default='')
        # if key:
        #     max_count = orgs[key]
        #     goal_orgs = next((key for key, value in orgs.items() if key in title and value == max_count), '')
        # else:
        #     goal_orgs = ''
        # method 2
        goal_orgs = ''
        max_count = -1
        for key, value in orgs.items():
            if org_in_text(key, title):
                if value > max_count:
                    goal_orgs = key
                    max_count = value

        if goal_orgs:
            output = goal_orgs
        else:
            # method 1
            if max_bm25 > threshold:
                keys = [key for key, value in bm25.items() if value == max_bm25]
                output = keys[0]
            # method 2
            # keys = [key for key, value in bm25.items() if value == max_bm25]
            # if keys:
            #     output = keys[0]

        if output:
            detection = detection + 1
            output = output.lower()
            label = row[1].lower()
            if output in label or label in output:
                success = success + 1
                successlist.append(max_bm25)
            else:
                faillist.append(max_bm25)
        total = total + 1

    if len(faillist) > 0:
        print(len(faillist), sum(faillist) / len(faillist), statistics.median(faillist))
    show_distribution(faillist, successlist)

    save_result(success, detection, total, 'title+bm25')


def evaluation_tfidf():
    """
        Evaluate the extraction results using TF-IDF approach.

        This function evaluates the extraction results by comparing the detected organizations with the ground truth
        using TF-IDF approach.
    """

    df = pd.read_csv('/Users/northarbour/Downloads/bulk_export/organization_descriptions.csv', nrows=4000)

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
    threshold = 2.5
    for row in df.itertuples(index=False):
        output = ''
        doc = nlp(data_prepocessing(row[8]))
        orgs = {}
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                org = data_prepocessing(ent.text)
                if org in orgs.keys():
                    orgs[org] = orgs[org] + 1
                else:
                    orgs[org] = 1
        tfidf, max_tfidf = tf_idf(orgs, dictionary, total_doc)
        # # method 1
        # if max_tfidf > threshold:
        #     keys = [key for key, value in tfidf.items() if value == max_tfidf]
        #     output = keys[0]
        # method 2
        keys = [key for key, value in tfidf.items() if value == max_tfidf]
        if keys:
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
    show_distribution(faillist, successlist)

    save_result(success, detection, total, 'tfidf')


def evaluation_bm25():
    """
       Evaluate the extraction results using BM25 approach.

       This function evaluates the extraction results by comparing the detected organizations with the ground truth
       using BM25 approach.
    """

    df = pd.read_csv('/Users/northarbour/Downloads/bulk_export/organization_descriptions.csv', nrows=10000)

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
    threshold = 2.5

    # 获得文档长度
    doc_lengths = []
    for row in df.itertuples(index=False):
        doc_lengths.append(len(nlp(data_prepocessing(row[8]))))
    avg_doc_length = sum(doc_lengths) / len(doc_lengths)

    for row in df.itertuples(index=False):
        output = ''
        doc = nlp(data_prepocessing(row[8]))
        orgs = {}
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                org = data_prepocessing(ent.text)
                if org in orgs.keys():
                    orgs[org] = orgs[org] + 1
                else:
                    orgs[org] = 1
        doc_length = len(doc)
        bm25, max_bm25 = bm_25(orgs, dictionary, doc_length, avg_doc_length, total_doc)
        # # method 1
        if max_bm25 > threshold:
            keys = [key for key, value in bm25.items() if value == max_bm25]
            output = keys[0]
        # method 2
        # keys = [key for key, value in bm25.items() if value == max_bm25]
        # if keys:
        #     output = keys[0]

        if output:
            detection = detection + 1
            output = output.lower()
            label = row[1].lower()
            if output in label or label in output:
                success = success + 1
                successlist.append(max_bm25)
            else:
                print(output + '    ' + row[1])
                print(max_bm25)
                faillist.append(max_bm25)
        total = total + 1

    if len(faillist) > 0:
        print(len(faillist), sum(faillist) / len(faillist), statistics.median(faillist))
    show_distribution(faillist, successlist)

    save_result(success, detection, total, 'bm25')


def show_distribution(faillist, successlist):
    """
        Display two histograms showing the distribution of TF-IDF values for failed and successful cases.

        Parameters:
            faillist (list): List of TF-IDF values for failed cases.
            successlist (list): List of TF-IDF values for successful cases.
        """
    # Plot the first histogram
    plt.subplot(2, 1, 1)
    plt.hist(faillist, bins=50, edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('TF-IDF')
    plt.title('Histogram of Fail TF-IDF')

    # Plot the second histogram
    plt.subplot(2, 1, 2)
    plt.hist(successlist, bins=50, edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('TF-IDF')
    plt.title('Histogram of Success TF-IDF')

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.5)

    # Display the plots
    plt.show()


def f1_score(success, detection, total):
    """
        Calculate the F1 score based on the success, detection, and total count.

        Parameters:
            success (int): Number of successful cases.
            detection (int): Number of detected cases.
            total (int): Total number of cases.

        Returns:
            float: Calculated F1 score.
        """
    precision = success / detection
    recall = success / (success + total - detection)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def save_result(success, detection, total, method):
    """
    Save the evaluation results to a CSV file.

    Parameters:
        success (int): Number of successful cases.
        detection (int): Number of detected cases.
        total (int): Total number of cases.
        method (str): Method used for evaluation.
    """

    print('total', 'success', 'detection', 'precision', 'recall', 'R1')
    print(total, success, detection, success / detection, success / total, f1_score(success, detection, total))
    title = [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ' News NER', method]
    format = ['total', 'success', 'detection', 'precision', 'recall', 'R1']
    result = [total, success, detection, success / detection, success / total, f1_score(success, detection, total)]
    with open('../result.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(title)
        writer.writerow(format)
        writer.writerow(result)