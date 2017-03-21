#!/usr/bin/env python

import os
from nltk import word_tokenize
import codecs
import config as cfg
import math


def calc_scores(terms):
    '''
    The Tf-idf measure is calculated for the given terms and all the documents placed in the specified directory. The
    path of the directory must be set in the config.py file.

    :param terms: all the terms at least one user has interests in.
    :return: tf-idf measure for all terms and documents.
    '''
    if not os.path.isdir(cfg.doc_path):
        raise IOError('Directory does not exist: %s' % cfg.doc_path)
    docs = os.listdir(cfg.doc_path)
    num_docs = float(len(docs))
    tf = []
    for doc in docs:
        # Tokenize document
        # nltk.download()
        with codecs.open(cfg.doc_path + doc, encoding='ISO-8859-1') as doc_file:
            doc_string = doc_file.read().replace('\n', '')
            tokens = word_tokenize(doc_string)

            # TODO: Frequency for 'data science' cannot be found because it is two words and we loop over single words.
            term_frequency(terms, tf, tokens)

    idf = invers_document_freq(num_docs, terms, tf)

    tf_idf = score(docs, idf, terms, tf)

    return tf_idf


def term_frequency(terms, tf, tokens):
    '''
    Calculates the term frequency for the tokens of a given document
    :param terms: terms to calculate the frequency for
    :param tf:
    :param tokens: tokens of document
    :return: dictionary with frequencies of tokens
    '''
    term_freq = dict()
    for token in tokens:
        if token in terms:
            for term in terms:
                if term == token:
                    term_freq[term] = term_freq.get(term, 0) + 1
    tf.append(term_freq)


def score(docs, idf, terms, tf):
    '''

    :param docs:
    :param idf:
    :param terms:
    :param tf:
    :return:
    '''
    tf_idf = dict()
    for d in range(len(docs)):
        for term in terms:
            if term not in tf[d]:
                tf_idf[term, d] = 0 * idf[term]
            else:
                tf_idf[term, d] = math.log(1 + tf[d][term]) * idf[term]
    return tf_idf

def invers_document_freq(num_docs, terms, tf):
    '''

    :param num_docs:
    :param terms:
    :param tf:
    :return:
    '''
    idf = {}
    for term in terms:
        df = len([term_freq[term] for term_freq in tf if term in term_freq.keys()])
        if df == 0:
            idf[term] = 0
        else:
            idf[term] = math.log(num_docs / df)
    return idf

def assign_documents(tf_idf):
    num_docs = len(os.listdir(cfg.doc_path))
    docs_for_user = {}
    for user in cfg.users:
        scores =[]
        for i in range(num_docs):
            score = 0
            for interest in cfg.users[user]:
                score += tf_idf[(interest, i)]
            scores.append(score)

        #scores_user[user] = scores
        relevant_docs = []
        for doc_index, s in enumerate(scores):
            if s >= cfg.threshold:
                relevant_docs.append(doc_index)
        docs_for_user[user] = relevant_docs
    return docs_for_user




if __name__ == '__main__':
    terms = sorted({term for value in cfg.users.itervalues() for term in value})
    scores = calc_scores(terms)
    print assign_documents(scores)
