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
            tf.append(term_frequency(terms, tokens))

    idf = invers_document_freq(num_docs, terms, tf)

    tf_idf = score(docs, idf, terms, tf)

    return tf_idf


def term_frequency(terms, tokens):
    '''
    Calculates the term frequency for the tokens of a given document
    :param terms: terms to calculate the frequency for
    :param tokens: tokens of document
    :return: dictionary with frequencies of tokens
    '''
    term_freq = dict()
    for token in tokens:
        if token in terms:
            for term in terms:
                if term == token:
                    term_freq[term] = term_freq.get(term, 0) + 1
    return term_freq


def score(docs, idf, terms, tf):
    '''
    Calculates the actual score by multiplying tf and idf.
    :param docs: list of documents
    :param idf: collection of inverse document frequencies
    :param terms: list of terms
    :param tf: collection of term frequencies
    :return: tf-idf
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
    Calculates the inverse document frequency (IDF).
    :param num_docs: number of documents.
    :param terms: list of terms
    :param tf: collection of term frequencies.
    :return: idf
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
    '''
    Assigns each user the documents of interest. To do this a threshold is used. In case the score of a
    document is higher than the threshold, it is considered relevant for the user. Documents are represented by their
    ID.
    :param tf_idf: score for given term and document.
    :return: returns a collection of users with their documents of interest.
    '''
    num_docs = len(os.listdir(cfg.doc_path))
    docs_for_user = {}
    for user in cfg.users:
        scores = []
        for i in range(num_docs):
            score = 0
            for interest in cfg.users[user]:
                score += tf_idf[(interest, i)]
            scores.append(score)

        # scores_user[user] = scores
        relevant_docs = []
        for doc_index, s in enumerate(scores):
            if s >= cfg.threshold:
                relevant_docs.append(doc_index)
        docs_for_user[user] = relevant_docs
    return docs_for_user


if __name__ == '__main__':
    terms = sorted({term for value in cfg.users.itervalues() for term in value})
    scores = calc_scores(terms)
    res = assign_documents(scores)
    print res
