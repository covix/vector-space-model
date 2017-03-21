#!/usr/bin/env python

import os
import nltk
from nltk import word_tokenize
import codecs
import config as cfg
import math


def calc_scores(terms):
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
    term_freq = dict()
    for token in tokens:
        if token in terms:
            for term in terms:
                if term == token:
                    term_freq[term] = term_freq.get(term, 0) + 1
    tf.append(term_freq)


def score(docs, idf, terms, tf):
    tf_idf = dict()
    for d in range(len(docs)):
        for term in terms:
            if term not in tf[d]:
                tf_idf[term, d] = 0 * idf[term]
            else:
                tf_idf[term, d] = math.log(1 + tf[d][term]) * idf[term]
    return tf_idf


def invers_document_freq(num_docs, terms, tf):
    idf = {}
    for term in terms:
        df = len([term_freq[term] for term_freq in tf if term in term_freq.keys()])
        if df == 0:
            idf[term] = 0
        else:
            idf[term] = math.log(num_docs / df)
    return idf


if __name__ == '__main__':
    terms = sorted({term for value in cfg.users.itervalues() for term in value})
    matrix = calc_scores(terms)
