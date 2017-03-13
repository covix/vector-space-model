#!/usr/bin/env python

import os
import nltk
from nltk import word_tokenize
import codecs
import config as cfg


def calc_score_matrix(terms):
    if not os.path.isdir(cfg.doc_path):
        raise IOError('Directory does not exist: %s' % cfg.doc_path)

    for doc in os.listdir(cfg.doc_path):
        scores = calc_scores(doc, terms)


def calc_scores(doc_name, terms):
    # Tokenize document
    # nltk.download()
    with codecs.open(cfg.doc_path + doc_name, encoding = 'ISO-8859-1') as doc_file:
        doc_string = doc_file.read().replace('\n', '')
        tokens = word_tokenize(doc_string)

    for term in terms:
        pass


if __name__ == '__main__':
    terms = sorted({term for value in cfg.users.itervalues() for term in value})
    matrix = calc_score_matrix(terms)
