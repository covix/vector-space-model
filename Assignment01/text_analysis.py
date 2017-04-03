#!/usr/bin/env python

import os
import argparse
import config as cfg
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def analyse(filepath, vocabulary):
    if not os.path.isfile(filepath):
        raise IOError('File does not exist: %s' % filepath)

    with open(filepath) as file:
        text = file.read().replace('\n', ' ')

        vect = TfidfVectorizer(use_idf=True, sublinear_tf=True, max_df=0.5, analyzer='word',
                               stop_words='english', vocabulary=vocabulary)

        score_vec = vect.fit_transform([text])
        return score_vec


def assign_user(vector, terms, users):
    '''
    Assigns the documents to those useres, that have an interest in this document.
    :param vector: score for given term and document.
    :return: returns a collection of users.
    '''
    index = np.argmax(vector[0])
    term = terms[index]

    relev_users = []
    for user in users:
        if term in users[user]:
            relev_users.append(user)
    return term, relev_users


if __name__ == '__main__':
    voc = sorted({term for value in cfg.users.itervalues() for term in value})

    parser = argparse.ArgumentParser(description='Text analysis tool for incoming documents')
    parser.add_argument('-p', '--path', dest='path', default=cfg.doc_path,
                        help='The filepath where the text file of the '
                             'document resides)')
    parser.add_argument('-v', '--vocabulary', dest='voc', default=voc, nargs='+', help='List of terms that can be '
                                                                                       'analysed.')

    args = parser.parse_args()
    vector = analyse(args.path, args.voc)

    found_term, users = assign_user(vector, args.voc, cfg.users)
    print "The given text is about %s." % (found_term)
    print "The text is relevant for the following users: %s." % (', '.join(users))
