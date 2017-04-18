#!/usr/bin/env python

import os
import argparse
from unicodedata import category

import config as cfg
import numpy as np
from time import time
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from nltk.corpus import brown

# parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
#               'tfidf__use_idf': (True, False),
#               'clf__alpha': (1e-2, 1e-3)}

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),
])

categories = brown.categories()

parameters = {
'vect__max_df': (0.5, 0.75, 1.0),
# 'vect__max_features': (None, 5000, 10000, 50000),
'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
# 'tfidf__use_idf': (True, False),
# 'tfidf__norm': ('l1', 'l2'),
'clf__alpha': (0.00001, 0.000001),
'clf__penalty': ('l2', 'elasticnet'),  # 'clf__n_iter': (10, 50, 80),
}

def analyse(filepath, clf):
    if not os.path.isfile(filepath):
        raise IOError('File does not exist: %s' % filepath)

    with open(filepath) as file:
        text = file.read().replace('\n', ' ')

        score_vec = clf.predict([text])
        return score_vec


def train():
    # Grid search to find best parameters
    # gs_clf = GridSearchCV(pipeline, parameters, n_jobs=-1)

    # print "Best score: %s" %(gs_clf.best_score_)
    # for param_name in sorted(parameters.keys()):
    #     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
    print "Training ..."
    t0 = time()
    data = [" ".join(brown.words(categories=category)) for category in categories]
    y = [i for i, _ in enumerate(categories)]
    clf = pipeline.fit(data, y)
    print("done in %0.3fs" % (time() - t0))

    # print("Best score: %0.3f" % gs_clf.best_score_)
    # print("Best parameters set:")
    # best_parameters = gs_clf.best_estimator_.get_params()
    # for param_name in sorted(parameters.keys()):
    #     print("\t%s: %r" % (param_name, best_parameters[param_name]))
    return clf


def assign_user(prediction, terms, users):
    '''
    Assigns the documents to those useres, that have an interest in this document.
    :param prediction: score for given term and document.
    :return: returns a collection of users.
    '''
    index = prediction[0]
    label = categories[index]

    relev_users = []
    for user in users:
        if label in users[user]:
            relev_users.append(user)
    return label, relev_users


if __name__ == '__main__':
    voc = sorted({term for value in cfg.users.itervalues() for term in value})

    parser = argparse.ArgumentParser(description='Text analysis tool for incoming documents')
    parser.add_argument('-p', '--path', dest='path', default=cfg.doc_path,
                        help='The filepath where the text file of the '
                             'document resides)')
    parser.add_argument('-v', '--vocabulary', dest='voc', default=voc, nargs='+', help='List of terms that can be '
                                                                                       'analysed.')

    args = parser.parse_args()
    clf = train()
    prediction = analyse(args.path, clf)

    found_term, users = assign_user(prediction, args.voc, cfg.users)
    print "RESULTS:"
    print "====================="
    print "The given text is about %s." % (found_term)
    if not users:
        print "The text is not relevant for any user."
    else:
        print "The text is relevant for the following users: %s." % (', '.join(users))
