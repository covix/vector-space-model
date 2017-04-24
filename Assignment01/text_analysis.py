#!/usr/bin/env python2

import os
import argparse
from unicodedata import category

import config as cfg
import numpy as np
from time import time, strftime
import sys
from pprint import pprint
import logging

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from sklearn.datasets import fetch_20newsgroups

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# from nltk.corpus import brown

# parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
#               'tfidf__use_idf': (True, False),
#               'clf__alpha': (1e-2, 1e-3)}


def analyse(filepath, clf):
    if not os.path.isfile(filepath):
        raise IOError('File does not exist: %s' % filepath)

    with open(filepath) as file:
        text = file.read().replace('\n', ' ')

        score_vec = clf.predict([text])
        return score_vec


def load(model_name):
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = cfg.model_path + model_name

    print "Loading model ", model_path

    clf = joblib.load(model_path)
    return clf


def train(save=False):
    print "Training ..."

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        # ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),
        ('clf', SGDClassifier()),
    ])

    # categories = brown.categories()

    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (0.00001, 0.000001),
        'clf__penalty': ('l2', 'elasticnet'),  # 'clf__n_iter': (10, 50, 80),
    }
    
    # data = [" ".join(brown.words(categories=category)) for category in cfg.categories]
    # # y = [i for i, _ in enumerate(categories)]
    # y = range(len(categories))
    data = fetch_20newsgroups(subset='train', categories=cfg.categories)

    # Grid search to find best parameters
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(data.data, data.target)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
    # Previous
    # clf = pipeline.fit(data, y)
    # print("done in %0.3fs" % (time.time() - t0))

    clf = grid_search.best_estimator_

    if save:
        if not os.path.exists(cfg.model_path):
            os.makedirs(cfg.model_path)

        file_path = "{}model_{}.model".format(cfg.model_path, strftime("%Y%m%d_%H%M%S"))
        joblib.dump(clf, file_path, compress=9)

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
    label = cfg.categories[index]

    relev_users = []
    for user in users:
        if label in users[user]:
            relev_users.append(user)
    return label, relev_users


if __name__ == '__main__':
    voc = sorted({term for value in cfg.users.itervalues() for term in value})

    parser = argparse.ArgumentParser(description='Text analysis tool for incoming documents')
    parser.add_argument('-d', '--doc', dest='doc', required=True,
                        help='The filepath where the text file of the document resides)')
    parser.add_argument('-v', '--vocabulary', dest='voc', default=voc, nargs='+', 
                        help='List of terms that can be analysed.')

    parser.add_argument('-s', '--save', dest='save', action='store_true', default=True,
                        help='If set, model will be saved after execution.')

    parser.add_argument('-m', '--model', dest='model', help='The filepath to the previously saved model.')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    if args.model:
        clf = load(args.model)
    else:
        clf = train(args.save)

    prediction = analyse(args.doc, clf)

    found_term, users = assign_user(prediction, args.voc, cfg.users)
    print "RESULTS:"
    print "====================="
    print "The given text is about %s." % (found_term)
    if not users:
        print "The text is not relevant for any user."
    else:
        print "The text is relevant for the following users: %s." % (', '.join(users))
