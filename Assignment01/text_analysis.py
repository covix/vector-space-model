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


def load_data():
    new_categories = {
        'comp': 'computer',
        'rec': 'sports',
        'sci': 'science',
        'religion': 'religion',
        'politics': 'politics',
    }

    data = fetch_20newsgroups(subset='train', categories=cfg.categories)
    new_target = np.zeros_like(data.target)
    new_target_names = new_categories.values()
    new_categories_index = {i: idx for idx, i in enumerate(new_target_names)}

    for oc, nc in new_categories.iteritems():
        for idx, i in enumerate(data.target_names):
            if oc in i:
                new_target[data.target == idx] = new_categories_index[nc]

    data.target = new_target
    data.target_names = new_target_names

    return data


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
    data = load_data()

    return clf, data.target_names


def train(save=False):
    print "Training ..."

    data = load_data()

    # TfidfVectorizer combines all the options of CountVectorizer and TfidfTransformer in a single model:
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        # ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),
        ('clf', SGDClassifier()),
    ])

    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (0.00001, 0.000001),
        'clf__penalty': ('l2', 'elasticnet'),  # 'clf__n_iter': (10, 50, 80),
    }
    
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
    
    clf = grid_search.best_estimator_

    if save:
        if not os.path.exists(cfg.model_path):
            os.makedirs(cfg.model_path)

        file_path = "{}model_{}.model".format(cfg.model_path, strftime("%Y%m%d_%H%M%S"))
        joblib.dump(clf, file_path, compress=9)

    return clf, data.target_names


def assign_user(prediction, terms, users, labels):
    '''
    Assigns the documents to those useres, that have an interest in this document.
    :param prediction: score for given term and document.
    :return: returns a collection of users.
    '''
    index = prediction[0]
    label = labels[index]

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

    parser.add_argument('-s', '--save', dest='save', action='store_true',
                        help='If set, model will be saved after execution.')

    parser.add_argument('-m', '--model', dest='model', help='The filepath to the previously saved model.')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    clf, labels = load(args.model) if args.model else train(args.save)

    prediction = analyse(args.doc, clf)

    found_term, users = assign_user(prediction, args.voc, cfg.users, labels)
    print "RESULTS:"
    print "====================="
    print "The given text is about %s." % (found_term)
    if not users:
        print "The text is not relevant for any user."
    else:
        print "The text is relevant for the following users: %s." % (', '.join(users))
