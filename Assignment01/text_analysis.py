#!/usr/bin/env python2

import argparse
import logging
import os
import sys
from pprint import pprint
from time import time, strftime

from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import config as cfg
from utils import load_model, load_target_names, tokenize, load_data

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def analyse(filepath, clf):
    if not os.path.exists(filepath):
        raise IOError('File does not exist: %s' % filepath)

    if os.path.isdir(filepath):
        names = os.listdir(filepath)
        files = [os.path.join(filepath, i) for i in names]
    else:
        files = [filepath]
        names = [os.path.basename(filepath)]

    texts = []
    print "Loading files..."
    for path in files:
        print "\tLoading", path
        with open(path) as f:
            text = f.read().replace('\n', ' ').decode('latin-1')
            text = tokenize(text)
            texts.append(text)

    scores = clf.predict(texts)

    return scores, names


def train(save=False):
    print "Training ..."

    data = load_data()

    # TfidfVectorizer combines all the options of CountVectorizer and TfidfTransformer in a single model:
    pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words='english')),
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

        file_path = "{}model_{}.model".format(
            cfg.model_path, strftime("%Y%m%d_%H%M%S"))
        joblib.dump(clf, file_path, compress=9)

    return clf


def assign_user(predictions, users, labels):
    """
    Assigns the documents to those useres, that have an interest in this document.
    :param predictions: score for given term and document.
    :param users: dict in the form of { user_name : list_of_interests }
    :param labels: labels of the documents
    :return: returns a collection of users.
    """
    topics, interested_users = [], []
    for prediction in predictions:
        label = labels[prediction]

        u = []
        for user in users:
            if label in users[user]:
                u.append(user)

        topics.append(label)
        interested_users.append(u)

    return topics, interested_users


def main():
    parser = argparse.ArgumentParser(description='Text analysis tool for incoming documents')
    parser.add_argument('-d', '--doc', dest='doc', required=True,
                        help='The filepath where the text file of the document resides)')
    parser.add_argument('-s', '--save', dest='save', action='store_true',
                        help='If set, model will be saved after execution.')
    parser.add_argument('-m', '--model', dest='model', help='The filepath to the previously saved model.')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    clf = load_model(args.model) if args.model else train(args.save)
    labels = load_target_names()

    predictions, names = analyse(args.doc, clf)

    topics, users = assign_user(predictions, cfg.users, labels)
    print "RESULTS:"
    for i in range(len(names)):
        print "====================="
        print "%s is about %s." % (names[i], topics[i])
        if not users[i]:
            print "The text is not relevant for any user."
        else:
            print "The text is relevant for the following users: %s." % (', '.join(users[i]))


if __name__ == '__main__':
    main()
