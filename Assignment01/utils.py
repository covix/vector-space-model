import os

import numpy as np
from nltk import word_tokenize
from nltk.stem.snowball import EnglishStemmer
from sklearn.datasets import fetch_20newsgroups
from sklearn.externals import joblib

import config as cfg


def tokenize(text, stem=False):
    tokens = [word for word in word_tokenize(text) if word.isalpha()]

    if stem:
        stemmer = EnglishStemmer()
        tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)


def load_data(clean=True, subset='train'):
    print "Loading %s data..." % subset
    data = fetch_20newsgroups(subset=subset, categories=cfg.categories)
    new_target = np.zeros_like(data.target)
    new_target_names = list(set(cfg.new_categories.values()))
    new_categories_index = {i: idx for idx, i in enumerate(new_target_names)}

    for oc, nc in cfg.new_categories.iteritems():
        for idx, i in enumerate(data.target_names):
            if oc in i:
                new_target[data.target == idx] = new_categories_index[nc]

    data.target = new_target
    data.target_names = new_target_names


    for i in range(len(data.data)):
        if clean:
            data.data[i] = tokenize(data.data[i])
        data.data[i] = data.data[i].lower()

    return data


def load_target_names():
    data = load_data(clean=False)
    return data.target_names


def load_model(model_name):
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = cfg.model_path + model_name

    print "Loading model ", model_path
    clf = joblib.load(model_path)

    return clf
