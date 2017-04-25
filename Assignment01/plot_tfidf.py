#!/usr/bin/env python2

import numpy as np

from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt

from text_analysis import load, load_data

import sys


if __name__ == '__main__':
    model_name = sys.argv[1]
    clf, target_names = load(model_name)
    data = load_data()

    target_labels = np.array([data.target_names[i] for i in data.target])

    vect = clf.steps[0][1]
    tfidf = clf.steps[1][1]

    vectors = tfidf.transform(vect.transform(data.data))
    X_reduced = TruncatedSVD(
        n_components=50, random_state=0).fit_transform(vectors)

    X_embedded = TSNE(n_components=2, perplexity=40,
                      verbose=2).fit_transform(X_reduced)

    for i in np.unique(data.target):
        X = X_embedded[data.target == i]
        plt.scatter(X[:, 0], X[:, 1], s=8, label=target_names[i], alpha=0.5)

    plt.legend(scatterpoints=1)
    plt.axes().set_aspect('equal', 'datalim')

    plt.show()


    
