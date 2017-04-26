#!/usr/bin/env python2

import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score

from utils import load_data, load_model


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        print "Normalized confusion matrix"
    else:
        print 'Confusion matrix, without normalization'

    print cm

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == "__main__":
    model_name = sys.argv[1]

    train_data = load_data(subset='train')
    test_data = load_data(subset='test')
    clf = load_model(model_name)

    X_train, y_train = train_data.data, train_data.target
    X_test, y_test = test_data.data, test_data.target

    n_folds = 5
    print "Starting %s-folds cross validation..." % n_folds
    # cross_val_scores refit the classifier on each fold (even if it was already fitted)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=n_folds)
    print "Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2)

    # refitting the classifier only on training data
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print classification_report(y_test, y_pred, target_names=train_data.target_names)
    print "Accuracy on 20 newspaper test data: %0.2f" % accuracy_score(y_test, y_pred)

    np.set_printoptions(precision=2)
    cnf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=train_data.target_names,
                          title='Confusion matrix, without normalization')

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=train_data.target_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()


# clf__alpha: 1e-05
# 	clf__penalty: 'l2'
# 	tfidf__use_idf: True
# 	vect__max_df: 0.5
# 	vect__ngram_range: (1, 2)
