import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

def train(train_features, train_labels):
    clf = LinearDiscriminantAnalysis()
    clf.fit(train_features, train_labels)
    return clf

def evaluate(clf, test_features, test_labels):
    preds = clf.predict(test_features)
    acc = accuracy_score(test_labels, preds)
    return acc, preds
 