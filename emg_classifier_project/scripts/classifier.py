import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def train(train_features, train_labels):
    clf = LinearDiscriminantAnalysis()
    clf.fit(train_features, train_labels)
    return clf

def evaluate(clf, test_features, test_labels):
    preds = clf.predict(test_features)
    acc = accuracy_score(test_labels, preds)
    return acc, preds

def predict_probs(model, test_set):
    # probs -> probability distribution over classes for each sample, shape (n_samples, n_classes)
    probs = model.predict_proba(test_set)    
    return probs

# LDA produces probs that are too confident -> try logistic regression for softer probs
def train_logreg(train_features, train_labels):
    # clf = LogisticRegression(max_iter=1000) # convergeance issues
    clf = LogisticRegression(max_iter=10000, solver="saga") # seems to be a solid middle ground
    # clf = make_pipeline(
    #     StandardScaler(),
    #     LogisticRegression(max_iter=5000, solver="lbfgs")
    # ) # accuracy issues
    clf.fit(train_features, train_labels)
    return clf
