import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, KFold, LeavePOut, cross_val_score
from sklearn.metrics import accuracy_score
from features import extract_features

# TRAINING/EVALUATION
def train_lda(train_features, train_labels, test_size=0.3, random_state=33):
    X_train, X_test, y_train, y_test = train_test_split(
        train_features, train_labels, test_size=test_size, random_state=random_state
    )
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    return clf, acc

def evaluate(clf, test_features, test_labels):
    """Return accuracy and predictions"""
    preds = clf.predict(test_features)
    acc = accuracy_score(test_labels, preds)
    return acc, preds

def predict_probs(clf, test_set):
    """Return probability distribution over classes"""
    return clf.predict_proba(test_set)

# RTOS TESTING
def test_stream(clf, signal, fs, window_ms=200, overlap=0.5, return_probs=False):
    win_len = int(fs * window_ms / 1000)
    step = int(win_len * (1 - overlap))

    preds, probs = [], []
    for start in range(0, len(signal) - win_len, step):
        window = signal[start:start+win_len]
        feats = extract_features([window])[0]
        pred = clf.predict([feats])[0]
        preds.append(pred)

        if return_probs and hasattr(clf, "predict_proba"):
            probs.append(clf.predict_proba([feats])[0])

    if return_probs:
        return np.array(preds), np.array(probs)
    else:
        return np.array(preds)

# CROSS-VALIDATION WRAPPER
def cross_validate(clf, X, y, method="kfold", k=5):
    if method == "lpo":
        # WARNING: extremely slow for large datasets
        cv = LeavePOut(p=2)
    else:  # default = kfold
        cv = KFold(n_splits=k, shuffle=True, random_state=42)

    scores = cross_val_score(clf, X, y, cv=cv)
    return scores.mean() * 100, scores

# PROBABILITY TRACING
def predict_probs(clf, X):
    """
    Return probability distribution over classes for each sample.
    Shape: (n_samples, n_classes)
    """
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    else:
        raise ValueError("Classifier does not support probability outputs.")
