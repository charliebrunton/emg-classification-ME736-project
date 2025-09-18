from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from processing import segment_stream

def train_lda(train_features, train_labels, test_size=0.3, random_state=33):
    X_train, X_test, y_train, y_test = train_test_split(
        train_features, train_labels, test_size=test_size, random_state=random_state
    )
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    return clf, acc

def cross_validate(clf, X, y, k=5):
    cv = KFold(n_splits=k, shuffle=True, random_state=33)
    scores = cross_val_score(clf, X, y, cv=cv)
    return scores.mean() * 100, scores

# TEST CONTINUOUS STREAM (RTOS SIM)
def test_stream(clf, signal, fs, window_ms=200, overlap=0.5):
    """
    Run classifier on a continuous EMG stream.

    Args:
        clf: trained classifier
        signal: raw 1D EMG signal
        fs: sampling frequency
        window_ms: window size in milliseconds
        overlap: fraction of overlap between consecutive windows
        return_probs: if True, also return probability distributions

    Returns:
        preds (np.ndarray): predicted class for each window
        probs (np.ndarray): class probabilities for each window (if return_probs=True)
    """
    # segment into windows + features
    X = segment_stream(signal, fs, window_ms, overlap)

    # predict
    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    return preds, probs
