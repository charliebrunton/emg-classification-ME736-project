import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# UTILITY
def _save_or_show(save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()

# PLOTTING
def plot_raw_signal(signal, fs, title="Raw EMG", save_path=None):
    """Plot raw EMG signal (single channel)."""
    t = np.arange(len(signal)) / fs
    plt.figure(figsize=(10, 4))
    plt.plot(t, signal, label="TA")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    _save_or_show(save_path)

def plot_predictions(preds, title="Predicted States", save_path=None):
    """Plot classifier predictions over time (discrete states)."""
    plt.figure(figsize=(10, 4))
    plt.plot(preds, drawstyle="steps-post")
    plt.xlabel("Window index")
    plt.ylabel("Class label")
    plt.title(title)
    plt.tight_layout()
    _save_or_show(save_path)

def plot_confusion(y_true, y_pred, classes=None, title="Confusion Matrix", save_path=None):
    """Plot confusion matrix for predictions vs. true labels."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.tight_layout()
    _save_or_show(save_path)

def plot_feature_space(X, y, title="Feature Space (RMS vs WL)", save_path=None):
    """
    Plot RMS vs WL feature space with LDA decision boundaries.
    Assumes features[:,0] = RMS, features[:,1] = WL.
    """
    X = np.array(X)
    y = np.array(y)

    if X.shape[1] < 2:
        raise ValueError("Need at least 2 features (RMS and WL) to plot feature space")

    clf = LinearDiscriminantAnalysis()
    clf.fit(X[:, :2], y)  # only use first two features

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_pad, y_pad = 0.1 * (x_max - x_min), 0.1 * (y_max - y_min)

    xx, yy = np.meshgrid(
        np.linspace(x_min - x_pad, x_max + x_pad, 200),
        np.linspace(y_min - y_pad, y_max + y_pad, 200)
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(6, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")
    plt.xlabel("RMS")
    plt.ylabel("Waveform Length")
    plt.title(title)
    plt.legend(*scatter.legend_elements(), title="Class")
    plt.tight_layout()
    _save_or_show(save_path)
