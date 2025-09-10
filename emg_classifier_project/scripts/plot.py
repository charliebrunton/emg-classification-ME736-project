import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scripts.features import rms, wl

def preprocess(epoch):
    # subtract mean to account for DC offset (drift)
    return epoch - np.mean(epoch)

# plotting functions taken from ChatGPT
def plot_raw_signals(ta_data, gas_data, fs, save_path=None):
    t = np.arange(len(ta_data)) / fs
    plt.figure(figsize=(10, 5))
    plt.plot(t, ta_data, label="TA")
    plt.plot(t, gas_data, label="GAS")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Raw EMG Signals")
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()

def plot_preprocessed_signals(ta_data, gas_data, fs, save_path=None):
    t = np.arange(len(ta_data)) / fs
    plt.figure(figsize=(10, 5))
    plt.plot(t, preprocess(ta_data), label="TA")
    plt.plot(t, preprocess(gas_data), label="GAS")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Preprocessed EMG Signals (DC offset removed)")
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()

def plot_feature_space(features, labels, title="Feature Space", save_path=None):
    # only use rms and wl features
    rms_vals = [rms(f) for f in features]
    wl_vals = [wl(f) for f in features]

    X = np.column_stack((rms_vals, wl_vals))
    y = labels

    clf = LinearDiscriminantAnalysis()
    clf.fit(X, y)

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_pad = 0.1 * (x_max - x_min)
    y_pad = 0.1 * (y_max - y_min)
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(6, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")
    plt.xlabel("RMS")
    plt.ylabel("Waveform Length")
    plt.title(title)
    plt.legend(*scatter.legend_elements(), title="Class")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()
