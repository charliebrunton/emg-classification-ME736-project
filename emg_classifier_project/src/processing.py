import numpy as np
import scipy.io.wavfile as wav
from features import extract_features

# LOAD
def load_epochs(path, epoch_time, n_classes=4, channel=0):
    """
    Load EMG from wav file and cut into fixed-length epochs with class labels.
    
    Args:
        path (str): path to wav file
        epoch_time (int): epoch length in seconds
        n_classes (int): number of classes (labels cycle 0..n_classes-1)
        channel (int): which channel to use (default: 0 = TA)
    
    Returns:
        fs (int): sampling rate
        epochs (list of np.ndarray): list of 1D signal arrays
        labels (np.ndarray): class label for each epoch
    """
    fs, data = wav.read(path)
    signal = data[:,0] if data.ndim > 1 else data

    epoch_length = epoch_time * fs
    n_epochs = len(signal) // epoch_length
    signal = signal[:n_epochs * epoch_length]  # trim extras

    epochs = [signal[i*epoch_length:(i+1)*epoch_length] for i in range(n_epochs)]
    labels = np.arange(n_epochs) % n_classes

    return fs, epochs, labels

# FOR TRAIN DATA
def segment_windows(epochs, labels, fs, window_ms=200, overlap=0.5):
    """
    Break each epoch into overlapping windows and extract features.
    
    Args:
        epochs (list of np.ndarray): list of 1D signals
        labels (np.ndarray): class label for each epoch
        fs (int): sampling rate
        window_ms (int): window length in ms
        overlap (float): fraction overlap (0–1)
    
    Returns:
        X (np.ndarray): feature matrix
        y (np.ndarray): labels per window
    """
    win_len = int(fs * window_ms / 1000)
    step = int(win_len * (1 - overlap))

    X, y = [], []
    for epoch, label in zip(epochs, labels):
        for start in range(0, len(epoch) - win_len, step):
            window = epoch[start:start+win_len]
            feats = extract_features([window])[0]  # wrap in list, unpack
            X.append(feats)
            y.append(label)
    return np.array(X), np.array(y)

# FOR TEST DATA
def segment_stream(signal, fs, window_ms=200, overlap=0.5):
    """
    Segment continuous EMG signal into overlapping windows and extract features.
    Useful for real-time testing where labels are unknown.
    
    Args:
        signal (np.ndarray): continuous EMG
        fs (int): sampling rate
        window_ms (int): window length in ms
        overlap (float): fraction overlap (0–1)
    
    Returns:
        X (np.ndarray): feature matrix for each window
    """
    win_len = int(fs * window_ms / 1000)
    step = int(win_len * (1 - overlap))

    X = []
    for start in range(0, len(signal) - win_len, step):
        window = signal[start:start+win_len]
        feats = extract_features([window])[0]
        X.append(feats)
    return np.array(X)
