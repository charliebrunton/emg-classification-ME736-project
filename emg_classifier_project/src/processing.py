import numpy as np
import scipy.io.wavfile as wav
from features import extract_features

def load_epochs(path, epoch_time, n_classes=4, channel=0):
    """
    Load EMG from wav file and cut into fixed-length epochs with class labels
    
    Returns:
        fs (int): sampling rate
        epochs (list of np.ndarray): list of epochs
        labels (npndarray): class label for each epoch
    """
    fs, data = wav.read(path)
    signal = data[:,channel] # channel = 0 -> TA recording only

    epoch_length = epoch_time * fs
    n_epochs = len(signal) // epoch_length
    signal = signal[:n_epochs * epoch_length]  # trim extras

    epochs = [signal[i*epoch_length:(i+1)*epoch_length] for i in range(n_epochs)]
    labels = np.arange(n_epochs) % n_classes

    return fs, epochs, labels

# FOR TRAINING EPOCHS
def segment_windows(epochs, labels, fs, window_ms=200, overlap=0.5):
    """
    Break each epoch into overlapping windows and extract features
    
    Returns:
        X (np.ndarray): list of features (lists)
        y (np.ndarray): label array for each window
    """
    win_len = int(fs * window_ms / 1000) # num samples in window
    step = int(win_len * (1 - overlap)) # num samples to move window

    X, y = [], []
    for epoch, label in zip(epochs, labels):
        for start in range(0, len(epoch) - win_len, step):
            window = epoch[start:start+win_len]
            feats = extract_features(window)
            X.append(feats)
            y.append(label)
    return np.array(X), np.array(y)

# FOR CONTINUOUS STREAM (RTOS SIM)
def segment_stream(signal, fs, window_ms=200, overlap=0.5):
    """
    Segment continuous EMG signal into overlapping windows and extract features.
    Useful for real-time testing where labels are unknown.
    
    Returns:
        X (np.ndarray): feature matrix for each window
    """
    win_len = int(fs * window_ms / 1000)
    step = int(win_len * (1 - overlap))

    X = []
    for start in range(0, len(signal) - win_len, step):
        window = signal[start:start+win_len]
        feats = extract_features(window)
        X.append(feats)
    return np.array(X)

# PREPROCESSING FUNCTIONS
