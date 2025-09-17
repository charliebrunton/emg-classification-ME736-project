import numpy as np

# FEATURE FUNCTIONS
def rms(signal):
    """Root Mean Square"""
    return np.sqrt(np.mean(signal**2))

def wl(signal):
    """Waveform Length"""
    return np.sum(np.abs(np.diff(signal)))

def zc(signal, threshold=0.01):
    """Zero Crossings (counts sign changes above threshold)"""
    return np.sum(((signal[:-1] * signal[1:]) < 0) &
                  (np.abs(signal[:-1] - signal[1:]) >= threshold))

def ssc(signal, threshold=0.01):
    """Slope Sign Changes (gradient reversals above threshold)"""
    dif1 = signal[1:-1] - signal[:-2]
    dif2 = signal[1:-1] - signal[2:]
    return np.sum(((dif1 * dif2) > 0) &
                  ((np.abs(dif1) >= threshold) | (np.abs(dif2) >= threshold)))

def wamp(signal, threshold=0.01):
    """Willison Amplitude (counts jumps above threshold)"""
    return np.sum(np.abs(np.diff(signal)) >= threshold)

def remove_offset(epoch):
    """Subtract mean to remove DC offset/drift"""
    return epoch - np.mean(epoch)

# FEATURE EXTRACTION WRAPPER
def extract_features(epochs, threshold=0.01):
    """
    Given list/array of epochs, return features:
    [RMS, WL, ZC, SSC, WAMP]
    """
    feats = []
    for epoch in epochs:
        epoch = remove_offset(epoch)
        feats.append([
            rms(epoch),
            wl(epoch),
            zc(epoch, threshold),
            ssc(epoch, threshold),
            wamp(epoch, threshold)
        ])
    return np.array(feats)
