import numpy as np

def rms(signal):
    # compute root mean square of signal array
    return np.sqrt(np.mean(signal**2))

def wl(signal):
    # compute waveform length of signal array
    return np.sum(np.abs(np.diff(signal)))

def zc(signal, threshold=0.01):
    # compute zero crossings (how many times signal changes sign)
    count = np.sum(((signal[:-1] * signal[1:]) < 0) &
                   (np.abs(signal[:-1] - signal[1:]) >= threshold))
    return count

def ssc(signal, threshold=0.01):
    # compute slope sign changes (how many times the gradient changes sign)
    count = np.sum(((signal[1:-1] - signal[:-2]) * (signal[1:-1] - signal[2:])) > 0 &
                   ((np.abs(signal[1:-1] - signal[:-2]) >= threshold) |
                    (np.abs(signal[1:-1] - signal[2:]) >= threshold)))
    return count

def wamp(signal, threshold=0.01):
    # compute willison amplitute (counts significant signal jumps)
    return np.sum(np.abs(np.diff(signal)) >= threshold)

def preprocess(epoch):
    # subtract mean to account for DC offset (drift)
    return epoch - np.mean(epoch)

def extract_features(epochs):
    # given list of epochs
    # returns array of features [[RMS, WL, ZC, SSC, WAMP], ...]
    feats = []
    for epoch in epochs:
        epoch = preprocess(epoch)
        feats.append([
            rms(epoch),
            wl(epoch),
            zc(epoch),
            ssc(epoch),
            wamp(epoch)
        ])
    return np.array(feats)
