import numpy as np
import os
import scipy.io.wavfile as wav
from processing import load_epochs, segment_windows, bandpass_filter, notch_filter
from classify import train_lda, cross_validate, test_stream
from plot import plot_predictions, plot_raw_signal

# SETTINGS
TRAIN_PATH = "../data/newdata.wav"   # 5 s epochs (labelled)
TEST_PATH = "../data/footSweeps.wav" # continuous sweeps (to test trained clf)
EPOCH_TIME = 5
WINDOW_MS = 200 # CHECK LITERATURE
OVERLAP = 0.5 # ^
CHANNEL = 0 # TA only

# TRAINING PHASE
print("\n--- TRAINING ---")
fs, epochs, labels = load_epochs(TRAIN_PATH, EPOCH_TIME, n_classes=4, channel=CHANNEL)
X, y = segment_windows(epochs, labels, fs, WINDOW_MS, OVERLAP)

clf, acc = train_lda(X, y)
print(f"Classifier accuracy: {acc:.2f}%")

cv_acc, scores = cross_validate(clf, X, y, k=5)
print(f"Cross-validation accuracy (5-fold): {cv_acc:.2f}%")

# TESTING PHASE
print("\n--- TESTING ---")
fs, data = wav.read(TEST_PATH)
signal = data[:,CHANNEL] # channel = 0 -> TA only

# plot raw
plot_raw_signal(signal, fs, title="Raw EMG", save_path="../plots/raw_footSweepsTA.png")

signal = bandpass_filter(signal, fs, 15, 450)

# plot raw (preprocessed)
plot_raw_signal(signal, fs, title="Raw EMG (preprocessed)", save_path="../plots/rawpre_footSweepsTA.png")

preds, probs = test_stream(clf, signal, fs, WINDOW_MS, OVERLAP)
# print("\nExample probability outputs:")
# for i, (p, pr) in enumerate(zip(preds[:20], probs[:20])): # first 20 windows
#     probs_str = ", ".join([f"{x*100:.2f}%" for x in pr])
#     print(f"Window {i}: Pred = {p}, Probs = [{probs_str}]")
    
# export probs for FLC
win_len = int(fs * WINDOW_MS / 1000)
step = int(win_len * (1 - OVERLAP))
times = np.arange(len(preds)) * (step / fs)

out = np.column_stack((times, preds, probs))
header = "Time_s,Predicted_Class,Prob_Class0,Prob_Class1,Prob_Class2,Prob_Class3"

np.savetxt(
    "../data/preds_probs.csv",
    out,
    delimiter=",",
    fmt=["%.3f", "%d", "%.4f", "%.4f", "%.4f", "%.4f"],  # 6 formats
    header=header,
    comments=""
)

# plot results
plot_predictions(preds, title="Predicted ankle states (continuous movement)", save_path="../plots/predictions.png")
