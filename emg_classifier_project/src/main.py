import scipy.io.wavfile as wav
from processing import load_epochs, segment_windows
from classify import train_lda, cross_validate, test_stream
from plot import plot_predictions

# SETTINGS
TRAIN_PATH = "../data/newdata.wav"   # 5 s epochs (labelled)
TEST_PATH = "../data/footSweeps.wav" # continuous sweeps
EPOCH_TIME = 5
WINDOW_MS = 200
OVERLAP = 0.5

# -----------------------------
# TRAINING PHASE
# -----------------------------
print("\n--- TRAINING ---")
fs, epochs, labels = load_epochs(TRAIN_PATH, EPOCH_TIME, n_classes=4, channel=0)
X, y = segment_windows(epochs, labels, fs, WINDOW_MS, OVERLAP)

clf, acc = train_lda(X, y)
print(f"Classifier accuracy: {acc:.2f}%")

# cross-validation
cv_acc, scores = cross_validate(clf, X, y, method="kfold", k=5)
print(f"Cross-validation accuracy (5-fold): {cv_acc:.2f}%")

# -----------------------------
# TEST PHASE (continuous)
# -----------------------------
print("\n--- TESTING ---")
fs, data = wav.read(TEST_PATH)
signal = data[:,0] if data.ndim > 1 else data  # TA only

# predictions + probabilities
preds, probs = test_stream(clf, signal, fs, WINDOW_MS, OVERLAP, return_probs=True)
print("\nExample probability outputs:")
for i, (p, pr) in enumerate(zip(preds[:20], probs[:20])):   # first 20 windows
    probs_str = ", ".join([f"{x*100:.2f}%" for x in pr])
    print(f"Window {i}: Pred = {p}, Probs = [{probs_str}]")

# plot results
plot_predictions(preds, title="Predicted ankle states (continuous movement)")
