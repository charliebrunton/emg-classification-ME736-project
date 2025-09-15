import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, LeavePOut
from scripts.features import extract_features
from scripts.classifier import train, evaluate, predict_probs, train_logreg
from scripts.plot import plot_raw_signals, plot_preprocessed_signals, plot_feature_space, plot_confusion_matrices

# definitions
DATA_PATH = "data/run_4.wav"
EPOCH_TIME = 10

# import and segment data
fs, data = wav.read(DATA_PATH)
epoch_length = EPOCH_TIME * fs
data = data[:epoch_length*(len(data)//epoch_length)] # trim extra samples
ta_data = data[:, 0] # tibialis anterior
gas_data = data[:, 1] # gastrocnemius (medial)
ta_epochs = [ta_data[i:i+epoch_length] for i in range(0, len(ta_data), epoch_length)]
gas_epochs = [gas_data[i:i+epoch_length] for i in range(0, len(gas_data), epoch_length)]
labels = np.arange(len(ta_epochs)) % 4 # same for ta and gas

# extract features
ta_features = extract_features(ta_epochs)
gas_features = extract_features(gas_epochs)

# split into train/test sets using same seed
ta_train, ta_test, ta_train_labs, ta_test_labs = train_test_split(ta_features,
                                                            labels,
                                                            test_size=0.3,
                                                            random_state=33)
gas_train, gas_test, gas_train_labs, gas_test_labs = train_test_split(gas_features,
                                                            labels,
                                                            test_size=0.3,
                                                            random_state=33)

# train classifiers
ta_clf = train(ta_train, ta_train_labs)
gas_clf = train(gas_train, gas_train_labs)
# ta_clf = train_logreg(ta_train, ta_train_labs)
# gas_clf = train_logreg(gas_train, gas_train_labs)

# test
ta_acc, ta_preds = evaluate(ta_clf, ta_test, ta_test_labs)
gas_acc, gas_preds = evaluate(gas_clf, gas_test, gas_test_labs)
print(f"\nTA test accuracy: {ta_acc*100:.2f}%")
print(f"GAS test accuracy: {gas_acc*100:.2f}%")

# get probabilities (for fuzzy logic)
ta_probs = predict_probs(ta_clf, ta_test)
gas_probs = predict_probs(gas_clf, gas_test)

# cross-validate (leave-two-out)
lpo = LeavePOut(p=2) 
ta_scores = cross_val_score(ta_clf, ta_features, labels, cv=lpo)
gas_scores = cross_val_score(gas_clf, gas_features, labels, cv=lpo)
print(f"\nTA mean accuracy (leave-two-out cross-validation): {ta_scores.mean()*100:.2f}%")
print(f"GAS mean accuracy (leave-two-out cross-validation): {gas_scores.mean()*100:.2f}%")

# simulate prosthesis state updates (redundant measurements update on AND condition)
agreement_mask = ta_preds == gas_preds
agreement_rate = np.mean(agreement_mask) * 100
agreed_indices = np.where(agreement_mask)[0]
if len(agreed_indices) > 0:
    correct_agreement_rate = np.mean(ta_preds[agreed_indices] == ta_test_labs[agreed_indices]) * 100
else:
    correct_agreement_rate = 0.0
print(f"\nAgreement rate between TA and GAS predictions: {agreement_rate:.2f}%")
print(f"Correct agreement accuracy (within agreement set): {correct_agreement_rate:.2f}%")

# generate plots
plot_raw_signals(ta_data, gas_data, fs, save_path="plots/raw_signals.png")
plot_preprocessed_signals(ta_data, gas_data, fs, save_path="plots/preprocessed_signals.png")
plot_feature_space(ta_epochs, labels, title="TA Feature Space (RMS vs WL)", save_path="plots/ta_feature_space.png")
plot_feature_space(gas_epochs, labels, title="GAS Feature Space (RMS vs WL)", save_path="plots/gas_feature_space.png")

# print some class probabilites
print("\nTA probabilities:")
for i, (pred, prob_vec) in enumerate(zip(ta_preds, ta_probs)):
    probs_str = ", ".join([f"{p*100:.2f}%" for p in prob_vec])
    print(f"Sample {i}: Pred = {pred}, Probs = [{probs_str}]")
print("\nGAS probabilities:")
for i, (pred, prob_vec) in enumerate(zip(gas_preds, gas_probs)):
    probs_str = ", ".join([f"{p*100:.2f}%" for p in prob_vec])
    print(f"Sample {i}: Pred = {pred}, Probs = [{probs_str}]")
    
# confusion matrices
plot_confusion_matrices(
    ta_test_labs, ta_preds,
    gas_test_labs, gas_preds,
    save_path="plots/confusion_matrices.png"
)
