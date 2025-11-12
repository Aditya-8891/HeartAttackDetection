"""
ecg_lstm_st_classifier.py

Pipeline:
 - Load CSV (timestamp(ms), ecg_value, ecg_denoised)
 - Bandpass filter -> R-peak detection (scipy find_peaks)
 - Extract per-beat windows centered on R (fixed length)
 - Compute an ST-level feature for each beat and create binary labels:
       label = 1 if mean(ST_window) - baseline > ST_ELEV_THRESHOLD else 0
   (This synthetic labeling lets the model learn to detect ST-elevation patterns.)
 - Train a Keras LSTM classifier on sequences (per-beat windows)
 - Evaluate on train/val split and show some predictions

Disclaimer: Synthetic labels are for demonstration. Use clinically-annotated datasets
(e.g., MIT-BIH, PTB-XL) for production / research. See references in comments.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import os

# -------------------------
# USER CONFIG
# -------------------------
INPUT_CSV = "ecg_denoised.csv"  # your CSV file
FS = 250               # sampling frequency (Hz) — matches example CSV (4 ms step)
BEAT_WINDOW_MS = 600   # window length centered on R-peak (ms)
ST_WINDOW_AFTER_S_MS = (40, 140)  # ST window relative to S-wave in ms (start,end)
ST_ELEV_THRESHOLD = 0.20  # synthetic threshold for ST elevation (units same as ecg_value)
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 30
BATCH_SIZE = 32
MODEL_SAVE = "lstm_st_classifier.h5"
# -------------------------

# helper functions
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(x, lowcut=0.5, highcut=40.0, fs=FS, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, x)

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # find time column and ecg column
    time_col = [c for c in df.columns if "time" in c.lower()][0]
    if 'ecg_denoised' in df.columns:
        sig_col = 'ecg_denoised'
    elif 'ecg_value' in df.columns:
        sig_col = 'ecg_value'
    else:
        raise ValueError("CSV must contain 'ecg_denoised' or 'ecg_value' and a time column.")
    t = df[time_col].to_numpy()
    sig = df[sig_col].astype(float).to_numpy()
    return t, sig, df

# R-peaks detection (simple amplitude-based with refractory)
def detect_r_peaks(signal, fs=FS):
    # filter first
    filtered = bandpass_filter(signal, fs=fs)
    # adaptive height threshold = mean + 0.5*std
    height_thr = np.mean(filtered) + 0.5 * np.std(filtered)
    min_distance = int(0.25 * fs)  # 250ms refractory minimum
    peaks, props = find_peaks(filtered, height=height_thr, distance=min_distance)
    return peaks, filtered

# create per-beat windows centered on R peaks
def extract_beat_windows(signal, peaks, fs=FS, window_ms=BEAT_WINDOW_MS):
    half = int((window_ms/1000.0) * fs / 2)
    windows = []
    valid_peaks = []
    for p in peaks:
        start = p - half
        end = p + half
        if start >= 0 and end <= len(signal):
            windows.append(signal[start:end])
            valid_peaks.append(p)
    return np.array(windows), np.array(valid_peaks)

# estimate S index (simplified: local min shortly after R)
def estimate_s_index_for_peak(filtered_signal, r_idx, fs=FS):
    # search 10-80ms after R for local minimum
    start = r_idx + int(0.01 * fs)
    end = r_idx + int(0.08 * fs)
    if end >= len(filtered_signal):
        return None
    local_min_idx = start + np.argmin(filtered_signal[start:end])
    return local_min_idx

# compute synthetic ST elevation measure per beat
def compute_st_elevation(filtered_signal, r_indices, fs=FS, st_window_after_s_ms=ST_WINDOW_AFTER_S_MS):
    st_measures = []
    s_list = []
    for r in r_indices:
        s_idx = estimate_s_index_for_peak(filtered_signal, r, fs)
        if s_idx is None:
            st_measures.append(np.nan)
            s_list.append(None)
            continue
        s_list.append(s_idx)
        st_start = s_idx + int(st_window_after_s_ms[0] * fs / 1000.0)
        st_end   = s_idx + int(st_window_after_s_ms[1] * fs / 1000.0)
        if st_end >= len(filtered_signal):
            st_measures.append(np.nan)
            continue
        # compute baseline around PR segment: naive baseline = median around 120-20ms before R
        baseline_start = max(0, r - int(0.12 * fs))
        baseline_end = max(0, r - int(0.02 * fs))
        baseline = np.median(filtered_signal[baseline_start:baseline_end]) if baseline_end>baseline_start else np.median(filtered_signal[max(0,r-10):r])
        st_mean = np.mean(filtered_signal[st_start:st_end])
        st_measures.append(st_mean - baseline)
    return np.array(st_measures), s_list

# main flow
if __name__ == "__main__":
    print("Loading:", INPUT_CSV)
    t, sig, df = load_data(INPUT_CSV)
    peaks, filtered = detect_r_peaks(sig)
    print(f"Detected {len(peaks)} R-peaks (before trimming).")
    windows, valid_peaks = extract_beat_windows(filtered, peaks)
    print(f"Extracted {len(windows)} beat windows (window {BEAT_WINDOW_MS} ms).")
    st_measures, s_indices = compute_st_elevation(filtered, valid_peaks)

    # create labels by thresholding ST measure (synthetic)
    labels = (st_measures > ST_ELEV_THRESHOLD).astype(int)
    # remove beats with NaN (edge beats)
    valid_mask = ~np.isnan(st_measures)
    X = windows[valid_mask]
    y = labels[valid_mask]
    peaks_used = valid_peaks[valid_mask]
    st_measures = st_measures[valid_mask]

    print("Class balance:", np.bincount(y))
    if len(X) == 0:
        raise ValueError("No valid beats found. Check your signal / parameters.")

    # normalize per-window (z-score)
    X_mean = X.mean(axis=1, keepdims=True)
    X_std  = X.std(axis=1, keepdims=True) + 1e-8
    X_norm = (X - X_mean) / X_std

    # reshape for LSTM: (samples, timesteps, features)
    X_lstm = X_norm.reshape((X_norm.shape[0], X_norm.shape[1], 1))

    # train/test split
    X_train, X_val, y_train, y_val = train_test_split(X_lstm, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y if len(np.unique(y))>1 else None)
    print("Train shape:", X_train.shape, "Val shape:", X_val.shape)

    # build LSTM model
    tf.keras.backend.clear_session()
    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(0.2),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    ]

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb, verbose=2)

    # evaluate
    y_pred_proba = model.predict(X_val).ravel()
    y_pred = (y_pred_proba >= 0.5).astype(int)
    print("\nClassification report (validation):")
    print(classification_report(y_val, y_pred, digits=4))
    try:
        auc = roc_auc_score(y_val, y_pred_proba)
        print(f"ROC AUC: {auc:.4f}")
    except Exception:
        pass
    print("Confusion matrix:\n", confusion_matrix(y_val, y_pred))

    # save model
    model.save(MODEL_SAVE)
    print("Saved model to", MODEL_SAVE)

    # show some example beats with predictions
    n_show = min(8, X_val.shape[0])
    fig, axes = plt.subplots(n_show, 1, figsize=(8, 2*n_show), sharex=False)
    idxs = np.random.choice(np.arange(X_val.shape[0]), size=n_show, replace=False)
    for i, ax in enumerate(axes):
        xi = X_val[idxs[i]].squeeze()
        ax.plot(np.linspace(-BEAT_WINDOW_MS/2, BEAT_WINDOW_MS/2, xi.shape[0]), xi, label='beat (norm)')
        ax.set_title(f"True={y_val[idxs[i]]}  Pred={y_pred[idxs[i]]}  p={y_pred_proba[idxs[i]]:.2f}  ST={st_measures[valid_mask][idxs[i]]:.3f}")
        ax.axvline(0, color='k', linestyle='--', alpha=0.6)  # R center
        ax.legend()
    plt.tight_layout()
    plt.show()

    # quick metric: show beats predicted positive with ST measure
    pos_idxs = np.where(y_pred==1)[0]
    print(f"Validation predicted positives: {len(pos_idxs)}")
    if len(pos_idxs)>0:
        for i in pos_idxs[:10]:
            print(f"idx {i} proba {y_pred_proba[i]:.3f} st_measure {st_measures[valid_mask][i]:.3f}")
