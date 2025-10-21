import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# === CONFIG ===
INPUT_FILE = "ecg_log.csv"   # your CSV file from Arduino
OUTPUT_FILE = "ecg_denoised.csv"
WINDOW_SIZE = 100  # number of samples per PCA window

# === LOAD DATA ===
df = pd.read_csv(INPUT_FILE)
signal = df['ecg_value'].astype(float).to_numpy()

# === PREPROCESSING ===
# Remove mean for each window
signal_mean = np.mean(signal)
signal_centered = signal - signal_mean

# Split signal into overlapping windows
def create_windows(data, window_size):
    n_windows = len(data) - window_size + 1
    return np.array([data[i:i + window_size] for i in range(n_windows)])

windows = create_windows(signal_centered, WINDOW_SIZE)

# === PCA DENOISING ===
pca = PCA()
pca.fit(windows)
components = pca.components_

# Keep only top N principal components (adjust for noise level)
n_components_keep = 3
pca_reduced = PCA(n_components=n_components_keep)
windows_reconstructed = pca_reduced.fit_transform(windows)
windows_denoised = pca_reduced.inverse_transform(windows_reconstructed)

# === RECONSTRUCT FULL SIGNAL ===
denoised_signal = np.zeros_like(signal_centered)
counts = np.zeros_like(signal_centered)

for i in range(len(windows_denoised)):
    denoised_signal[i:i + WINDOW_SIZE] += windows_denoised[i]
    counts[i:i + WINDOW_SIZE] += 1

denoised_signal /= counts
denoised_signal += signal_mean  # add back mean

# === SAVE RESULTS ===
df['ecg_denoised'] = denoised_signal
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Denoised ECG saved to {OUTPUT_FILE}")

# === PLOT ===
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp(ms)'], signal, label='Original ECG', alpha=0.5)
plt.plot(df['timestamp(ms)'], df['ecg_denoised'], label='Denoised ECG', linewidth=1.5)
plt.xlabel("Time (ms)")
plt.ylabel("ECG Signal")
plt.title("ECG Denoising using PCA")
plt.legend()
plt.show()
