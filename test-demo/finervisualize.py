import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

# === CONFIG ===
INPUT_FILE = "ecg_denoised.csv"   # CSV with columns: timestamp(ms), ecg_value, ecg_denoised
USE_DENOISED = True               # use denoised signal if available
FS = 250                          # sampling rate in Hz (adjust to your setup!)
ST_WINDOW = (0.08, 0.12)          # window (in seconds) after S-wave for ST segment

# === LOAD DATA ===
df = pd.read_csv(INPUT_FILE)
t = df[[c for c in df.columns if "time" in c.lower()][0]].to_numpy()
ecg = df['ecg_denoised'].to_numpy() if USE_DENOISED and 'ecg_denoised' in df.columns else df['ecg_value'].to_numpy()

# === PREPROCESSING ===
# Bandpass filter (remove baseline drift and high freq noise)
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(signal, lowcut=0.5, highcut=40.0):
    b, a = butter_bandpass(lowcut, highcut, FS)
    return filtfilt(b, a, signal)

filtered_ecg = bandpass_filter(ecg)

# === R-PEAK DETECTION ===
# Use scipy’s peak finder, with adaptive height and distance
distance = int(0.25 * FS)  # 250 ms between beats (≈ 240 bpm max)
peaks, _ = find_peaks(filtered_ecg, distance=distance, height=np.mean(filtered_ecg) + 0.5*np.std(filtered_ecg))

# === ESTIMATE S-WAVE AND ST SEGMENT LOCATIONS ===
# S-wave: typically just after the R-peak local minimum
S_indices = []
ST_indices = []

for r in peaks:
    # Search right after R-peak (~40 ms)
    window_start = r + int(0.02 * FS)
    window_end = r + int(0.08 * FS)
    if window_end < len(filtered_ecg):
        # S-wave = minimum in that window
        s_idx = window_start + np.argmin(filtered_ecg[window_start:window_end])
        S_indices.append(s_idx)
        # ST segment starts after S + ST_WINDOW[0]s and ends at S + ST_WINDOW[1]s
        st_start = s_idx + int(ST_WINDOW[0] * FS)
        st_end = s_idx + int(ST_WINDOW[1] * FS)
        if st_end < len(filtered_ecg):
            ST_indices.append((st_start, st_end))

# === PLOT RESULTS ===
plt.figure(figsize=(14, 7))
plt.plot(t, filtered_ecg, label='Filtered ECG', linewidth=1.2, color='black')
plt.scatter(t[peaks], filtered_ecg[peaks], color='red', label='R-peaks', zorder=5)
plt.scatter(t[S_indices], filtered_ecg[S_indices], color='blue', label='S-waves', zorder=5)

# Highlight ST segments
for (st_start, st_end) in ST_indices:
    plt.axvspan(t[st_start], t[st_end], color='yellow', alpha=0.3)

plt.title("ECG Signal with R-Peak & ST Segment Detection")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# === OPTIONAL: HEART RATE CALCULATION ===
if len(peaks) > 1:
    rr_intervals = np.diff(t[peaks]) / 1000.0  # convert ms to seconds
    bpm = 60.0 / np.mean(rr_intervals)
    print(f"💓 Estimated Heart Rate: {bpm:.1f} bpm")
