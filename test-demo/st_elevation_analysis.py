import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# === CONFIG ===
INPUT_FILE = "ecg_denoised.csv"
ST_THRESHOLD = 0.1       # 0.1 mV elevation = ST elevation
SAMPLING_INTERVAL = 4e-3  # 4 ms per sample → 250 Hz (based on your data)
ST_OFFSET_MS = 80         # ST segment ~80 ms after R peak

# === LOAD DATA ===
df = pd.read_csv(INPUT_FILE)
time = df.iloc[:, 0].values
ecg = df.iloc[:, 2].values  # use denoised

# === FIND R PEAKS ===
peaks, _ = find_peaks(ecg, distance=50, height=np.mean(ecg)+0.5*np.std(ecg))
print(f"Detected {len(peaks)} R-peaks.")

# === ST SEGMENT ANALYSIS ===
st_events = []
baseline_vals = []

for i, r in enumerate(peaks):
    # Approximate PR baseline: average of 40–60 ms before R
    pr_start = max(0, r - int(60e-3 / SAMPLING_INTERVAL))
    pr_end = max(0, r - int(20e-3 / SAMPLING_INTERVAL))
    baseline = np.mean(ecg[pr_start:pr_end]) if pr_end > pr_start else 0
    baseline_vals.append(baseline)
    
    # ST segment level: 60–80 ms after R
    st_index = r + int(ST_OFFSET_MS / 500 / SAMPLING_INTERVAL)
    if st_index >= len(ecg): 
        continue
    st_level = ecg[st_index]
    
    st_diff = st_level - baseline
    if st_diff > ST_THRESHOLD:
        st_events.append((time[r], st_diff))
        print(f"⚠️  ST elevation at {time[r]} ms: +{st_diff:.3f} mV")

print(f"\nSummary: {len(st_events)} ST elevation events detected.")

# === VISUALIZATION ===
plt.figure(figsize=(14, 6))
plt.plot(time, ecg, label='ECG (denoised)', linewidth=1.2)

# Mark R peaks
plt.scatter(time[peaks], ecg[peaks], color='r', marker='o', label='R Peaks')

# Highlight ST elevation points
if st_events:
    st_times, st_diffs = zip(*st_events)
    st_indices = [np.argmin(np.abs(time - t)) for t in st_times]
    plt.scatter(np.array(st_times), ecg[st_indices], color='orange', s=60, label='ST Elevation')

plt.title("ECG with R Peaks and ST Elevations")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (mV)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
