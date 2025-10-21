import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# === CONFIG ===
INPUT_FILE = "ecg_denoised.csv"  # should contain columns: timestamp(ms), ecg_value, ecg_denoised
REALTIME_MODE = False            # True = animate playback
SAVE_FIGURE = False              # Save static figure as PNG

# === LOAD DATA ===
df = pd.read_csv(INPUT_FILE)

# Support both raw and denoised columns
time_col = [c for c in df.columns if "time" in c.lower()][0]
signal_col = [c for c in df.columns if "ecg_value" in c.lower()][0]
denoised_col = None
for c in df.columns:
    if "denoise" in c.lower():
        denoised_col = c

t = df[time_col].to_numpy()
raw = df[signal_col].astype(float).to_numpy()
denoised = df[denoised_col].astype(float).to_numpy() if denoised_col else None

# === STATIC PLOT ===
if not REALTIME_MODE:
    plt.figure(figsize=(12, 6))
    plt.plot(t, raw, label='Raw ECG', alpha=0.6)
    if denoised_col:
        plt.plot(t, denoised, label='Denoised ECG', linewidth=1.2)
    plt.title("ECG Signal Visualization")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if SAVE_FIGURE:
        plt.savefig("ecg_plot.png", dpi=300)
        print("✅ Saved ECG plot as ecg_plot.png")
    plt.show()

# === REALTIME / ANIMATED MODE ===
else:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(np.min(raw) - 100, np.max(raw) + 100)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Real-time ECG Playback")

    raw_line, = ax.plot([], [], lw=1, label="Raw ECG", color='tab:red', alpha=0.6)
    den_line = None
    if denoised_col:
        den_line, = ax.plot([], [], lw=1.2, label="Denoised ECG", color='tab:blue')

    ax.legend()

    def init():
        raw_line.set_data([], [])
        if den_line:
            den_line.set_data([], [])
        return (raw_line,) if not den_line else (raw_line, den_line)

    def update(frame):
        idx = frame * 10  # adjust playback speed
        xdata = t[:idx]
        ydata = raw[:idx]
        raw_line.set_data(xdata, ydata)
        if den_line:
            den_line.set_data(xdata, denoised[:idx])
        return (raw_line,) if not den_line else (raw_line, den_line)

    ani = animation.FuncAnimation(fig, update, frames=len(t)//10,
                                  init_func=init, blit=True, interval=30, repeat=False)

    plt.show()
