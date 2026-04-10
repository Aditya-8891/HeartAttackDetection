"""
live_demo.py

Live ECG streaming demo with real-time arrhythmia detection + ST elevation analysis.

Streams a MIT-BIH record sample-by-sample (simulating a wearable device),
runs CNN ResNet / Attention LSTM / BiGRU on every beat, measures ST elevation
per beat as a rule-based clinical indicator, and shows a 5-panel dark-theme
matplotlib visualization.

Two separate detection layers:
  - ML models (panels 2 & 4): arrhythmia type classification trained on MIT-BIH
  - ST elevation (panel 3): rule-based measurement — the actual STEMI marker

Reuses ECGDataBuffer from test-demo/live_ecg_analysis.py.

Usage:
    python demo/live_demo.py --record data/mitdb_raw/119 --speed 5.0 --loop
    python demo/live_demo.py --record data/mitdb_raw/108 --speed 3.0
    python demo/live_demo.py --no-plot
    python demo/live_demo.py --help

Panels:
    1. Scrolling ECG waveform (last 5 seconds) with R-peak markers
    2. Per-beat arrhythmia probability — CNN (blue), LSTM (orange), GRU (green)
    3. ST elevation per beat with clinical threshold lines (0.1mV / 0.2mV)
    4. Real-time heart rate trend
    5. Ensemble decision gauge with NORMAL / ANOMALY alert banner
"""

import os
import sys
import time
import threading
import argparse
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt, find_peaks

# Import ECGDataBuffer from existing code — avoid duplication
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_project_root, 'test-demo'))
from live_ecg_analysis import ECGDataBuffer  # noqa: E402

# ==================== CONFIGURATION ====================
FS = 360                        # MIT-BIH sampling rate
SAMPLE_INTERVAL = 1.0 / FS     # ~2.78ms per sample
BEAT_WINDOW_SAMPLES = 216       # 600ms at 360Hz
HALF = BEAT_WINDOW_SAMPLES // 2
DISPLAY_SECONDS = 5             # ECG waveform window shown in panel 1
DISPLAY_SAMPLES = DISPLAY_SECONDS * FS   # 1800 samples
ENSEMBLE_THRESHOLD = 0.6        # Alert if ensemble prob >= this
MODEL_THRESHOLD = 0.5           # Per-model binary decision threshold
MIN_PEAK_DISTANCE = int(0.25 * FS)   # 250ms refractory period (min 40 BPM)

# ST elevation thresholds (mV)
ST_CLINICAL_THRESHOLD = 0.1    # Clinically notable
ST_CRITICAL_THRESHOLD = 0.2    # Possible STEMI

# ST measurement windows (seconds, relative to S-wave)
ST_MEASURE_START = 0.040       # 40ms after S-wave (J-point + 40ms)
ST_MEASURE_END   = 0.140       # 140ms after S-wave
# =======================================================

# Default paths to demo models (trained by demo/models/train_all.py)
_SAVE_DIR = os.path.join(_script_dir, 'models', 'saved')
DEFAULT_CNN  = os.path.join(_SAVE_DIR, 'cnn_resnet.h5')
DEFAULT_LSTM = os.path.join(_SAVE_DIR, 'lstm_attention.h5')
DEFAULT_GRU  = os.path.join(_SAVE_DIR, 'gru_model.h5')

MODEL_COLORS = {'cnn': '#2196F3', 'lstm': '#FF9800', 'gru': '#4CAF50'}
MODEL_LABELS = {'cnn': 'CNN ResNet', 'lstm': 'LSTM+Attn', 'gru': 'BiGRU'}


# ======================================================================
# Analyzer

class MultiModelAnalyzer:
    """Real-time ECG analyzer: CNN / LSTM / GRU predictions + ST elevation."""

    def __init__(self):
        self.buffer = ECGDataBuffer(
            window_size_seconds=DISPLAY_SECONDS + 5, sampling_rate=FS
        )

        # Per-model prediction histories
        self.prediction_histories = {
            'cnn':  deque(maxlen=100),
            'lstm': deque(maxlen=100),
            'gru':  deque(maxlen=100),
        }
        self.ensemble_history = deque(maxlen=100)

        # ST elevation history: {'time', 'elevation'}
        self.st_history = deque(maxlen=100)

        # Heart rate tracking
        self.detected_peaks = []
        self.last_peak_time = 0.0
        self.heart_rate_history = deque(maxlen=30)

        # Alert state
        self.alert_active = False
        self.latest_ensemble = 0.0
        self.latest_st = np.nan
        self.total_beats = 0
        self.abnormal_beats = 0
        self.st_alert_count = 0

        self.models = {}

    # ------------------------------------------------------------------
    def load_models(self, cnn_path, lstm_path, gru_path):
        """Load trained models; demo still runs with any subset loaded."""
        import tensorflow as tf

        paths = {'cnn': cnn_path, 'lstm': lstm_path, 'gru': gru_path}
        loaded = []
        for name, path in paths.items():
            abs_path = path if os.path.isabs(path) else os.path.join(_project_root, path)
            if os.path.exists(abs_path):
                try:
                    self.models[name] = tf.keras.models.load_model(
                        abs_path, compile=False
                    )
                    print(f"  [OK]   {MODEL_LABELS[name]}: {abs_path}")
                    loaded.append(name)
                except Exception as e:
                    print(f"  [FAIL] {MODEL_LABELS[name]}: {e}")
            else:
                print(f"  [SKIP] {MODEL_LABELS[name]} — not found: {abs_path}")

        if not loaded:
            print("\n  WARNING: No models loaded. ST elevation only mode.")
        else:
            print(f"\n  Loaded {len(loaded)}/{len(paths)} models: {loaded}")
        return loaded

    # ------------------------------------------------------------------
    # Signal processing helpers

    def _bandpass_filter(self, signal):
        """0.5–40 Hz 3rd-order Butterworth at 360 Hz."""
        nyq = 0.5 * FS
        b, a = butter(3, [0.5 / nyq, 40.0 / nyq], btype='band')
        return filtfilt(b, a, signal)

    def _normalize_beat(self, window):
        """Per-beat z-score normalization (same as preprocessing)."""
        mean = np.mean(window)
        std = np.std(window) + 1e-8
        return (window - mean) / std

    # ------------------------------------------------------------------
    # ST elevation measurement

    def _estimate_s_index(self, filtered_signal, r_idx):
        """Find S-wave as the local minimum in the 10–80ms window after R-peak."""
        start = r_idx + int(0.010 * FS)   # 10ms = ~4 samples
        end   = r_idx + int(0.080 * FS)   # 80ms = ~29 samples
        if end >= len(filtered_signal):
            return None
        return start + int(np.argmin(filtered_signal[start:end]))

    def _compute_st_elevation(self, filtered_signal, r_idx):
        """Measure ST elevation in mV relative to PR baseline.

        Method (same as existing test-demo code, parameterised for 360Hz):
          1. Find S-wave (local min after R)
          2. ST segment = 40–140ms after S-wave
          3. Baseline = median of PR segment (120–20ms before R)
          4. ST elevation = mean(ST) - baseline

        Returns np.nan if windows fall outside signal bounds.
        """
        s_idx = self._estimate_s_index(filtered_signal, r_idx)
        if s_idx is None:
            return np.nan

        st_start = s_idx + int(ST_MEASURE_START * FS)
        st_end   = s_idx + int(ST_MEASURE_END   * FS)

        if st_end >= len(filtered_signal):
            return np.nan

        baseline_start = max(0, r_idx - int(0.12 * FS))
        baseline_end   = max(0, r_idx - int(0.02 * FS))

        if baseline_end <= baseline_start:
            return np.nan

        baseline   = np.median(filtered_signal[baseline_start:baseline_end])
        st_mean    = np.mean(filtered_signal[st_start:st_end])
        return float(st_mean - baseline)

    # ------------------------------------------------------------------
    # Per-beat ML inference

    def _predict_all_models(self, beat_window):
        """Normalise once, run all loaded models.

        Returns dict with 'cnn', 'lstm', 'gru' (None if model absent)
        and 'ensemble' (mean of available models).
        """
        norm = self._normalize_beat(beat_window)
        X = norm.reshape(1, len(norm), 1).astype(np.float32)

        probs = {}
        for name, model in self.models.items():
            try:
                probs[name] = float(model.predict(X, verbose=0)[0][0])
            except Exception:
                probs[name] = 0.0

        for name in ['cnn', 'lstm', 'gru']:
            probs.setdefault(name, None)

        loaded_vals = [p for p in probs.values() if p is not None]
        probs['ensemble'] = float(np.mean(loaded_vals)) if loaded_vals else 0.0
        return probs

    # ------------------------------------------------------------------
    # Main per-frame analysis

    def analyze_new_peaks(self):
        """Detect new R-peaks and run both ML predictions and ST measurement."""
        signal, timestamps = self.buffer.get_array()
        if len(signal) < int(0.5 * FS):
            return

        filtered = self._bandpass_filter(signal)
        height_thr = np.mean(filtered) + 0.5 * np.std(filtered)
        peak_indices, _ = find_peaks(
            filtered, height=height_thr, distance=MIN_PEAK_DISTANCE
        )

        for peak_idx in peak_indices:
            peak_time = float(timestamps[peak_idx])
            if peak_time <= self.last_peak_time:
                continue

            # ---- ML predictions ----
            start = peak_idx - HALF
            end   = peak_idx + HALF
            if start >= 0 and end < len(filtered):
                beat_window = filtered[start:end]
                probs = self._predict_all_models(beat_window)
                ensemble = probs['ensemble']

                for name in ['cnn', 'lstm', 'gru']:
                    p = probs[name]
                    if p is not None:
                        self.prediction_histories[name].append({
                            'time': peak_time,
                            'probability': p,
                            'prediction': int(p >= MODEL_THRESHOLD),
                        })

                self.ensemble_history.append({'time': peak_time, 'probability': ensemble})
                self.latest_ensemble = ensemble
                self.alert_active = ensemble >= ENSEMBLE_THRESHOLD
                if ensemble >= MODEL_THRESHOLD:
                    self.abnormal_beats += 1

            else:
                ensemble = 0.0

            # ---- ST elevation measurement ----
            st_elev = self._compute_st_elevation(filtered, peak_idx)
            self.latest_st = st_elev
            if not np.isnan(st_elev):
                self.st_history.append({'time': peak_time, 'elevation': st_elev})
                if st_elev >= ST_CRITICAL_THRESHOLD:
                    self.st_alert_count += 1

            # ---- Heart rate ----
            if self.detected_peaks:
                last_time = self.detected_peaks[-1][1]
                rr_s = (peak_time - last_time) / 1000.0
                if 0.3 < rr_s < 2.0:
                    self.heart_rate_history.append(60.0 / rr_s)

            self.detected_peaks.append((peak_idx, peak_time))
            self.last_peak_time = peak_time
            self.total_beats += 1

    def current_hr(self):
        if not self.heart_rate_history:
            return 0.0
        return float(np.mean(list(self.heart_rate_history)[-5:]))


# ======================================================================
# Visualizer

class LiveDemoVisualizer:
    """5-panel dark-theme real-time matplotlib visualizer.

    Layout (GridSpec 4 rows × 2 cols):
      Row 0 full:  ECG waveform
      Row 1 full:  Per-model arrhythmia probabilities
      Row 2 left:  ST elevation
      Row 2 right: Heart rate trend
      Row 3 full:  Ensemble gauge + alert banner
    """

    _BG_DARK  = '#1a1a2e'
    _BG_PANEL = '#16213e'
    _FG       = '#aaaacc'
    _WHITE    = 'white'

    def __init__(self, analyzer):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        self.analyzer = analyzer
        self.plt = plt

        self.fig = plt.figure(figsize=(16, 14))
        self.fig.patch.set_facecolor(self._BG_DARK)

        gs = gridspec.GridSpec(4, 2, figure=self.fig,
                               hspace=0.55, wspace=0.35,
                               left=0.07, right=0.97, top=0.94, bottom=0.05)

        self.ax_ecg      = self.fig.add_subplot(gs[0, :])   # ECG waveform
        self.ax_pred     = self.fig.add_subplot(gs[1, :])   # Model probabilities
        self.ax_st       = self.fig.add_subplot(gs[2, 0])   # ST elevation
        self.ax_hr       = self.fig.add_subplot(gs[2, 1])   # Heart rate
        self.ax_ensemble = self.fig.add_subplot(gs[3, :])   # Ensemble gauge

        for ax in [self.ax_ecg, self.ax_pred, self.ax_st,
                   self.ax_hr, self.ax_ensemble]:
            self._style(ax)

        self.fig.suptitle(
            'Live ECG Analysis  ·  CNN ResNet  ·  LSTM+Attention  ·  BiGRU  ·  ST Elevation',
            fontsize=13, fontweight='bold', color=self._WHITE, y=0.98
        )

    def _style(self, ax):
        ax.set_facecolor(self._BG_PANEL)
        ax.tick_params(colors=self._FG, labelsize=8)
        ax.xaxis.label.set_color(self._FG)
        ax.yaxis.label.set_color(self._FG)
        ax.title.set_color(self._WHITE)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

    # ------------------------------------------------------------------
    def update(self, frame):
        a = self.analyzer
        signal, timestamps = a.buffer.get_array()

        # ---- Panel 1: Scrolling ECG ----
        ax = self.ax_ecg
        ax.clear()
        self._style(ax)
        if len(signal) > 10:
            disp_sig = signal[-DISPLAY_SAMPLES:]
            disp_ts  = timestamps[-DISPLAY_SAMPLES:]
            t_sec = (disp_ts - disp_ts[0]) / 1000.0

            ax.plot(t_sec, disp_sig, color='#00e5ff', linewidth=0.9, alpha=0.9)

            # R-peak markers
            if a.detected_peaks:
                win_start = float(disp_ts[0])
                win_end   = float(disp_ts[-1])
                visible = [(idx, t) for idx, t in a.detected_peaks
                           if win_start <= t <= win_end]
                if visible:
                    p_ts  = [(t - win_start) / 1000.0 for _, t in visible]
                    p_vals = []
                    for _, t in visible:
                        i = int(np.searchsorted(disp_ts, t))
                        p_vals.append(disp_sig[min(i, len(disp_sig)-1)])
                    ax.scatter(p_ts, p_vals, c='#ff5252', s=25, zorder=5)

        hr = a.current_hr()
        st_str = f'{a.latest_st:.3f} mV' if not np.isnan(a.latest_st) else '—'
        ax.set_title(
            f'ECG Signal  |  HR: {hr:.0f} BPM  |  Beats: {a.total_beats}  |  '
            f'Arrhythmia: {a.abnormal_beats} ({100*a.abnormal_beats/max(1,a.total_beats):.0f}%)  |  '
            f'Latest ST: {st_str}',
            fontsize=8.5, color=self._WHITE
        )
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Amplitude', fontsize=8)
        ax.set_xlim(0, DISPLAY_SECONDS)
        ax.grid(True, alpha=0.12, color='#ffffff')

        # ---- Panel 2: Arrhythmia probabilities ----
        ax = self.ax_pred
        ax.clear()
        self._style(ax)
        for name in ['cnn', 'lstm', 'gru']:
            hist = list(a.prediction_histories[name])
            if hist:
                ts   = [h['time'] / 1000.0 for h in hist]
                prob = [h['probability'] for h in hist]
                ax.plot(ts, prob, '-o', color=MODEL_COLORS[name],
                        linewidth=1.2, markersize=3.5, alpha=0.85,
                        label=MODEL_LABELS[name])
        ax.axhline(y=MODEL_THRESHOLD, color='#ff9800', linestyle='--',
                   linewidth=1.0, alpha=0.6, label='Threshold 0.5')
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Arrhythmia Probability', fontsize=8)
        ax.set_title('Per-Beat Arrhythmia Detection (ML models — trained on MIT-BIH)',
                     fontsize=8.5, color=self._WHITE)
        ax.legend(fontsize=7, loc='upper left',
                  facecolor=self._BG_DARK, labelcolor=self._WHITE, framealpha=0.7)
        ax.grid(True, alpha=0.12, color='#ffffff')

        # ---- Panel 3: ST elevation ----
        ax = self.ax_st
        ax.clear()
        self._style(ax)
        st_hist = list(a.st_history)
        if st_hist:
            ts   = [s['time'] / 1000.0 for s in st_hist]
            elevs = [s['elevation'] for s in st_hist]
            colors = ['#ff5252' if e >= ST_CRITICAL_THRESHOLD
                      else '#ff9800' if e >= ST_CLINICAL_THRESHOLD
                      else '#69f0ae' for e in elevs]
            ax.scatter(ts, elevs, c=colors, s=30, zorder=5)
            ax.plot(ts, elevs, color='#546e7a', linewidth=0.8, alpha=0.5)

        ax.axhline(y=ST_CLINICAL_THRESHOLD, color='#ff9800', linestyle='--',
                   linewidth=1.0, alpha=0.8, label=f'{ST_CLINICAL_THRESHOLD} mV clinical')
        ax.axhline(y=ST_CRITICAL_THRESHOLD, color='#ff5252', linestyle='--',
                   linewidth=1.0, alpha=0.8, label=f'{ST_CRITICAL_THRESHOLD} mV STEMI')
        ax.axhline(y=0, color='#546e7a', linewidth=0.6, alpha=0.4)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('ST Elevation (mV)', fontsize=8)
        st_alerts = a.st_alert_count
        ax.set_title(f'ST Elevation  |  Critical alerts: {st_alerts}',
                     fontsize=8.5, color=self._WHITE)
        ax.legend(fontsize=6.5, loc='upper left',
                  facecolor=self._BG_DARK, labelcolor=self._WHITE, framealpha=0.7)
        ax.grid(True, alpha=0.12, color='#ffffff')

        # ---- Panel 4: Heart rate trend ----
        ax = self.ax_hr
        ax.clear()
        self._style(ax)
        hr_list = list(a.heart_rate_history)
        if hr_list:
            x = list(range(len(hr_list)))
            ax.plot(x, hr_list, color='#e91e63', linewidth=1.5,
                    marker='o', markersize=2.5)
            ax.axhspan(60, 100, alpha=0.07, color='#4caf50')
            ax.axhline(y=60,  color='#4caf50', linestyle=':', linewidth=0.8, alpha=0.5)
            ax.axhline(y=100, color='#4caf50', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.set_ylim(30, 200)
        ax.set_xlabel('Beat #', fontsize=8)
        ax.set_ylabel('BPM', fontsize=8)
        ax.set_title(f'Heart Rate  |  Current: {a.current_hr():.0f} BPM',
                     fontsize=8.5, color=self._WHITE)
        ax.grid(True, alpha=0.12, color='#ffffff')

        # ---- Panel 5: Ensemble gauge ----
        ax = self.ax_ensemble
        ax.clear()
        ensemble = a.latest_ensemble
        st_elev  = a.latest_st

        # Determine alert level — worst of ML ensemble OR ST elevation
        st_critical = not np.isnan(st_elev) and st_elev >= ST_CRITICAL_THRESHOLD
        st_warning  = not np.isnan(st_elev) and st_elev >= ST_CLINICAL_THRESHOLD

        if ensemble >= ENSEMBLE_THRESHOLD or st_critical:
            bg_color    = '#b71c1c'
            bar_color   = '#ff5252'
            status_text = 'ANOMALY DETECTED'
            if st_critical:
                sub_text = f'STEMI WARNING: ST +{st_elev:.3f} mV | Ensemble: {ensemble:.3f}'
            else:
                sub_text = f'Arrhythmia: {ensemble:.3f} | ST: {"N/A" if np.isnan(st_elev) else f"{st_elev:.3f} mV"}'
        elif ensemble >= 0.4 or st_warning:
            bg_color    = '#e65100'
            bar_color   = '#ff9800'
            status_text = 'ELEVATED RISK'
            sub_text    = f'Ensemble: {ensemble:.3f} | ST: {"N/A" if np.isnan(st_elev) else f"{st_elev:.3f} mV"}'
        else:
            bg_color    = '#1b5e20'
            bar_color   = '#69f0ae'
            status_text = 'NORMAL'
            sub_text    = f'Ensemble: {ensemble:.3f} | ST: {"N/A" if np.isnan(st_elev) else f"{st_elev:.3f} mV"}'

        ax.set_facecolor(bg_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(bg_color)

        ax.barh(0, ensemble, height=0.5, color=bar_color, alpha=0.85)
        ax.barh(0, 1.0, height=0.5, color='#ffffff', alpha=0.07)
        ax.axvline(x=ENSEMBLE_THRESHOLD, color='white', linewidth=2,
                   linestyle='--', alpha=0.7)

        ax.text(0.5, 0.74, status_text, transform=ax.transAxes,
                ha='center', va='center', fontsize=18, fontweight='bold',
                color='white')
        ax.text(0.5, 0.24, sub_text, transform=ax.transAxes,
                ha='center', va='center', fontsize=9, color='white', alpha=0.9)

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 1.0)
        ax.set_xlabel('Arrhythmia Probability', fontsize=8, color='white')
        ax.tick_params(colors='white')
        ax.set_title('Ensemble Decision', fontsize=8.5, color=self._WHITE)
        ax.set_yticks([])

        # Console status line
        if frame % 25 == 0:
            st_str = f'{st_elev:.3f}' if not np.isnan(st_elev) else '  N/A'
            print(f"\r  Beats={a.total_beats:4d} | "
                  f"HR={a.current_hr():5.0f} BPM | "
                  f"Arr={a.abnormal_beats:3d} | "
                  f"ST={st_str} mV | "
                  f"Ens={ensemble:.3f} | "
                  f"{'ALERT' if a.alert_active or st_critical else 'OK   '}",
                  end='', flush=True)


# ======================================================================
# MIT-BIH record streaming

def load_mitdb_record(record_path):
    """Read a MIT-BIH record via wfdb, return (timestamps_ms, signal)."""
    import wfdb
    abs_path = record_path if os.path.isabs(record_path) \
        else os.path.join(_project_root, record_path)
    rec = wfdb.rdrecord(abs_path)
    signal = rec.p_signal[:, 0].astype(np.float32)
    n = len(signal)
    timestamps_ms = (np.arange(n, dtype=np.float64) / FS * 1000.0)
    return timestamps_ms, signal


def simulate_live_stream(record_path, analyzer, speed=1.0, loop=False):
    """Stream a MIT-BIH record sample-by-sample.

    Same threading pattern as simulate_live_stream() in
    test-demo/live_ecg_analysis.py, updated for 360Hz.
    """
    print(f"\n  Streaming: {record_path}")
    print(f"  Speed: {speed}x  |  Loop: {loop}")
    print(f"  Rate: {FS} Hz ({SAMPLE_INTERVAL*1000:.2f} ms/sample)\n")

    timestamps, signal = load_mitdb_record(record_path)
    n_samples = len(signal)
    iteration = 0

    try:
        while True:
            iteration += 1
            if loop and iteration > 1:
                print(f"\n  [LOOP {iteration}]")

            start_wall = time.time()
            ts_offset = (timestamps[-1] + 10.0) * (iteration - 1)

            for i in range(n_samples):
                analyzer.buffer.add_sample(timestamps[i] + ts_offset, signal[i])

                if i % 10 == 0:
                    analyzer.analyze_new_peaks()

                # Real-time pacing
                elapsed  = time.time() - start_wall
                expected = (i + 1) * SAMPLE_INTERVAL / speed
                wait     = expected - elapsed
                if wait > 0:
                    time.sleep(wait)

            if not loop:
                print(f"\n  Stream complete: {n_samples} samples "
                      f"({n_samples/FS/60:.1f} min)")
                break

    except KeyboardInterrupt:
        print(f"\n  Interrupted after {analyzer.total_beats} beats")


# ======================================================================
# Entry point

def main():
    parser = argparse.ArgumentParser(
        description='Live ECG demo — arrhythmia detection + ST elevation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Records to try:
  119 — mix of normal beats and PVCs (good for arrhythmia demo)
  108 — high proportion of arrhythmias and signal artifacts
  100 — mostly normal sinus rhythm (good baseline)
  106 — frequent PVCs

Examples:
  python demo/live_demo.py --record data/mitdb_raw/119 --speed 5.0 --loop
  python demo/live_demo.py --record data/mitdb_raw/108 --speed 3.0 --loop
  python demo/live_demo.py --no-plot   # console only
        """
    )
    parser.add_argument('--cnn',  default=DEFAULT_CNN,
                        help='CNN ResNet model path')
    parser.add_argument('--lstm', default=DEFAULT_LSTM,
                        help='Attention LSTM model path')
    parser.add_argument('--gru',  default=DEFAULT_GRU,
                        help='BiGRU model path')
    parser.add_argument('--record', default='data/mitdb_raw/119',
                        help='MIT-BIH record path without extension')
    parser.add_argument('--speed', type=float, default=5.0,
                        help='Playback speed multiplier (default: 5.0)')
    parser.add_argument('--loop', action='store_true',
                        help='Loop record continuously')
    parser.add_argument('--no-plot', action='store_true',
                        help='Console output only')
    args = parser.parse_args()

    # Validate record
    rec_abs = args.record if os.path.isabs(args.record) \
        else os.path.join(_project_root, args.record)
    if not os.path.exists(rec_abs + '.dat'):
        print(f"ERROR: Record not found: {rec_abs}.dat")
        print("Run: python demo/download_mitdb.py")
        sys.exit(1)

    try:
        import wfdb  # noqa: F401
    except ImportError:
        print("ERROR: wfdb not installed. Run: pip install wfdb")
        sys.exit(1)

    print("=" * 60)
    print("  Heart Attack Detection — Live Demo")
    print("=" * 60)
    print(f"  Record : {args.record}")
    print(f"  Speed  : {args.speed}x real-time")
    print(f"  Loop   : {args.loop}")
    print("\nLoading models...")

    analyzer = MultiModelAnalyzer()
    analyzer.load_models(args.cnn, args.lstm, args.gru)

    stream_thread = threading.Thread(
        target=simulate_live_stream,
        args=(args.record, analyzer, args.speed, args.loop),
        daemon=True
    )
    stream_thread.start()

    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation

            viz = LiveDemoVisualizer(analyzer)
            ani = FuncAnimation(
                viz.fig, viz.update,
                interval=100,
                blit=False,
                cache_frame_data=False
            )
            print("\nVisualization started. Close the window to stop.\n")
            plt.show()

        except Exception as e:
            print(f"\nVisualization error: {e}")
            args.no_plot = True

    if args.no_plot:
        try:
            stream_thread.join()
        except KeyboardInterrupt:
            pass

    # Final summary
    a = analyzer
    print("\n\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  Total beats:      {a.total_beats}")
    if a.total_beats > 0:
        pct = 100 * a.abnormal_beats / a.total_beats
        print(f"  Arrhythmia beats: {a.abnormal_beats} ({pct:.1f}%)")
    print(f"  ST alerts (≥{ST_CRITICAL_THRESHOLD}mV): {a.st_alert_count}")
    if a.current_hr() > 0:
        print(f"  Final heart rate: {a.current_hr():.0f} BPM")
    print(f"  Models loaded:    {list(a.models.keys())}")
    print("=" * 60)


if __name__ == '__main__':
    main()
