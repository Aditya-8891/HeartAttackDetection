"""
live_ecg_analysis.py

Simulates live ECG data streaming and performs real-time LSTM analysis.

This script:
- Reads ECG data from CSV and streams it sample-by-sample (simulating Arduino input)
- Maintains a sliding window buffer for real-time processing
- Detects R-peaks as data arrives
- Runs LSTM predictions on each detected heartbeat
- Displays real-time results, heart rate, and alerts

Usage:
    python live_ecg_analysis.py --input ecg_denoised.csv
    
    python live_ecg_analysis.py --input ecg_denoised.csv --speed 1.0  # 1.0 = real-time, 2.0 = 2x speed
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import butter, filtfilt, find_peaks
import tensorflow as tf
from tensorflow.keras import models
import argparse
import time
import os
import sys
from collections import deque
import threading

# ==================== CONFIGURATION ====================
FS = 250                    # Sampling frequency (Hz)
SAMPLE_INTERVAL = 1.0 / FS  # Time between samples (4ms)
BEAT_WINDOW_MS = 600        # Window length centered on R-peak (ms)
ST_WINDOW_AFTER_S_MS = (40, 140)  # ST window relative to S-wave (ms)
PREDICTION_THRESHOLD = 0.5  # Probability threshold for positive prediction
BUFFER_SIZE_SECONDS = 10    # Size of sliding window buffer (seconds)
MIN_BEATS_FOR_ANALYSIS = 3  # Minimum beats needed before analysis
# =======================================================

class ECGDataBuffer:
    """Circular buffer for real-time ECG data streaming."""
    def __init__(self, window_size_seconds=BUFFER_SIZE_SECONDS, sampling_rate=FS):
        self.window_size = window_size_seconds
        self.sampling_rate = sampling_rate
        self.buffer_size = int(window_size_seconds * sampling_rate)
        self.data = deque(maxlen=self.buffer_size)
        self.timestamps = deque(maxlen=self.buffer_size)
        self.lock = threading.Lock()
    
    def add_sample(self, timestamp, value):
        """Add a new ECG sample to the buffer."""
        with self.lock:
            self.data.append(value)
            self.timestamps.append(timestamp)
    
    def get_array(self):
        """Get current buffer as numpy array."""
        with self.lock:
            return np.array(self.data), np.array(self.timestamps)
    
    def is_ready(self, min_samples=None):
        """Check if buffer has enough data for analysis."""
        if min_samples is None:
            min_samples = int(2.0 * self.sampling_rate)  # At least 2 seconds
        return len(self.data) >= min_samples

class LiveECGAnalyzer:
    """Real-time ECG analyzer with LSTM predictions."""
    def __init__(self, model_path=None, csv_path=None):
        self.buffer = ECGDataBuffer()
        self.model = None
        self.model_path = model_path
        self.csv_path = csv_path
        
        # Analysis state
        self.detected_peaks = []
        self.last_peak_time = 0
        self.heart_rate_history = deque(maxlen=10)
        self.prediction_history = deque(maxlen=50)
        self.st_elevation_history = deque(maxlen=50)
        self.alerts = deque(maxlen=20)
        
        # Statistics
        self.total_beats = 0
        self.abnormal_beats = 0
        self.max_st_elevation = 0.0
        
        # Load model
        if model_path and os.path.exists(model_path):
            print(f"🤖 Loading LSTM model from: {model_path}")
            self.model = tf.keras.models.load_model(model_path)
        else:
            print("⚠️  WARNING: No model found. Predictions will be disabled.")
            print("   Train a model first using: python predict_heart_problems.py --input <csv> --train")
    
    def butter_bandpass(self, lowcut, highcut, fs, order=3):
        nyquist = 0.5 * fs
        low, high = lowcut / nyquist, highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def bandpass_filter(self, x, lowcut=0.5, highcut=40.0, fs=FS, order=3):
        nyquist = 0.5 * fs
        low, high = lowcut / nyquist, highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, x)
    
    def detect_r_peaks(self, signal, timestamps):
        """Detect R-peaks in the current buffer."""
        if len(signal) < int(0.5 * FS):  # Need at least 0.5 seconds
            return [], signal
        
        filtered = self.bandpass_filter(signal)
        height_thr = np.mean(filtered) + 0.5 * np.std(filtered)
        min_distance = int(0.25 * FS)  # 250ms refractory minimum
        
        peaks, props = find_peaks(filtered, height=height_thr, distance=min_distance)
        
        # Convert peak indices to timestamps
        peak_times = timestamps[peaks] if len(peaks) > 0 else []
        
        # Only return new peaks (not already detected)
        new_peaks = []
        for peak_idx, peak_time in zip(peaks, peak_times):
            if peak_time > self.last_peak_time:
                new_peaks.append((peak_idx, peak_time))
                self.last_peak_time = peak_time
        
        return new_peaks, filtered
    
    def estimate_s_index(self, filtered_signal, r_idx, fs=FS):
        """Estimate S-wave index (local minimum after R peak)."""
        start = r_idx + int(0.01 * fs)
        end = r_idx + int(0.08 * fs)
        if end >= len(filtered_signal):
            return None
        local_min_idx = start + np.argmin(filtered_signal[start:end])
        return local_min_idx
    
    def compute_st_elevation(self, filtered_signal, r_idx, fs=FS):
        """Compute ST elevation for a single beat."""
        s_idx = self.estimate_s_index(filtered_signal, r_idx, fs)
        if s_idx is None:
            return np.nan
        
        st_start = s_idx + int(ST_WINDOW_AFTER_S_MS[0] * fs / 1000.0)
        st_end = s_idx + int(ST_WINDOW_AFTER_S_MS[1] * fs / 1000.0)
        
        if st_end >= len(filtered_signal):
            return np.nan
        
        # Compute baseline around PR segment
        baseline_start = max(0, r_idx - int(0.12 * fs))
        baseline_end = max(0, r_idx - int(0.02 * fs))
        baseline = np.median(filtered_signal[baseline_start:baseline_end]) if baseline_end > baseline_start else np.median(filtered_signal[max(0,r_idx-10):r_idx])
        
        st_mean = np.mean(filtered_signal[st_start:st_end])
        st_elevation = st_mean - baseline
        
        return st_elevation
    
    def extract_beat_window(self, signal, r_idx, window_ms=BEAT_WINDOW_MS, fs=FS):
        """Extract a single beat window centered on R peak."""
        half = int((window_ms/1000.0) * fs / 2)
        start = r_idx - half
        end = r_idx + half
        
        if start >= 0 and end <= len(signal):
            return signal[start:end]
        return None
    
    def predict_beat(self, beat_window):
        """Run LSTM prediction on a single beat."""
        if self.model is None or beat_window is None:
            return None, None
        
        # Normalize
        beat_mean = np.mean(beat_window)
        beat_std = np.std(beat_window) + 1e-8
        beat_norm = (beat_window - beat_mean) / beat_std
        
        # Reshape for LSTM: (1, timesteps, features)
        X = beat_norm.reshape((1, len(beat_norm), 1))
        
        # Predict
        prediction_proba = self.model.predict(X, verbose=0)[0][0]
        prediction = 1 if prediction_proba >= PREDICTION_THRESHOLD else 0
        
        return prediction, prediction_proba
    
    def analyze_new_peaks(self):
        """Analyze newly detected R-peaks."""
        signal, timestamps = self.buffer.get_array()
        
        if not self.buffer.is_ready():
            return
        
        new_peaks, filtered = self.detect_r_peaks(signal, timestamps)
        
        for peak_idx, peak_time in new_peaks:
            # Extract beat window
            beat_window = self.extract_beat_window(filtered, peak_idx)
            
            if beat_window is None:
                continue
            
            # Compute ST elevation
            st_elevation = self.compute_st_elevation(filtered, peak_idx)
            
            # LSTM prediction
            prediction, proba = self.predict_beat(beat_window)
            
            # Update statistics
            self.total_beats += 1
            if prediction == 1:
                self.abnormal_beats += 1
            
            if not np.isnan(st_elevation):
                self.max_st_elevation = max(self.max_st_elevation, st_elevation)
                self.st_elevation_history.append({
                    'time': peak_time,
                    'elevation': st_elevation,
                    'prediction': prediction,
                    'probability': proba
                })
            
            if proba is not None:
                self.prediction_history.append({
                    'time': peak_time,
                    'probability': proba,
                    'prediction': prediction,
                    'st_elevation': st_elevation
                })
            
            # Calculate heart rate
            if len(self.detected_peaks) > 0:
                last_peak_time = self.detected_peaks[-1][1]
                rr_interval = (peak_time - last_peak_time) / 1000.0  # Convert ms to seconds
                if rr_interval > 0:
                    heart_rate = 60.0 / rr_interval
                    self.heart_rate_history.append(heart_rate)
            
            self.detected_peaks.append((peak_idx, peak_time))
            
            # Generate alerts
            self.check_alerts(peak_time, proba, st_elevation, prediction)
    
    def check_alerts(self, timestamp, proba, st_elevation, prediction):
        """Check for conditions that require alerts."""
        alert = None
        
        # Critical ST elevation
        if not np.isnan(st_elevation) and st_elevation >= 0.2:
            alert = {
                'time': timestamp,
                'level': 'CRITICAL',
                'type': 'ST_ELEVATION',
                'message': f'⚠️ CRITICAL: ST elevation {st_elevation:.3f}mV detected! Possible STEMI!',
                'value': st_elevation
            }
        elif not np.isnan(st_elevation) and st_elevation >= 0.15:
            alert = {
                'time': timestamp,
                'level': 'HIGH',
                'type': 'ST_ELEVATION',
                'message': f'⚠️ HIGH: Significant ST elevation {st_elevation:.3f}mV detected!',
                'value': st_elevation
            }
        elif proba is not None and proba >= 0.9:
            alert = {
                'time': timestamp,
                'level': 'CRITICAL',
                'type': 'ML_PREDICTION',
                'message': f'⚠️ CRITICAL: High abnormality probability {proba:.3f}',
                'value': proba
            }
        elif proba is not None and proba >= 0.75:
            alert = {
                'time': timestamp,
                'level': 'HIGH',
                'type': 'ML_PREDICTION',
                'message': f'⚠️ HIGH: Abnormal pattern detected (probability: {proba:.3f})',
                'value': proba
            }
        
        if alert:
            self.alerts.append(alert)
            print(f"\n🚨 ALERT [{alert['level']}]: {alert['message']}")
            print(f"   Timestamp: {timestamp}ms\n")

class LiveECGVisualizer:
    """Real-time visualization of ECG analysis."""
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.fig = None
        self.axes = None
        self.setup_plot()
    
    def setup_plot(self):
        """Setup matplotlib figure and axes."""
        self.fig, self.axes = plt.subplots(3, 1, figsize=(14, 10))
        self.fig.suptitle('Live ECG Analysis - Real-time LSTM Monitoring', fontsize=14, fontweight='bold')
        
        # ECG Signal plot
        self.ax1 = self.axes[0]
        self.ax1.set_xlabel('Time (ms)')
        self.ax1.set_ylabel('Amplitude')
        self.ax1.set_title('ECG Signal (Live)')
        self.ax1.grid(True, alpha=0.3)
        
        # Predictions plot
        self.ax2 = self.axes[1]
        self.ax2.set_xlabel('Time (ms)')
        self.ax2.set_ylabel('Prediction Probability')
        self.ax2.set_title('LSTM Predictions (Real-time)')
        self.ax2.set_ylim(0, 1)
        self.ax2.axhline(y=PREDICTION_THRESHOLD, color='r', linestyle='--', alpha=0.5, label='Threshold')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend()
        
        # ST Elevation plot
        self.ax3 = self.axes[2]
        self.ax3.set_xlabel('Time (ms)')
        self.ax3.set_ylabel('ST Elevation (mV)')
        self.ax3.set_title('ST Segment Elevation Analysis')
        self.ax3.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Clinical Threshold (0.1mV)')
        self.ax3.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Critical Threshold (0.2mV)')
        self.ax3.grid(True, alpha=0.3)
        self.ax3.legend()
        
        plt.tight_layout()
    
    def update_plot(self, frame):
        """Update plots with latest data."""
        signal, timestamps = self.analyzer.buffer.get_array()
        
        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # Plot 1: ECG Signal
        if len(signal) > 0:
            self.ax1.plot(timestamps, signal, 'b-', linewidth=0.8, label='ECG Signal')
            
            # Mark detected R-peaks
            if len(self.analyzer.detected_peaks) > 0:
                peak_indices = [p[0] for p in self.analyzer.detected_peaks if p[0] < len(signal)]
                if len(peak_indices) > 0:
                    peak_times = timestamps[peak_indices]
                    peak_values = signal[peak_indices]
                    self.ax1.scatter(peak_times, peak_values, c='red', marker='o', s=30, label='R-peaks', zorder=5)
            
            self.ax1.set_xlabel('Time (ms)')
            self.ax1.set_ylabel('Amplitude')
            self.ax1.set_title(f'ECG Signal (Live) - Beats: {self.analyzer.total_beats} | HR: {self.get_current_hr():.1f} BPM')
            self.ax1.grid(True, alpha=0.3)
            self.ax1.legend()
        
        # Plot 2: Predictions
        if len(self.analyzer.prediction_history) > 0:
            pred_times = [p['time'] for p in self.analyzer.prediction_history]
            pred_probas = [p['probability'] for p in self.analyzer.prediction_history]
            pred_colors = ['red' if p >= 0.75 else 'orange' if p >= 0.5 else 'green' for p in pred_probas]
            
            self.ax2.scatter(pred_times, pred_probas, c=pred_colors, alpha=0.6, s=50)
            self.ax2.axhline(y=PREDICTION_THRESHOLD, color='r', linestyle='--', alpha=0.5, label='Threshold')
            self.ax2.set_xlabel('Time (ms)')
            self.ax2.set_ylabel('Prediction Probability')
            self.ax2.set_title(f'LSTM Predictions - Abnormal: {self.analyzer.abnormal_beats}/{self.analyzer.total_beats} ({self.analyzer.abnormal_beats/max(1,self.analyzer.total_beats)*100:.1f}%)')
            self.ax2.set_ylim(0, 1)
            self.ax2.grid(True, alpha=0.3)
            self.ax2.legend()
        
        # Plot 3: ST Elevation
        if len(self.analyzer.st_elevation_history) > 0:
            st_times = [s['time'] for s in self.analyzer.st_elevation_history]
            st_elevs = [s['elevation'] for s in self.analyzer.st_elevation_history]
            st_colors = ['red' if e >= 0.2 else 'orange' if e >= 0.1 else 'green' for e in st_elevs]
            
            self.ax3.scatter(st_times, st_elevs, c=st_colors, alpha=0.6, s=50)
            self.ax3.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Clinical Threshold (0.1mV)')
            self.ax3.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Critical Threshold (0.2mV)')
            self.ax3.set_xlabel('Time (ms)')
            self.ax3.set_ylabel('ST Elevation (mV)')
            self.ax3.set_title(f'ST Elevation Analysis - Max: {self.analyzer.max_st_elevation:.3f}mV')
            self.ax3.grid(True, alpha=0.3)
            self.ax3.legend()
        
        # Print status
        if frame % 10 == 0:  # Print every 10 frames
            self.print_status()
    
    def get_current_hr(self):
        """Get current heart rate."""
        if len(self.analyzer.heart_rate_history) > 0:
            return np.mean(list(self.analyzer.heart_rate_history)[-5:])  # Average of last 5
        return 0.0
    
    def print_status(self):
        """Print current analysis status."""
        hr = self.get_current_hr()
        print(f"\r📊 Status: Beats={self.analyzer.total_beats} | HR={hr:.1f} BPM | "
              f"Abnormal={self.analyzer.abnormal_beats} | Max ST={self.analyzer.max_st_elevation:.3f}mV | "
              f"Alerts={len(self.analyzer.alerts)}", end='', flush=True)

def simulate_live_stream(csv_path, analyzer, speed=1.0):
    """Simulate live ECG data streaming from CSV file."""
    print(f"\n📡 Starting live ECG stream simulation...")
    print(f"   Source: {csv_path}")
    print(f"   Speed: {speed}x real-time")
    print(f"   Sampling rate: {FS} Hz ({SAMPLE_INTERVAL*1000:.1f}ms per sample)\n")
    
    # Load data
    df = pd.read_csv(csv_path)
    time_col = [c for c in df.columns if "time" in c.lower()][0]
    if 'ecg_denoised' in df.columns:
        sig_col = 'ecg_denoised'
    elif 'ecg_value' in df.columns:
        sig_col = 'ecg_value'
    else:
        raise ValueError("CSV must contain 'ecg_denoised' or 'ecg_value' column.")
    
    timestamps = df[time_col].values
    values = df[sig_col].values.astype(float)
    
    # Stream data sample by sample
    start_time = time.time()
    sample_count = 0
    
    try:
        for i, (ts, val) in enumerate(zip(timestamps, values)):
            # Add sample to buffer
            analyzer.buffer.add_sample(ts, val)
            
            # Analyze new peaks periodically
            if i % 10 == 0:  # Check every 10 samples
                analyzer.analyze_new_peaks()
            
            # Simulate real-time delay
            elapsed = time.time() - start_time
            expected_time = (i + 1) * SAMPLE_INTERVAL / speed
            sleep_time = max(0, expected_time - elapsed)
            time.sleep(sleep_time)
            
            sample_count += 1
        
        print(f"\n✅ Stream complete: {sample_count} samples processed")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Stream interrupted by user")
        print(f"   Processed {sample_count} samples")

def main():
    parser = argparse.ArgumentParser(
        description='Simulate live ECG streaming with real-time LSTM analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (real-time speed)
  python live_ecg_analysis.py --input ecg_denoised.csv
  
  # Run with specific model
  python live_ecg_analysis.py --input ecg_denoised.csv --model lstm_st_classifier.h5
  
  # Run at 2x speed (faster simulation)
  python live_ecg_analysis.py --input ecg_denoised.csv --speed 2.0
  
  # Run at 0.5x speed (slower, easier to observe)
  python live_ecg_analysis.py --input ecg_denoised.csv --speed 0.5
  
  # Run without plotting (console output only)
  python live_ecg_analysis.py --input ecg_denoised.csv --no-plot
        """
    )
    parser.add_argument('--input', '-i', required=True, help='Input CSV file with ECG data')
    parser.add_argument('--model', '-m', default='lstm_st_classifier.h5', help='Path to trained LSTM model')
    parser.add_argument('--speed', '-s', type=float, default=1.0, help='Playback speed (1.0 = real-time, 2.0 = 2x speed)')
    parser.add_argument('--no-plot', action='store_true', help='Disable real-time plotting')
    
    args = parser.parse_args()
    
    # Check input file
    if not os.path.exists(args.input):
        print(f"❌ ERROR: Input file not found: {args.input}")
        sys.exit(1)
    
    # Initialize analyzer
    analyzer = LiveECGAnalyzer(model_path=args.model, csv_path=args.input)
    
    # Setup visualization
    visualizer = None
    ani = None
    if not args.no_plot:
        try:
            visualizer = LiveECGVisualizer(analyzer)
            # Start animation
            ani = FuncAnimation(visualizer.fig, visualizer.update_plot, interval=100, blit=False)
            print("📊 Real-time visualization enabled")
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize plotting: {e}")
            print("   Continuing with console output only...")
            args.no_plot = True
    
    # Start streaming in separate thread
    stream_thread = threading.Thread(
        target=simulate_live_stream,
        args=(args.input, analyzer, args.speed),
        daemon=True
    )
    stream_thread.start()
    
    # Keep main thread alive for plotting
    if not args.no_plot and visualizer:
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
    else:
        # Wait for stream to complete
        try:
            stream_thread.join()
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
    
    # Final summary
    print("\n" + "="*60)
    print("📋 FINAL ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total beats analyzed: {analyzer.total_beats}")
    if analyzer.total_beats > 0:
        print(f"Abnormal beats detected: {analyzer.abnormal_beats} ({analyzer.abnormal_beats/analyzer.total_beats*100:.1f}%)")
    print(f"Maximum ST elevation: {analyzer.max_st_elevation:.3f} mV")
    if len(analyzer.heart_rate_history) > 0:
        print(f"Average heart rate: {np.mean(list(analyzer.heart_rate_history)):.1f} BPM")
    print(f"Total alerts generated: {len(analyzer.alerts)}")
    
    if len(analyzer.alerts) > 0:
        print("\n🚨 ALERTS:")
        for alert in list(analyzer.alerts)[-10:]:  # Show last 10 alerts
            print(f"  [{alert['level']}] {alert['message']}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()

