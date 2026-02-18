"""
predict_heart_problems.py

Run LSTM model on ECG data to detect potential heart problems (ST elevation/STEMI).

Usage:
    python predict_heart_problems.py --input ecg_denoised.csv
    python predict_heart_problems.py --input ecg_denoised.csv --model lstm_st_classifier.h5
    python predict_heart_problems.py --input ecg_denoised.csv --train  # Train new model first
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import tensorflow as tf
from tensorflow.keras import layers, models
import argparse
import os
import sys

# ==================== CONFIGURATION ====================
FS = 250                    # Sampling frequency (Hz)
BEAT_WINDOW_MS = 600        # Window length centered on R-peak (ms)
ST_WINDOW_AFTER_S_MS = (40, 140)  # ST window relative to S-wave (ms)
ST_ELEV_THRESHOLD = 0.20    # ST elevation threshold for labeling
PREDICTION_THRESHOLD = 0.5  # Probability threshold for positive prediction
# =======================================================

# Helper functions (same as lstm_check.py)
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(x, lowcut=0.5, highcut=40.0, fs=FS, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, x)

def load_data(csv_path):
    """Load ECG data from CSV file."""
    df = pd.read_csv(csv_path)
    time_col = [c for c in df.columns if "time" in c.lower()][0]
    if 'ecg_denoised' in df.columns:
        sig_col = 'ecg_denoised'
    elif 'ecg_value' in df.columns:
        sig_col = 'ecg_value'
    else:
        raise ValueError("CSV must contain 'ecg_denoised' or 'ecg_value' column.")
    t = df[time_col].to_numpy()
    sig = df[sig_col].astype(float).to_numpy()
    return t, sig, df

def detect_r_peaks(signal, fs=FS):
    """Detect R-peaks in ECG signal."""
    filtered = bandpass_filter(signal, fs=fs)
    height_thr = np.mean(filtered) + 0.5 * np.std(filtered)
    min_distance = int(0.25 * fs)  # 250ms refractory minimum
    peaks, props = find_peaks(filtered, height=height_thr, distance=min_distance)
    return peaks, filtered

def extract_beat_windows(signal, peaks, fs=FS, window_ms=BEAT_WINDOW_MS):
    """Extract per-beat windows centered on R peaks."""
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

def estimate_s_index_for_peak(filtered_signal, r_idx, fs=FS):
    """Estimate S-wave index (local minimum after R peak)."""
    start = r_idx + int(0.01 * fs)
    end = r_idx + int(0.08 * fs)
    if end >= len(filtered_signal):
        return None
    local_min_idx = start + np.argmin(filtered_signal[start:end])
    return local_min_idx

def compute_st_elevation(filtered_signal, r_indices, fs=FS, st_window_after_s_ms=ST_WINDOW_AFTER_S_MS):
    """Compute ST elevation measure for each beat."""
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
        st_end = s_idx + int(st_window_after_s_ms[1] * fs / 1000.0)
        if st_end >= len(filtered_signal):
            st_measures.append(np.nan)
            continue
        # Compute baseline around PR segment
        baseline_start = max(0, r - int(0.12 * fs))
        baseline_end = max(0, r - int(0.02 * fs))
        baseline = np.median(filtered_signal[baseline_start:baseline_end]) if baseline_end > baseline_start else np.median(filtered_signal[max(0,r-10):r])
        st_mean = np.mean(filtered_signal[st_start:st_end])
        st_measures.append(st_mean - baseline)
    return np.array(st_measures), s_list

def build_model(input_shape):
    """Build LSTM model architecture."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(0.2),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model_on_data(csv_path, model_save_path="lstm_st_classifier.h5", epochs=30):
    """Train LSTM model on the provided data."""
    print(f" Training LSTM model on {csv_path}...")
    print("   (This may take a few minutes)")
    
    # Load and process data
    t, sig, df = load_data(csv_path)
    peaks, filtered = detect_r_peaks(sig)
    print(f"   Detected {len(peaks)} R-peaks")
    
    windows, valid_peaks = extract_beat_windows(filtered, peaks)
    print(f"   Extracted {len(windows)} beat windows")
    
    st_measures, s_indices = compute_st_elevation(filtered, valid_peaks)
    labels = (st_measures > ST_ELEV_THRESHOLD).astype(int)
    
    # Remove NaN values
    valid_mask = ~np.isnan(st_measures)
    X = windows[valid_mask]
    y = labels[valid_mask]
    
    if len(X) == 0:
        raise ValueError("No valid beats found. Check your signal quality.")
    
    print(f"   Class distribution: {np.bincount(y)}")
    
    # Normalize
    X_mean = X.mean(axis=1, keepdims=True)
    X_std = X.std(axis=1, keepdims=True) + 1e-8
    X_norm = (X - X_mean) / X_std
    
    # Reshape for LSTM
    X_lstm = X_norm.reshape((X_norm.shape[0], X_norm.shape[1], 1))
    
    # Build and train model
    model = build_model((X_lstm.shape[1], X_lstm.shape[2]))
    print("\n   Model architecture:")
    model.summary()
    
    # Train on all data (since we're using synthetic labels)
    print(f"\n   Training for {epochs} epochs...")
    history = model.fit(X_lstm, y, epochs=epochs, batch_size=32, verbose=1, validation_split=0.2)
    
    # Save model
    model.save(model_save_path)
    print(f"\n Model saved to {model_save_path}")
    
    return model

def predict_heart_problems(csv_path, model_path=None, train_if_missing=True):
    """
    Run LSTM model on ECG data to predict heart problems.
    
    Returns:
        dict with predictions, probabilities, and interpretation
    """
    print("=" * 60)
    print("HEART PROBLEM DETECTION ANALYSIS")
    print("=" * 60)
    
    # Load data
    print(f"\n Loading ECG data from: {csv_path}")
    t, sig, df = load_data(csv_path)
    print(f"   Signal length: {len(sig)} samples ({len(sig)/FS:.1f} seconds)")
    
    # Detect R-peaks
    print("\nDetecting heartbeats...")
    peaks, filtered = detect_r_peaks(sig)
    print(f"   Found {len(peaks)} R-peaks")
    
    if len(peaks) < 3:
        print("WARNING: Too few heartbeats detected. Results may be unreliable.")
        return None
    
    # Calculate heart rate
    if len(peaks) > 1:
        rr_intervals = np.diff(t[peaks]) / 1000.0  # Convert ms to seconds
        heart_rate = 60.0 / np.mean(rr_intervals)
        print(f"   Heart Rate: {heart_rate:.1f} BPM")
    
    # Extract beat windows
    print("\n📊 Extracting beat windows...")
    windows, valid_peaks = extract_beat_windows(filtered, peaks)
    print(f"   Extracted {len(windows)} valid beat windows")
    
    if len(windows) == 0:
        print("ERROR: Could not extract valid beat windows.")
        return None
    
    # Compute ST elevation measures
    st_measures, s_indices = compute_st_elevation(filtered, valid_peaks)
    valid_mask = ~np.isnan(st_measures)
    X = windows[valid_mask]
    st_measures_valid = st_measures[valid_mask]
    
    if len(X) == 0:
        print("ERROR: No valid beats after ST analysis.")
        return None
    
    # Normalize
    X_mean = X.mean(axis=1, keepdims=True)
    X_std = X.std(axis=1, keepdims=True) + 1e-8
    X_norm = (X - X_mean) / X_std
    X_lstm = X_norm.reshape((X_norm.shape[0], X_norm.shape[1], 1))
    
    # Load or train model
    if model_path and os.path.exists(model_path):
        print(f"\n🤖 Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
    elif train_if_missing:
        print(f"\n🤖 Model not found. Training new model...")
        model = train_model_on_data(csv_path, model_path or "lstm_st_classifier.h5")
    else:
        print(f"\n❌ ERROR: Model not found at {model_path}")
        print("   Use --train flag to train a new model first.")
        return None
    
    # Make predictions
    print("\n🔮 Running predictions...")
    predictions_proba = model.predict(X_lstm, verbose=0).ravel()
    predictions = (predictions_proba >= PREDICTION_THRESHOLD).astype(int)
    
    # Analyze results
    num_positive = np.sum(predictions)
    num_total = len(predictions)
    positive_percentage = (num_positive / num_total) * 100
    mean_probability = np.mean(predictions_proba)
    max_probability = np.max(predictions_proba)
    
    # ST elevation analysis
    mean_st_elevation = np.mean(st_measures_valid)
    max_st_elevation = np.max(st_measures_valid)
    elevated_beats = np.sum(st_measures_valid > 0.1)  # Clinical threshold: 0.1mV
    
    # Results summary
    print("\n" + "=" * 60)
    print("📋 RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n📊 Model Predictions:")
    print(f"   • Total beats analyzed: {num_total}")
    print(f"   • Beats flagged as abnormal: {num_positive} ({positive_percentage:.1f}%)")
    print(f"   • Mean prediction probability: {mean_probability:.3f}")
    print(f"   • Maximum prediction probability: {max_probability:.3f}")
    
    print(f"\n📈 ST Segment Analysis:")
    print(f"   • Mean ST elevation: {mean_st_elevation:.3f} mV")
    print(f"   • Maximum ST elevation: {max_st_elevation:.3f} mV")
    print(f"   • Beats with ST elevation >0.1mV: {elevated_beats} ({elevated_beats/num_total*100:.1f}%)")
    
    # Interpretation
    print(f"\n🏥 CLINICAL INTERPRETATION:")
    print("-" * 60)
    
    risk_level = "LOW"
    recommendation = "Continue routine monitoring."
    
    if max_probability >= 0.9 or max_st_elevation >= 0.2:
        risk_level = "CRITICAL"
        recommendation = "⚠️  SEEK IMMEDIATE MEDICAL ATTENTION - Possible STEMI detected!"
    elif max_probability >= 0.75 or positive_percentage >= 30 or max_st_elevation >= 0.15:
        risk_level = "HIGH"
        recommendation = "⚠️  URGENT: Contact healthcare provider immediately."
    elif max_probability >= 0.6 or positive_percentage >= 15 or max_st_elevation >= 0.1:
        risk_level = "MODERATE"
        recommendation = "⚠️  Schedule medical consultation soon."
    elif positive_percentage >= 5 or mean_probability >= 0.4:
        risk_level = "ELEVATED"
        recommendation = "Monitor closely and consider medical consultation."
    
    print(f"   Risk Level: {risk_level}")
    print(f"   Recommendation: {recommendation}")
    
    print("\n" + "=" * 60)
    print("⚠️  DISCLAIMER: This analysis is for supplementary monitoring only.")
    print("   Always consult qualified healthcare professionals for medical decisions.")
    print("   In case of chest pain or cardiac symptoms, call emergency services immediately.")
    print("=" * 60)
    
    # Return structured results
    results = {
        'total_beats': num_total,
        'positive_beats': num_positive,
        'positive_percentage': positive_percentage,
        'mean_probability': mean_probability,
        'max_probability': max_probability,
        'mean_st_elevation': mean_st_elevation,
        'max_st_elevation': max_st_elevation,
        'elevated_beats': elevated_beats,
        'risk_level': risk_level,
        'recommendation': recommendation,
        'heart_rate': heart_rate if len(peaks) > 1 else None,
        'predictions': predictions,
        'probabilities': predictions_proba,
        'st_measures': st_measures_valid
    }
    
    return results

def plot_results(csv_path, results):
    """Create visualization of predictions."""
    if results is None:
        return
    
    t, sig, df = load_data(csv_path)
    _, filtered = detect_r_peaks(sig)
    peaks, _ = detect_r_peaks(sig)
    windows, valid_peaks = extract_beat_windows(filtered, peaks)
    st_measures, _ = compute_st_elevation(filtered, valid_peaks)
    valid_mask = ~np.isnan(st_measures)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: ECG signal with predictions
    ax1 = axes[0]
    time_col = [c for c in df.columns if "time" in c.lower()][0]
    t_array = df[time_col].to_numpy()
    ax1.plot(t_array, filtered, 'b-', linewidth=0.8, label='Filtered ECG')
    ax1.scatter(t_array[peaks], filtered[peaks], c='red', marker='o', s=30, label='R-peaks', zorder=5)
    
    # Highlight beats with positive predictions
    positive_beats = results['predictions'] == 1
    if np.any(positive_beats):
        positive_peaks = valid_peaks[valid_mask][positive_beats]
        ax1.scatter(t_array[positive_peaks], filtered[positive_peaks], 
                   c='orange', marker='*', s=100, label='Abnormal beats', zorder=6)
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'ECG Signal Analysis - Risk Level: {results["risk_level"]}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Prediction probabilities and ST measures
    ax2 = axes[1]
    beat_numbers = np.arange(len(results['probabilities']))
    ax2_twin = ax2.twinx()
    
    # Probabilities
    colors = ['green' if p < 0.5 else 'orange' if p < 0.75 else 'red' for p in results['probabilities']]
    ax2.scatter(beat_numbers, results['probabilities'], c=colors, alpha=0.6, s=50, label='Prediction Probability')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold')
    ax2.set_ylabel('Prediction Probability', color='black')
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Beat Number')
    
    # ST measures
    ax2_twin.scatter(beat_numbers, results['st_measures'], c='blue', marker='x', alpha=0.5, s=30, label='ST Elevation (mV)')
    ax2_twin.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Clinical Threshold (0.1mV)')
    ax2_twin.set_ylabel('ST Elevation (mV)', color='blue')
    
    ax2.set_title('Beat-by-Beat Analysis: Prediction Probabilities & ST Elevation')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('heart_analysis_results.png', dpi=150, bbox_inches='tight')
    print(f"\n Visualization saved to: heart_analysis_results.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Run LSTM model on ECG data to detect heart problems',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run prediction with existing model
  python predict_heart_problems.py --input ecg_denoised.csv
  
  # Run with specific model file
  python predict_heart_problems.py --input ecg_denoised.csv --model lstm_st_classifier.h5
  
  # Train new model first, then predict
  python predict_heart_problems.py --input ecg_denoised.csv --train
  
  # Predict without training if model missing
  python predict_heart_problems.py --input ecg_denoised.csv --no-train
        """
    )
    parser.add_argument('--input', '-i', required=True, help='Input CSV file with ECG data')
    parser.add_argument('--model', '-m', default='lstm_st_classifier.h5', help='Path to trained model file')
    parser.add_argument('--train', action='store_true', help='Train new model if not found')
    parser.add_argument('--no-train', action='store_true', help='Do not train model if missing (will error)')
    parser.add_argument('--plot', action='store_true', help='Generate visualization plots')
    
    args = parser.parse_args()
    
    # Check input file
    if not os.path.exists(args.input):
        print(f" ERROR: Input file not found: {args.input}")
        sys.exit(1)
    
    # Run prediction
    train_if_missing = args.train or not args.no_train
    results = predict_heart_problems(args.input, args.model, train_if_missing)
    
    # Plot if requested
    if args.plot and results:
        plot_results(args.input, results)
    
    # Exit with appropriate code
    if results and results['risk_level'] in ['CRITICAL', 'HIGH']:
        sys.exit(1)  # Exit with error code for high-risk cases
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()

