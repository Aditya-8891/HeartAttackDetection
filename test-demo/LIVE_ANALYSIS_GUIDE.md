# Live ECG Analysis Guide

This guide explains how to use the `live_ecg_analysis.py` script to simulate real-time ECG streaming with LSTM analysis.

## Overview

The `live_ecg_analysis.py` script simulates live ECG data streaming (as if coming from an Arduino) and performs real-time analysis using the LSTM model. It:

- Streams ECG data sample-by-sample at 250Hz (4ms intervals)
- Maintains a sliding window buffer for continuous analysis
- Detects R-peaks in real-time as data arrives
- Runs LSTM predictions on each detected heartbeat
- Computes ST segment elevation measurements
- Generates alerts when abnormalities are detected
- Displays real-time visualizations (optional)

## Quick Start

### Basic Usage

```bash
cd test-demo
python live_ecg_analysis.py --input ecg_denoised.csv
```

This will:
- Load your ECG data from the CSV file
- Stream it at real-time speed (250Hz)
- Analyze each heartbeat as it arrives
- Display real-time plots (if matplotlib is available)
- Print alerts to console when abnormalities are detected

### With Specific Model

```bash
python live_ecg_analysis.py --input ecg_denoised.csv --model lstm_st_classifier.h5
```

### Adjust Playback Speed

**Faster (2x speed):**
```bash
python live_ecg_analysis.py --input ecg_denoised.csv --speed 2.0
```

**Slower (0.5x speed - easier to observe):**
```bash
python live_ecg_analysis.py --input ecg_denoised.csv --speed 0.5
```

### Console Output Only (No Plotting)

```bash
python live_ecg_analysis.py --input ecg_denoised.csv --no-plot
```

## What You'll See

### Real-Time Console Output

The script prints status updates and alerts:

```
📡 Starting live ECG stream simulation...
   Source: ecg_denoised.csv
   Speed: 1.0x real-time
   Sampling rate: 250 Hz (4.0ms per sample)

🤖 Loading LSTM model from: lstm_st_classifier.h5

📊 Status: Beats=5 | HR=72.3 BPM | Abnormal=0 | Max ST=0.045mV | Alerts=0

🚨 ALERT [HIGH]: ⚠️ HIGH: Significant ST elevation 0.152mV detected!
   Timestamp: 1800ms

📊 Status: Beats=12 | HR=75.1 BPM | Abnormal=3 | Max ST=0.220mV | Alerts=2
```

### Real-Time Plots (if enabled)

Three plots update in real-time:

1. **ECG Signal**: Shows the live ECG waveform with detected R-peaks marked
2. **LSTM Predictions**: Displays prediction probabilities for each beat
3. **ST Elevation**: Shows ST segment elevation measurements

## How It Works

### Data Flow

```
CSV File → Sample-by-Sample Stream → Circular Buffer
                                           ↓
                                    R-Peak Detection
                                           ↓
                                    Beat Window Extraction
                                           ↓
                                    ┌──────┴──────┐
                                    ↓             ↓
                              LSTM Prediction  ST Elevation
                                    ↓             ↓
                                    └──────┬──────┘
                                           ↓
                                    Alert Generation
```

### Key Components

1. **ECGDataBuffer**: Circular buffer that maintains the last 10 seconds of ECG data
2. **LiveECGAnalyzer**: Performs real-time analysis:
   - R-peak detection on the current buffer
   - Beat window extraction
   - LSTM prediction
   - ST elevation calculation
   - Alert generation

3. **LiveECGVisualizer**: Real-time plotting (optional)

### Alert Conditions

The script generates alerts when:

- **CRITICAL ST Elevation**: ST elevation ≥ 0.2mV (possible STEMI)
- **HIGH ST Elevation**: ST elevation ≥ 0.15mV (significant)
- **CRITICAL ML Prediction**: LSTM probability ≥ 0.9
- **HIGH ML Prediction**: LSTM probability ≥ 0.75

## Example Output

When running on the extended `ecg_denoised.csv` with ST elevation:

```
📡 Starting live ECG stream simulation...
   Source: ecg_denoised.csv
   Speed: 1.0x real-time
   Sampling rate: 250 Hz (4.0ms per sample)

🤖 Loading LSTM model from: lstm_st_classifier.h5

📊 Status: Beats=8 | HR=75.2 BPM | Abnormal=0 | Max ST=0.045mV | Alerts=0

🚨 ALERT [HIGH]: ⚠️ HIGH: Significant ST elevation 0.180mV detected!
   Timestamp: 1800ms

🚨 ALERT [CRITICAL]: ⚠️ CRITICAL: ST elevation 0.220mV detected! Possible STEMI!
   Timestamp: 1900ms

📊 Status: Beats=25 | HR=76.8 BPM | Abnormal=8 | Max ST=0.240mV | Alerts=6

✅ Stream complete: 1151 samples processed

============================================================
📋 FINAL ANALYSIS SUMMARY
============================================================
Total beats analyzed: 46
Abnormal beats detected: 12 (26.1%)
Maximum ST elevation: 0.240 mV
Average heart rate: 76.5 BPM
Total alerts generated: 8

🚨 ALERTS:
  [HIGH] ⚠️ HIGH: Significant ST elevation 0.180mV detected!
  [CRITICAL] ⚠️ CRITICAL: ST elevation 0.220mV detected! Possible STEMI!
  [CRITICAL] ⚠️ CRITICAL: ST elevation 0.240mV detected! Possible STEMI!
  ...
============================================================
```

## Requirements

```bash
pip install numpy pandas matplotlib scipy scikit-learn tensorflow
```

## Troubleshooting

### "No model found" Warning

If you see this warning, train a model first:

```bash
python predict_heart_problems.py --input ecg_denoised.csv --train
```

### Plotting Not Working

If plots don't appear:
- Try `--no-plot` flag for console-only mode
- Check if matplotlib backend is configured correctly
- On some systems, you may need: `export MPLBACKEND=TkAgg`

### Slow Performance

- Use `--speed 2.0` or higher for faster simulation
- Use `--no-plot` to disable plotting (faster)
- Reduce buffer size in the code if needed

## Integration with Real Hardware

To use with actual Arduino hardware, you would:

1. Replace `simulate_live_stream()` with actual serial reading
2. Read from `/dev/ttyUSB0` (Linux) or `COM3` (Windows)
3. Parse incoming data packets
4. Feed samples directly to `analyzer.buffer.add_sample()`

Example integration:

```python
import serial

ser = serial.Serial('/dev/ttyUSB0', 9600)
while True:
    line = ser.readline().decode().strip()
    timestamp = int(time.time() * 1000)
    value = float(line)
    analyzer.buffer.add_sample(timestamp, value)
    analyzer.analyze_new_peaks()
```

## Next Steps

- **For batch analysis**: Use `predict_heart_problems.py`
- **For visualization**: Use `visualize.py` or `finervisualize.py`
- **For ST analysis**: Use `st_elevation_analysis.py`

## Notes

- The script simulates real-time streaming but processes data as fast as possible
- Alerts are generated immediately when conditions are detected
- The buffer maintains the last 10 seconds of data for analysis
- R-peaks are detected incrementally as new data arrives


