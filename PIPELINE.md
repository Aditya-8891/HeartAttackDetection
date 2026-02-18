# Heart Attack Detection System - Processing Pipeline

## Overview

This document outlines the complete data processing pipeline for the Heart Attack Detection System, from raw ECG signal acquisition via Arduino hardware through real-time analysis and user alerts.

---

## 🔄 End-to-End Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HARDWARE LAYER (Arduino)                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  6-Lead ECG Electrodes → Signal Conditioning & ADC      │  │
│  │  Sampling Rate: ~250Hz (configurable)                   │  │
│  │  Resolution: 12-16 bit                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────┬──────────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│                  TRANSMISSION LAYER                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Bluetooth Module (HC-05/HC-06)                         │  │
│  │  Protocol: Wireless Serial Communication               │  │
│  │  Baud Rate: 9600 bps                                   │  │
│  │  Data Format: [Lead1, Lead2, ..., Lead6] + Checksum    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────┬──────────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│                  MOBILE APP / SERVER LAYER                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Data Reception & Buffering                             │  │
│  │  - Bluetooth listener (real-time)                       │  │
│  │  - Data validation & error checking                     │  │
│  │  - Circular buffer (5-10 second window)                 │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────┬──────────────────────────────────────┘
                          │
                          ↓
        ┌─────────────────────────────────────┐
        │    DATA PREPROCESSING PIPELINE       │
        └─────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ↓                 ↓                 ↓
   ┌─────────┐    ┌─────────────┐    ┌──────────────┐
   │ Noise   │    │ Baseline    │    │ Normalization│
   │Filtering│    │ Drift       │    │  & Scaling   │
   │(Bandpass)   │ Correction  │    │              │
   └────┬────┘    └──────┬──────┘    └──────┬───────┘
        └─────────────────┼──────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│              FEATURE EXTRACTION LAYER                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Statistical Features:                                   │  │
│  │  - RR Interval, QT Interval, ST Segment                │  │
│  │  - Heart Rate, Heart Rate Variability (HRV)            │  │
│  │  - Peak Detection (P, Q, R, S, T waves)                │  │
│  │  - Morphological Features                              │  │
│  │                                                          │  │
│  │  Frequency Domain:                                      │  │
│  │  - FFT Analysis, Power Spectral Density                │  │
│  │  - Wavelet Transform Coefficients                      │  │
│  │  - Frequency Bands (VLF, LF, HF)                       │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────┬──────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ↓                 ↓                 ↓
┌─────────────────┐ ┌─────────────────┐ ┌──────────────┐
│  CNN Model      │ │  LSTM Model     │ │  RNN Model   │
│ (Spatial Patterns)│(Temporal Seq)   │ (Sequential) │
│ Models/CNN/     │ │ Models/LSTM/    │ │Models/RNN/   │
│ test.py         │ │ test.py         │ │test.py       │
└────────┬────────┘ └────────┬────────┘ └──────┬───────┘
         │                   │                 │
         └───────────────────┼─────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│              ENSEMBLE & DECISION LAYER                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Model Fusion:                                           │  │
│  │  - Weighted averaging of model outputs                   │  │
│  │  - Consensus decision making                            │  │
│  │  - Confidence scoring (0-1 range)                        │  │
│  │  - Anomaly flagging threshold: 0.75+                     │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────┬──────────────────────────────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        ↓                                   ↓
    ┌────────┐                        ┌─────────┐
    │ Normal │                        │Anomaly  │
    │ State  │                        │Detected │
    └────┬───┘                        └────┬────┘
         │                                 │
         │                                 ↓
         │                        ┌──────────────────┐
         │                        │ Alert Generation │
         │                        │ - Severity Level │
         │                        │ - Event Type     │
         │                        │ - Timestamp      │
         │                        └────────┬─────────┘
         │                                 │
         └────────────────┬────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│              APPLICATION LAYER                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  ✓ Live Tracking Dashboard                              │  │
│  │    - Real-time ECG visualization                        │  │
│  │    - Heart rate display                                 │  │
│  │    - Live alert notifications                           │  │
│  │                                                          │  │
│  │  ✓ Recorded Tracking Module                             │  │
│  │    - Historical data storage                            │  │
│  │    - Event replay and analysis                          │  │
│  │    - Trend analysis                                     │  │
│  │                                                          │  │
│  │  ✓ Emergency Features                                   │  │
│  │    - Quick contact to health professionals              │  │
│  │    - Emergency call integration                         │  │
│  │    - Location sharing                                   │  │
│  │                                                          │  │
│  │  ✓ Device Management                                    │  │
│  │    - Battery status monitoring                          │  │
│  │    - Bluetooth connectivity                             │  │
│  │    - Electrode contact quality                          │  │
│  │    - System diagnostics                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Detailed Pipeline Stages

### Stage 1: Signal Acquisition (Hardware Layer)

**Location**: Arduino Microcontroller

```
Input:  6 analog signals from ECG electrodes
        ↓
Processing:
- Multiplexing 6-lead inputs
- 12-16 bit ADC conversion
- Real-time sampling @250Hz (configurable)
- Optional: On-board filtering/amplification
        ↓
Output: Digital ECG samples → Bluetooth TX
```

**Data Format**:
```
[Byte0-1: Lead1] [Byte2-3: Lead2] ... [Byte10-11: Lead6] [Checksum]
Total: 13 bytes per sample @ 250Hz = 3.25 KB/s
```

---

### Stage 2: Wireless Transmission

**Protocol**: Bluetooth Serial (HC-05/HC-06)
- **Baud Rate**: 9600 bps
- **Latency**: ~50-100ms
- **Reliability**: Error checking via checksum
- **Range**: ~10m line-of-sight

**Error Handling**:
```python
def validate_packet(packet):
    expected_checksum = sum(packet[:-1]) % 256
    received_checksum = packet[-1]
    return expected_checksum == received_checksum
```

---

### Stage 3: Data Reception & Buffering

**Component**: Mobile App / Backend Server

```python
class ECGDataBuffer:
    def __init__(self, window_size=5, sampling_rate=250):
        self.window_size = window_size  # seconds
        self.sampling_rate = sampling_rate
        self.buffer_size = window_size * sampling_rate
        self.data = np.zeros((self.buffer_size, 6))  # 6 leads
        self.timestamps = []
    
    def add_sample(self, sample):
        # Shift buffer and add new sample
        self.data = np.roll(self.data, -1, axis=0)
        self.data[-1] = sample
        self.timestamps.append(time.time())
    
    def is_ready(self):
        return len(self.timestamps) >= self.buffer_size
```

---

### Stage 4: Preprocessing

#### 4.1 Noise Filtering
- **Type**: Bandpass IIR filter
- **Frequency Range**: 0.5 - 100 Hz
- **Order**: 4th order Butterworth
- **Purpose**: Remove DC offset, 50/60Hz powerline noise, high-frequency artifacts

```python
from scipy import signal

def bandpass_filter(data, lowcut=0.5, highcut=100, fs=250, order=4):
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)
```

#### 4.2 Baseline Drift Correction
- **Method**: Polynomial fitting + subtraction
- **Order**: 2-3 degree polynomial
- **Purpose**: Remove low-frequency baseline wandering

```python
def remove_baseline_drift(data, poly_order=3):
    x = np.arange(len(data))
    coeffs = np.polyfit(x, data, poly_order)
    baseline = np.polyval(coeffs, x)
    return data - baseline
```

#### 4.3 Normalization & Scaling
- **Z-score normalization**: (x - mean) / std
- **Min-Max scaling**: (x - min) / (max - min)
- **Per-lead normalization**: Separate for each ECG lead

---

### Stage 5: Feature Extraction

#### 5.1 Time-Domain Features

```python
class ECGFeatureExtractor:
    def __init__(self, sampling_rate=250):
        self.fs = sampling_rate
    
    def extract_rr_intervals(self, ecg_signal):
        """Extract R-R intervals (time between heartbeats)"""
        # Peak detection on R waves
        r_peaks = self.find_r_peaks(ecg_signal)
        rr_intervals = np.diff(r_peaks) / self.fs  # Convert to seconds
        return rr_intervals
    
    def calculate_heart_rate(self, rr_intervals):
        """Calculate HR from RR intervals"""
        return 60.0 / np.mean(rr_intervals)
    
    def calculate_hrv(self, rr_intervals):
        """Heart Rate Variability metrics"""
        return {
            'sdnn': np.std(rr_intervals),
            'pnn50': self.pnn50(rr_intervals),
            'rmssd': np.sqrt(np.mean(np.diff(rr_intervals)**2))
        }
    
    def extract_wave_parameters(self, ecg_signal):
        """Extract P, Q, R, S, T wave characteristics"""
        return {
            'p_amplitude': None,
            'qrs_duration': None,
            'st_segment': None,
            't_amplitude': None,
            'qt_interval': None
        }
    
    def find_r_peaks(self, signal):
        """R peak detection (e.g., Pan-Tompkins algorithm)"""
        # Implementation of R peak detection
        pass
    
    def analyze_st_segment_elevation(self, ecg_signal, lead_name='Lead_II'):
        """
        Analyze ST segment elevation - Critical indicator of STEMI (ST-Elevation MI)
        
        ST segment is measured from the J-point (end of QRS complex) to the start of T wave.
        Normal baseline: Should return to isoelectric line (0mV reference)
        Elevation: >1mm (0.1mV) in contiguous leads is significant
        Depression: <0 indicates other cardiac conditions
        """
        
        # Step 1: Detect QRS complex boundaries
        r_peaks = self.find_r_peaks(ecg_signal)
        q_peaks = self.find_q_peaks(ecg_signal, r_peaks)
        s_peaks = self.find_s_peaks(ecg_signal, r_peaks)
        
        st_measurements = []
        st_elevations = []
        
        # For each heartbeat, measure ST segment
        for idx, r_peak in enumerate(r_peaks[:-1]):
            # Define window: from S-wave to T-wave
            s_peak = s_peaks[idx]
            next_t_wave_start = r_peaks[idx + 1] - int(0.2 * self.fs)  # Look 200ms before next R
            
            # J-point (junction between QRS and ST segment)
            j_point_idx = s_peak + int(0.04 * self.fs)  # ~40ms after S peak
            
            if j_point_idx >= len(ecg_signal) or next_t_wave_start >= len(ecg_signal):
                continue
            
            # Extract ST segment (40-120ms window after J-point)
            st_window_samples = int(0.08 * self.fs)  # 80ms window
            st_end_idx = min(j_point_idx + st_window_samples, next_t_wave_start)
            
            st_segment = ecg_signal[j_point_idx:st_end_idx]
            
            # Calculate ST level (mean of segment)
            st_level = np.mean(st_segment)
            
            # Calculate elevation relative to PR interval baseline
            pr_baseline = self.calculate_pr_baseline(ecg_signal, r_peaks[idx])
            st_elevation = st_level - pr_baseline
            
            st_measurements.append({
                'beat_number': idx,
                'j_point_idx': j_point_idx,
                'st_level': st_level,
                'pr_baseline': pr_baseline,
                'st_elevation': st_elevation,
                'timestamp': j_point_idx / self.fs
            })
            
            st_elevations.append(st_elevation)
        
        # Step 2: Analyze ST elevation patterns
        if st_elevations:
            mean_st_elevation = np.mean(st_elevations)
            max_st_elevation = np.max(st_elevations)
            std_st_elevation = np.std(st_elevations)
            
            # Detect significant elevation
            elevation_threshold = 0.1  # 1mm = 0.1mV
            significant_elevations = sum(1 for e in st_elevations if e > elevation_threshold)
            elevation_percentage = (significant_elevations / len(st_elevations)) * 100
            
            analysis_result = {
                'lead': lead_name,
                'mean_st_elevation': mean_st_elevation,
                'max_st_elevation': max_st_elevation,
                'std_st_elevation': std_st_elevation,
                'num_beats_analyzed': len(st_elevations),
                'num_significant_elevations': significant_elevations,
                'elevation_percentage': elevation_percentage,
                'is_stemi_suspect': elevation_percentage > 50 and max_st_elevation > elevation_threshold,
                'detailed_measurements': st_measurements
            }
            
            return analysis_result
        else:
            return None
    
    def calculate_pr_baseline(self, ecg_signal, r_peak_idx):
        """
        Calculate PR interval baseline (isoelectric line)
        PR interval is from end of P wave to start of QRS
        Used as reference for ST segment measurement
        """
        # Typically 100-200ms before R peak
        pr_start = max(0, r_peak_idx - int(0.2 * self.fs))
        pr_end = r_peak_idx - int(0.04 * self.fs)  # 40ms before R
        
        if pr_end <= pr_start:
            return 0
        
        pr_segment = ecg_signal[pr_start:pr_end]
        return np.mean(pr_segment)
    
    def find_q_peaks(self, signal, r_peaks, window=int(0.04 * 250)):
        """Find Q wave peaks (negative deflection before R)"""
        q_peaks = []
        for r_peak in r_peaks:
            search_start = max(0, r_peak - window)
            search_end = r_peak
            if search_end > search_start:
                q_idx = search_start + np.argmin(signal[search_start:search_end])
                q_peaks.append(q_idx)
        return q_peaks
    
    def find_s_peaks(self, signal, r_peaks, window=int(0.04 * 250)):
        """Find S wave peaks (negative deflection after R)"""
        s_peaks = []
        for r_peak in r_peaks:
            search_start = r_peak
            search_end = min(len(signal), r_peak + window)
            if search_end > search_start:
                s_idx = search_start + np.argmin(signal[search_start:search_end])
                s_peaks.append(s_idx)
        return s_peaks
    
    def multi_lead_st_analysis(self, ecg_data_6leads):
        """
        Analyze ST segment across all 6 leads for comprehensive STEMI detection
        
        STEMI diagnostic criteria:
        - ST elevation in ≥2 contiguous leads
        - ST elevation >1mm in limb leads or >2mm in precordial leads
        - New LBBB (Left Bundle Branch Block)
        
        ecg_data_6leads: (num_samples, 6) array with 6 leads
        """
        
        # Standard lead mapping
        lead_names = [
            'Lead_I', 'Lead_II', 'Lead_III',
            'Lead_aVL', 'Lead_aVF', 'Lead_aVR'
        ]
        
        all_lead_analysis = {}
        elevated_leads = []
        
        for lead_idx, lead_name in enumerate(lead_names):
            lead_signal = ecg_data_6leads[:, lead_idx]
            
            st_analysis = self.analyze_st_segment_elevation(lead_signal, lead_name)
            
            if st_analysis:
                all_lead_analysis[lead_name] = st_analysis
                
                # Check for significant elevation
                if st_analysis['is_stemi_suspect']:
                    elevated_leads.append({
                        'lead': lead_name,
                        'elevation': st_analysis['max_st_elevation'],
                        'elevation_percentage': st_analysis['elevation_percentage']
                    })
        
        # Apply STEMI criteria
        stemi_risk = self.assess_stemi_risk(all_lead_analysis, elevated_leads)
        
        return {
            'all_leads': all_lead_analysis,
            'elevated_leads': elevated_leads,
            'stemi_assessment': stemi_risk
        }
    
    def assess_stemi_risk(self, all_lead_analysis, elevated_leads):
        """
        Assess STEMI (ST-Elevation Myocardial Infarction) risk based on multi-lead analysis
        """
        
        stemi_indicators = {
            'num_leads_with_elevation': len(elevated_leads),
            'contiguous_leads_affected': self.check_contiguous_leads(elevated_leads),
            'reciprocal_changes': False,  # Check for ST depression in opposite leads
            'risk_level': 'NORMAL'
        }
        
        # Risk stratification
        if stemi_indicators['num_leads_with_elevation'] >= 2:
            if stemi_indicators['contiguous_leads_affected']:
                stemi_indicators['risk_level'] = 'CRITICAL_STEMI'
            else:
                stemi_indicators['risk_level'] = 'HIGH_MI_RISK'
        elif stemi_indicators['num_leads_with_elevation'] == 1:
            stemi_indicators['risk_level'] = 'MODERATE_RISK'
        else:
            stemi_indicators['risk_level'] = 'LOW_RISK'
        
        return stemi_indicators
    
    def check_contiguous_leads(self, elevated_leads):
        """
        Check if elevated leads are contiguous (adjacent in cardiac anatomy)
        STEMI typically shows contiguous lead involvement
        """
        
        # Lead anatomical relationships
        lead_sequences = [
            ['Lead_II', 'Lead_III', 'Lead_aVF'],      # Inferior
            ['Lead_I', 'Lead_aVL'],                    # Lateral-High
            ['Lead_aVL', 'Lead_I', 'Lead_aVR'],       # Anterior
        ]
        
        elevated_lead_names = [e['lead'] for e in elevated_leads]
        
        for sequence in lead_sequences:
            contiguous_count = sum(1 for lead in sequence if lead in elevated_lead_names)
            if contiguous_count >= 2:
                return True
        
        return False
```

#### 5.2 Frequency-Domain Features

```python
def extract_frequency_features(signal, fs=250):
    """FFT and spectral analysis"""
    fft_vals = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), 1/fs)
    
    # Power spectral density
    psd = np.abs(fft_vals)**2
    
    # Frequency band analysis (HRV)
    vlf = (0.0033, 0.04)    # Very Low Frequency
    lf = (0.04, 0.15)       # Low Frequency
    hf = (0.15, 0.4)        # High Frequency
    
    return {
        'vlf_power': band_power(psd, vlf),
        'lf_power': band_power(psd, lf),
        'hf_power': band_power(psd, hf),
        'lf_hf_ratio': band_power(psd, lf) / band_power(psd, hf)
    }
```

#### 5.3 ST Segment Elevation Analysis (STEMI Detection)

**Critical for Heart Attack Detection**

```python
def analyze_st_segment_elevation_summary(ecg_data_6leads, sampling_rate=250):
    """
    ST Segment Elevation Analysis - PRIMARY indicator of STEMI
    
    STEMI (ST-Elevation Myocardial Infarction) Diagnostic Criteria:
    1. ST elevation ≥1mm in ≥2 contiguous leads
    2. Reciprocal ST depression in opposite leads
    3. Elevated cardiac biomarkers (troponin, CK-MB)
    4. New LBBB pattern
    
    ST Segment Location in QRS-T Complex:
    ├─ QRS Complex: 60-100ms
    ├─ J-Point: End of S wave (junction between QRS and ST)
    ├─ ST Segment: From J-point to T wave onset (~40-120ms)
    └─ T Wave: Repolarization phase
    
    Measurement:
    - Baseline: PR interval (isoelectric line)
    - Elevation: Vertical displacement of ST segment from baseline
    - Threshold: >0.1mV (1mm on ECG paper @ 10mm/mV)
    """
    
    extractor = ECGFeatureExtractor(sampling_rate)
    
    # Analyze all 6 leads
    st_analysis = extractor.multi_lead_st_analysis(ecg_data_6leads)
    
    return {
        'st_measurements': st_analysis,
        'critical_indicator': True,
        'requires_immediate_analysis': True
    }
```

#### 5.4 Wavelet Transform Features

```python
def extract_wavelet_features(signal, wavelet='db4', level=5):
    """Wavelet decomposition for multi-scale analysis"""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    features = {}
    for i, coeff in enumerate(coeffs):
        features[f'detail_{i}'] = {
            'energy': np.sum(coeff**2),
            'entropy': entropy(coeff),
            'std': np.std(coeff)
        }
    
    return features
```

---

### Stage 6: Model Inference

#### 6.1 CNN Model (Spatial Pattern Recognition)

```python
# models/cnn/test.py
class CNNModel:
    """
    Convolutional Neural Network for ECG pattern recognition
    - Input: 6-lead ECG signal (1250 samples @ 250Hz = 5 seconds)
    - Output: Classification probability [0-1]
    - Detects: Morphological anomalies (ST elevation, T wave inversion, etc.)
    """
    
    def __init__(self, model_path):
        self.model = load_model(model_path)
    
    def predict(self, ecg_data):
        # Reshape: (1250, 6) → (1, 1250, 6, 1) for CNN
        x = ecg_data.reshape(1, -1, 6, 1)
        return self.model.predict(x)
```

#### 6.2 LSTM Model (Temporal Sequence Analysis)

```python
# models/lstm/test.py
class LSTMModel:
    """
    LSTM for temporal ECG pattern analysis
    - Input: 6-lead ECG sequence
    - Output: Classification probability [0-1]
    - Detects: Time-dependent arrhythmias, rhythm changes
    """
    
    def __init__(self, model_path):
        self.model = load_model(model_path)
    
    def predict(self, ecg_data, lookback=100):
        # Create sequences for LSTM
        x = self.create_sequences(ecg_data, lookback)
        return self.model.predict(x)
```

#### 6.3 RNN Model (Sequential Pattern Recognition)

```python
# models/rnn/test.py
class RNNModel:
    """
    RNN for sequential ECG analysis
    - Input: 6-lead ECG sequences
    - Output: Classification probability [0-1]
    - Detects: Progressive cardiac events
    """
    
    def __init__(self, model_path):
        self.model = load_model(model_path)
    
    def predict(self, ecg_data):
        return self.model.predict(ecg_data)
```

---

### Stage 7: Ensemble & Decision Making

```python
class EnsembleDecisionEngine:
    def __init__(self, cnn_model, lstm_model, rnn_model):
        self.models = {
            'cnn': {'model': cnn_model, 'weight': 0.4},
            'lstm': {'model': lstm_model, 'weight': 0.35},
            'rnn': {'model': rnn_model, 'weight': 0.25}
        }
        self.anomaly_threshold = 0.75
        self.feature_extractor = ECGFeatureExtractor()
    
    def predict(self, features, ecg_data, ecg_data_6leads=None):
        predictions = {}
        
        # Get individual model predictions
        for name, model_info in self.models.items():
            pred = model_info['model'].predict(ecg_data)
            predictions[name] = pred[0][0]  # Assuming binary classification
        
        # Weighted ensemble
        weighted_score = sum(
            predictions[name] * model_info['weight']
            for name, model_info in self.models.items()
        )
        
        # **CRITICAL: ST Segment Elevation Analysis**
        st_elevation_risk = 0.0
        if ecg_data_6leads is not None:
            st_analysis = self.feature_extractor.multi_lead_st_analysis(ecg_data_6leads)
            st_elevation_risk = self.calculate_st_risk_score(st_analysis)
        
        # Boost score if STEMI indicators detected
        if st_elevation_risk > 0.5:
            weighted_score = min(1.0, weighted_score * 1.3 + st_elevation_risk * 0.3)
        
        # Decision
        is_anomaly = weighted_score >= self.anomaly_threshold
        confidence = weighted_score
        
        return {
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'individual_scores': predictions,
            'weighted_score': weighted_score,
            'st_elevation_risk': st_elevation_risk,
            'st_analysis': st_analysis if ecg_data_6leads is not None else None
        }
    
    def calculate_st_risk_score(self, st_analysis):
        """
        Convert ST elevation analysis into risk probability score (0-1)
        STEMI risk escalation:
        - CRITICAL_STEMI: 0.95
        - HIGH_MI_RISK: 0.85
        - MODERATE_RISK: 0.70
        - LOW_RISK: 0.10
        """
        stemi_assessment = st_analysis.get('stemi_assessment', {})
        risk_level = stemi_assessment.get('risk_level', 'LOW_RISK')
        
        risk_mapping = {
            'CRITICAL_STEMI': 0.95,
            'HIGH_MI_RISK': 0.85,
            'MODERATE_RISK': 0.70,
            'LOW_RISK': 0.10,
            'NORMAL': 0.05
        }
        
        return risk_mapping.get(risk_level, 0.05)
```

---

### Stage 8: Alert Generation & Severity Assessment

```python
class AlertSystem:
    def __init__(self):
        self.severity_levels = {
            'CRITICAL': 0.9,      # Immediate emergency
            'HIGH': 0.80,         # Urgent medical attention
            'MEDIUM': 0.75,       # Close monitoring recommended
            'LOW': 0.65           # Consultation advised
        }
    
    def generate_alert(self, decision, ecg_metadata):
        if not decision['is_anomaly']:
            return None
        
        confidence = decision['confidence']
        st_elevation_risk = decision.get('st_elevation_risk', 0.0)
        st_analysis = decision.get('st_analysis', None)
        
        # Determine severity - ST elevation is CRITICAL indicator
        severity = self.get_severity(confidence, st_elevation_risk, st_analysis)
        
        # Classify event type based on ST analysis
        event_type = self.classify_event_from_st_analysis(st_analysis, decision)
        
        alert = {
            'timestamp': datetime.now(),
            'event_type': event_type,
            'severity': severity,
            'confidence': confidence,
            'st_elevation_risk': st_elevation_risk,
            'st_analysis': st_analysis,
            'heart_rate': ecg_metadata['heart_rate'],
            'location': ecg_metadata.get('location', None),
            'recommended_action': self.get_recommendation(severity, event_type),
            'clinical_notes': self.generate_clinical_notes(st_analysis, decision)
        }
        
        return alert
    
    def get_severity(self, confidence, st_elevation_risk, st_analysis):
        """
        Determine severity with ST elevation as primary driver
        ST elevation indicating STEMI is always CRITICAL
        """
        
        # CRITICAL: ST elevation detected
        if st_analysis:
            stemi_assessment = st_analysis.get('stemi_assessment', {})
            risk_level = stemi_assessment.get('risk_level', 'LOW_RISK')
            
            if risk_level == 'CRITICAL_STEMI':
                return 'CRITICAL'
            elif risk_level == 'HIGH_MI_RISK':
                return 'HIGH'
            elif risk_level == 'MODERATE_RISK':
                return 'MEDIUM'
        
        # Fallback to confidence-based severity
        for severity, threshold in self.severity_levels.items():
            if confidence >= threshold:
                return severity
        return 'LOW'
    
    def classify_event_from_st_analysis(self, st_analysis, decision):
        """Classify cardiac event based on ST segment and ML models"""
        
        if st_analysis:
            stemi_assessment = st_analysis.get('stemi_assessment', {})
            risk_level = stemi_assessment.get('risk_level', 'LOW_RISK')
            
            if risk_level == 'CRITICAL_STEMI':
                return 'STEMI_DETECTED'
            elif risk_level == 'HIGH_MI_RISK':
                return 'ACUTE_MI_RISK'
            elif risk_level == 'MODERATE_RISK':
                return 'CARDIAC_ISCHEMIA'
        
        return 'CARDIAC_ANOMALY'
    
    def generate_clinical_notes(self, st_analysis, decision):
        """Generate detailed clinical notes for healthcare professionals"""
        
        notes = []
        
        if st_analysis:
            elevated_leads = st_analysis.get('elevated_leads', [])
            stemi_assessment = st_analysis.get('stemi_assessment', {})
            
            if elevated_leads:
                leads_str = ', '.join([e['lead'] for e in elevated_leads])
                notes.append(f"ST elevation detected in leads: {leads_str}")
            
            num_leads = stemi_assessment.get('num_leads_with_elevation', 0)
            if num_leads >= 2:
                notes.append(f"Multi-lead ST elevation (n={num_leads}) - STEMI criteria met")
            
            if stemi_assessment.get('contiguous_leads_affected'):
                notes.append("ST elevation in contiguous leads - suggests regional MI")
        
        model_scores = decision.get('individual_scores', {})
        if model_scores:
            notes.append(f"ML Model ensemble scores: CNN={model_scores.get('cnn', 0):.2f}, "
                        f"LSTM={model_scores.get('lstm', 0):.2f}, RNN={model_scores.get('rnn', 0):.2f}")
        
        return notes
    
    def get_recommendation(self, severity, event_type):
        """Generate clinical recommendations"""
        
        recommendations = {
            'CRITICAL': {
                'STEMI_DETECTED': 'ACTIVATE EMERGENCY PROTOCOL - CALL 911 IMMEDIATELY. Prepare for PCI/Thrombolytic therapy.',
                'ACUTE_MI_RISK': 'CALL EMERGENCY SERVICES IMMEDIATELY. Likely acute myocardial infarction.',
                'CARDIAC_ANOMALY': 'CALL EMERGENCY SERVICES IMMEDIATELY. Critical cardiac condition detected.'
            },
            'HIGH': {
                'STEMI_DETECTED': 'URGENT: Contact emergency services. ST elevation MI suspected.',
                'ACUTE_MI_RISK': 'URGENT: Seek immediate hospital care. Possible acute MI.',
                'CARDIAC_ANOMALY': 'Contact emergency services urgently. Significant cardiac abnormality.'
            },
            'MEDIUM': {
                'STEMI_DETECTED': 'Contact hospital urgently. ST changes detected.',
                'ACUTE_MI_RISK': 'Contact healthcare provider urgently. MI risk detected.',
                'CARDIAC_ANOMALY': 'Contact healthcare provider. Cardiac anomaly requires evaluation.'
            },
            'LOW': {
                'STEMI_DETECTED': 'Schedule medical consultation. Monitor closely.',
                'ACUTE_MI_RISK': 'Schedule healthcare provider consultation.',
                'CARDIAC_ANOMALY': 'Schedule medical consultation.'
            }
        }
        
        return recommendations.get(severity, {}).get(
            event_type,
            f'Contact healthcare provider for {severity.lower()} priority evaluation.'
        )
```

---

### Stage 9: Application Layer (User Interface)

#### 9.1 Live Tracking Dashboard

```python
class LiveTrackingDashboard:
    def __init__(self):
        self.update_rate = 1000  # ms
        self.ecg_buffer = ECGDataBuffer(window_size=10)
    
    def display_realtime_ecg(self):
        """Display live ECG waveform"""
        # Matplotlib / Plotly real-time visualization
        pass
    
    def display_heart_rate(self):
        """Show current HR and trend"""
        pass
    
    def display_alerts(self):
        """Show active alerts and notifications"""
        pass
```

#### 9.2 Recorded Tracking Module

```python
class RecordedTrackingModule:
    def __init__(self, db_path):
        self.db = ECGDatabase(db_path)
    
    def save_recording(self, ecg_data, metadata):
        """Store ECG recording and analysis results"""
        self.db.insert_recording({
            'timestamp': metadata['timestamp'],
            'duration': metadata['duration'],
            'heart_rate': metadata['heart_rate'],
            'alerts': metadata['alerts'],
            'raw_data': ecg_data,
            'filename': f"ecg_{metadata['timestamp']}.csv"
        })
    
    def retrieve_historical_data(self, start_date, end_date):
        """Fetch recordings for date range"""
        return self.db.query(start_date, end_date)
    
    def generate_report(self, recording_id):
        """Create detailed analysis report"""
        pass
```

#### 9.3 Emergency Contact System

```python
class EmergencyContactManager:
    def __init__(self):
        self.emergency_contacts = []
        self.health_professionals = []
    
    def trigger_emergency_alert(self, alert_data):
        """Send alert to emergency contacts"""
        for contact in self.emergency_contacts:
            self.send_notification(contact, alert_data)
            self.share_location(contact)
        
        # Optional: Call emergency services
        self.initiate_emergency_call()
    
    def notify_health_professional(self, alert_data):
        """Alert assigned healthcare provider"""
        for provider in self.health_professionals:
            self.send_detailed_report(provider, alert_data)
```

#### 9.4 Device Management

```python
class DeviceManager:
    def __init__(self):
        self.battery_level = 100
        self.signal_quality = 0.95
    
    def monitor_battery(self):
        """Check battery status"""
        if self.battery_level < 20:
            self.send_low_battery_alert()
    
    def check_bluetooth_connection(self):
        """Verify device connectivity"""
        return self.is_connected()
    
    def verify_electrode_contact(self):
        """Assess electrode quality"""
        # Based on signal impedance
        pass
    
    def run_system_diagnostics(self):
        """Comprehensive device health check"""
        return {
            'battery': self.battery_level,
            'signal_quality': self.signal_quality,
            'bluetooth': self.is_connected(),
            'electrode_contact': self.verify_electrode_contact()
        }
```

---

## 📈 Data Storage & Management

### Database Schema

```sql
-- ECG Recordings
CREATE TABLE recordings (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    timestamp DATETIME,
    duration_seconds INTEGER,
    sampling_rate INTEGER,
    num_leads INTEGER,
    raw_data BLOB,
    preprocessed_data BLOB
);

-- Analysis Results
CREATE TABLE analysis_results (
    id INTEGER PRIMARY KEY,
    recording_id INTEGER,
    model_name VARCHAR,
    prediction FLOAT,
    confidence FLOAT,
    features JSON,
    FOREIGN KEY (recording_id) REFERENCES recordings(id)
);

-- Alerts & Events
CREATE TABLE alerts (
    id INTEGER PRIMARY KEY,
    recording_id INTEGER,
    timestamp DATETIME,
    event_type VARCHAR,
    severity VARCHAR,
    description TEXT,
    user_notified BOOLEAN,
    FOREIGN KEY (recording_id) REFERENCES recordings(id)
);

-- Device Status
CREATE TABLE device_status (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    timestamp DATETIME,
    battery_level INTEGER,
    signal_quality FLOAT,
    bluetooth_connected BOOLEAN,
    electrode_contact_quality FLOAT
);
```

---

## 🔧 Implementation Roadmap

### Phase 1: Signal Acquisition & Preprocessing
- [ ] Arduino firmware development
- [ ] Bluetooth communication protocol
- [ ] Data reception & buffering
- [ ] Noise filtering & normalization

### Phase 2: Feature Extraction
- [ ] Time-domain feature extraction
- [ ] Frequency-domain analysis
- [ ] **ST Segment Elevation Analysis** (Critical for STEMI detection)
- [ ] Wavelet transformation
- [ ] Signal quality assessment
- [ ] Multi-lead ST elevation assessment

### Phase 3: Model Development
- [ ] CNN architecture design & training
- [ ] LSTM model implementation
- [ ] RNN model development
- [ ] Model validation & testing

### Phase 4: Ensemble & Decision Engine
- [ ] Ensemble architecture
- [ ] Weighted voting system
- [ ] Alert generation logic
- [ ] Severity classification

### Phase 5: Application Development
- [ ] Live tracking dashboard
- [ ] Recording analysis module
- [ ] Emergency contact system
- [ ] Device management interface

### Phase 6: Integration & Testing
- [ ] End-to-end pipeline testing
- [ ] Real-world validation
- [ ] Performance optimization
- [ ] Security & compliance

---

## ⚡ Performance Optimization

### Latency Targets
- **Data acquisition to preprocessing**: < 100ms
- **Feature extraction**: < 50ms
- **Model inference**: < 100ms (ensemble)
- **Alert generation to notification**: < 200ms
- **Total latency**: < 450ms

### Memory Management
- Circular buffers for continuous streaming
- Lazy loading of heavy models
- Model quantization for mobile deployment
- Efficient numpy operations

### Computation Efficiency
```python
# Use GPU acceleration where possible
import tensorflow as tf

gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
```

---

## 🛡️ Quality Assurance

### Testing Strategy

```python
# Unit Tests
def test_bandpass_filter():
    signal = generate_test_ecg()
    filtered = bandpass_filter(signal)
    assert filtered.shape == signal.shape

# ST Segment Analysis Tests
def test_st_elevation_detection():
    """Test STEMI detection algorithm"""
    # Create synthetic ECG with known ST elevation
    normal_st = 0.0
    elevated_st = 0.12  # 1.2mm elevation
    
    extractor = ECGFeatureExtractor()
    
    # Test 1: Normal baseline
    normal_ecg = generate_normal_ecg()
    result = extractor.analyze_st_segment_elevation(normal_ecg)
    assert result['mean_st_elevation'] < 0.05
    assert not result['is_stemi_suspect']
    
    # Test 2: Simulated ST elevation
    elevated_ecg = inject_st_elevation(normal_ecg, elevated_st)
    result = extractor.analyze_st_segment_elevation(elevated_ecg)
    assert result['mean_st_elevation'] > 0.08
    assert result['is_stemi_suspect']

def test_multi_lead_stemi_criteria():
    """Test multi-lead STEMI diagnostic criteria"""
    ecg_data = load_stemi_reference_data()  # Known STEMI case
    
    extractor = ECGFeatureExtractor()
    st_analysis = extractor.multi_lead_st_analysis(ecg_data)
    
    assert len(st_analysis['elevated_leads']) >= 2
    assert st_analysis['stemi_assessment']['risk_level'] == 'CRITICAL_STEMI'

def test_contiguous_lead_detection():
    """Test detection of contiguous lead involvement"""
    elevated_leads = [
        {'lead': 'Lead_II', 'elevation': 0.15},
        {'lead': 'Lead_III', 'elevation': 0.13},
        {'lead': 'Lead_aVF', 'elevation': 0.12}
    ]
    
    extractor = ECGFeatureExtractor()
    is_contiguous = extractor.check_contiguous_leads(elevated_leads)
    assert is_contiguous  # Inferior MI pattern

# Integration Tests
def test_st_elevation_through_full_pipeline():
    """Test ST elevation detection through entire pipeline"""
    raw_data = load_test_ecg_with_st_elevation()
    
    # Preprocessing
    processed = preprocess_pipeline(raw_data)
    
    # Feature extraction with ST analysis
    extractor = ECGFeatureExtractor()
    st_analysis = extractor.multi_lead_st_analysis(processed)
    
    # Ensemble decision
    decision = ensemble_engine.predict(features, processed, processed)
    
    # Alert generation
    alert = alert_system.generate_alert(decision, metadata)
    
    assert alert is not None
    assert alert['event_type'] == 'STEMI_DETECTED'
    assert alert['severity'] == 'CRITICAL'

# ST Elevation Accuracy Tests
def test_st_elevation_sensitivity_specificity():
    """Validate clinical sensitivity and specificity"""
    
    test_cases = [
        ('normal_ecg.csv', False),
        ('stemi_anterior.csv', True),
        ('stemi_inferior.csv', True),
        ('stemi_lateral.csv', True),
        ('artifact.csv', False),
        ('atrial_fibrillation.csv', False)
    ]
    
    extractor = ECGFeatureExtractor()
    tp = fp = tn = fn = 0
    
    for filename, has_stemi in test_cases:
        ecg_data = load_test_data(filename)
        st_analysis = extractor.multi_lead_st_analysis(ecg_data)
        
        is_stemi = st_analysis['stemi_assessment']['risk_level'] == 'CRITICAL_STEMI'
        
        if is_stemi and has_stemi:
            tp += 1
        elif is_stemi and not has_stemi:
            fp += 1
        elif not is_stemi and not has_stemi:
            tn += 1
        else:
            fn += 1
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    assert sensitivity > 0.90  # Clinical requirement: >90% sensitivity
    assert specificity > 0.95  # Clinical requirement: >95% specificity

# Model Tests
def test_ensemble_prediction_with_st_elevation():
    ecg_data = load_test_data()
    decision = ensemble_engine.predict(features, ecg_data, ecg_data)
    assert 'st_elevation_risk' in decision
    assert 0 <= decision['st_elevation_risk'] <= 1.0
```

---

## 📝 Configuration Files

### Arduino Configuration (`recieve.ino`)
```cpp
#define SAMPLING_RATE 250
#define NUM_LEADS 6
#define ADC_RESOLUTION 12
#define BAUD_RATE 9600
```

### Signal Processing Config
```python
# config.py
PREPROCESSING = {
    'bandpass_low': 0.5,
    'bandpass_high': 100,
    'filter_order': 4,
    'normalize': True
}

ML_MODELS = {
    'cnn': {'path': 'models/cnn/model.h5', 'weight': 0.4},
    'lstm': {'path': 'models/lstm/model.h5', 'weight': 0.35},
    'rnn': {'path': 'models/rnn/model.h5', 'weight': 0.25}
}

ALERT_CONFIG = {
    'anomaly_threshold': 0.75,
    'critical_threshold': 0.9,
    'update_interval': 1  # seconds
}
```

---

## 🔍 Monitoring & Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ecg_pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Usage
logger.info(f"Processing ECG sample - HR: {heart_rate} BPM")
logger.warning(f"Low signal quality: {signal_quality}")
logger.error(f"Model inference failed: {error_message}")
```

---

## 🏥 ST Segment Elevation Analysis - Clinical Guide

### What is ST Segment Elevation?

The ST segment is the portion of the ECG waveform between the S wave (end of QRS complex) and the T wave. Normally, it should return to the isoelectric baseline (0mV reference).

**ST Elevation**: Vertical displacement of the ST segment ABOVE the baseline
- **Pathological significance**: Indicates acute transmural myocardial infarction (STEMI)
- **Diagnostic threshold**: >1mm (0.1mV) in limb leads or >2mm in precordial leads

### STEMI (ST-Elevation Myocardial Infarction) Criteria

The system detects STEMI using the following diagnostic criteria:

#### 1. **Multi-Lead Involvement**
```
Criteria: ≥2 contiguous leads with ST elevation
Significance: Indicates regional ischemia (specific cardiac territory affected)

Anatomical Regions:
├─ Anterior MI: Leads I, aVL, V1-V4
├─ Inferior MI: Leads II, III, aVF
├─ Lateral MI: Leads I, aVL, V5-V6
└─ Posterior MI: V1-V2 depression (reciprocal)
```

#### 2. **Contiguous Lead Pattern**
```
True STEMI shows involvement of ADJACENT leads:
✓ VALID:   Lead_II + Lead_III + aVF (Inferior wall)
✓ VALID:   Lead_I + aVL (Lateral wall)
✗ INVALID: Random lead involvement (artifact/noise)
```

#### 3. **Magnitude Thresholds**
```
ST Elevation Thresholds:
- Limb Leads (I, II, III, aVL, aVF, aVR): >1mm (0.1mV)
- Precordial Leads (V1-V6): >2mm (0.2mV)
- Special: V2-V3 can have >2.5mm in young healthy males (variant)
```

#### 4. **Reciprocal Changes**
```
Reciprocal ST Depression:
- ST depression in leads OPPOSITE to the infarct territory
- Confirms diagnosis (not due to artifact)
- Example: Inferior MI shows ST depression in aVL, I
```

#### 5. **Temporal Evolution**
```
STEMI Timeline (hours to days):
├─ 0-2 hours:   ST elevation begins, T waves peak
├─ 2-12 hours:  Maximum ST elevation, pathological Q waves may appear
├─ 12-48 hours: ST elevation begins to normalize, T wave inversion
└─ 3+ days:     T wave normalization, Q waves persist
```

### Clinical STEMI Classifications by Location

```
1. ANTERIOR MI (LAD Occlusion)
   └─ Leads: I, aVL, V1-V4
   └─ Risk: Large infarct, high mortality
   └─ Recommended Treatment: Primary PCI

2. INFERIOR MI (RCA/LCx Occlusion)
   └─ Leads: II, III, aVF
   └─ Involvement: Right ventricle often affected
   └─ Risk: Bradycardia, heart blocks, shock

3. LATERAL MI (LCx Occlusion)
   └─ Leads: I, aVL, V5-V6
   └─ Often accompanies anterior/inferior MI
   └─ Risk: Medium to high

4. POSTERIOR MI (Rare)
   └─ Pattern: ST depression in V1-V2 (reciprocal)
   └─ Often missed if not specifically looked for
   └─ May need posterior leads (V8-V9) for confirmation
```

### Pipeline Implementation Details

#### ST Elevation Measurement Process

```python
ECG Waveform
    ├─ Identify QRS Complex (R peak, Q and S waves)
    ├─ Locate J-Point (junction: end of S wave to ST segment)
    ├─ Measure ST Segment (40-120ms after J-point)
    ├─ Calculate Baseline (PR interval = isoelectric reference)
    └─ Calculate Elevation = ST Level - PR Baseline

Result: Elevation value for each heartbeat
        Mean elevation across 5-10 beats
        Significance assessment (threshold comparison)
```

#### Multi-Lead Analysis Algorithm

```
Step 1: Process all 6 leads independently
        └─ Measure ST elevation in each lead
        
Step 2: Identify elevated leads
        └─ Threshold: >0.1mV in limb leads
        
Step 3: Check for contiguity
        └─ Verify leads are anatomically adjacent
        
Step 4: Count elevation percentage
        └─ % of heartbeats with significant elevation
        
Step 5: Apply STEMI criteria
        └─ ≥2 contiguous leads elevated? → CRITICAL_STEMI
        └─ 1 lead elevated? → MODERATE_RISK
        └─ No elevation? → NORMAL
```

### Decision Integration with ML Models

```
                    ┌─────────────┐
                    │ ST Analysis │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Risk Level  │
                    │ 0.05 - 0.95 │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ↓                  ↓                  ↓
    [CNN]             [LSTM]             [RNN]
    Spatial         Temporal          Sequential
    0.0 - 1.0       0.0 - 1.0         0.0 - 1.0
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    ┌──────▼──────────┐
                    │ Weighted Ensemble│
                    │ CNN: 0.4 weight  │
                    │ LSTM: 0.35       │
                    │ RNN: 0.25        │
                    │ ST: BOOST factor │
                    └──────┬───────────┘
                           │
                    ┌──────▼────────────┐
                    │ Final Score       │
                    │ (0.0 - 1.0)       │
                    └──────┬────────────┘
                           │
                    ┌──────▼────────────┐
                    │ Threshold Test    │
                    │ Score > 0.75?     │
                    └──────┬────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        ↓                                      ↓
   [ANOMALY]                             [NORMAL]
   Generate Alert                        Continue Monitoring
```

### Clinical Decision Support

```
IF ST Elevation Detected:
├─ Risk Level = CRITICAL_STEMI
├─ Severity = CRITICAL (regardless of ML score)
├─ Alert Type = STEMI_DETECTED
├─ Recommendation = CALL 911 IMMEDIATELY
└─ Actions:
    ├─ Activate emergency protocol
    ├─ Notify emergency contacts
    ├─ Share location with healthcare providers
    ├─ Notify cardiology department
    └─ Prepare for immediate transport to PCI center

TIMING IS CRITICAL: Door-to-balloon time <90 minutes is essential
```

### Validation Against Clinical Standards

```
Clinical Sensitivity: >90%
├─ Correctly identifies >90% of actual STEMIs
├─ Prevents false negatives (missing real infarcts)
└─ Critical for patient safety

Clinical Specificity: >95%
├─ Correctly identifies >95% of non-STEMI cases
├─ Minimizes false alarms
└─ Important for reducing unnecessary hospital visits

Clinical Performance:
├─ Positive Predictive Value (PPV): >85%
├─ Negative Predictive Value (NPV): >95%
└─ Area Under Curve (AUC): >0.95
```

### Edge Cases & Handling

```
1. LBBB (Left Bundle Branch Block)
   └─ ST elevation can be normal variant
   └─ Modified Sgarbossa criteria used
   └─ System flags for manual review

2. LVH (Left Ventricular Hypertrophy)
   └─ Can show ST depression mimicking ischemia
   └─ System cross-checks with other markers

3. Early Repolarization
   └─ Benign ST elevation in young, athletic individuals
   └─ Differentiated by upright T waves, J waves

4. Pericarditis
   └─ Can show diffuse ST elevation (not regional)
   └─ System checks for multi-regional pattern

5. Hyperkalemia
   └─ Peaked T waves, tall QRS
   └─ ST segment may appear elevated
   └─ System alerts for electrolyte assessment
```

---

## 📚 References

- **ST Elevation MI Guidelines**:
  - ACC/AHA STEMI Guidelines (2013)
  - ESC STEMI Guidelines (2017)

- **ECG Analysis**:
  - Pan-Tompkins R Peak Detection Algorithm
  - Wavelet Analysis for ECG Signal Processing
  - Sgarbossa Criteria for LBBB and STEMI

- **Machine Learning**:
  - Deep Learning for Cardiac Monitoring
  - CNN/LSTM/RNN architectures for ECG

- **Technical**:
  - Real-time Signal Processing Techniques
  - Arduino Wireless Communication Protocols
  - Clinical validation methodologies

- **Clinical References**:
  - ECG interpretation textbooks (Marriott's, Dubin's)
  - Acute coronary syndrome management protocols
  - Cardiology consultation standards

---

**This pipeline is designed for real-time cardiac monitoring with medical-grade accuracy requirements. ST Segment Elevation Analysis is the PRIMARY diagnostic marker for acute STEMI and receives priority in the decision-making process.**

⚠️ **DISCLAIMER**: This system is intended for supplementary cardiac monitoring. All alerts must be evaluated by qualified healthcare professionals. In case of chest pain or cardiac symptoms, call emergency services immediately - do not rely solely on this device.
