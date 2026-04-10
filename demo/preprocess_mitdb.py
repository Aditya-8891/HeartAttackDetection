"""
preprocess_mitdb.py

Extract labeled beat windows from MIT-BIH Arrhythmia Database records.

Uses cardiologist-validated R-peak annotations from .atr files — NOT our own
peak detector — so labels are clinically grounded.

Output:
    data/mitdb_processed/X.npy       float32, shape (N, 216)
    data/mitdb_processed/y.npy       int8, shape (N,)
    data/mitdb_processed/metadata.json

Usage:
    python demo/preprocess_mitdb.py
    python demo/preprocess_mitdb.py --raw-dir data/mitdb_raw --output-dir data/mitdb_processed
"""

import argparse
import json
import os
import sys
import numpy as np
from scipy.signal import butter, filtfilt

# Beat window: 600ms at 360Hz = 216 samples
FS = 360
WINDOW_SAMPLES = 216   # 600ms at 360Hz
HALF = WINDOW_SAMPLES // 2  # 108 samples each side

# All MIT-BIH beat annotation symbols that map to label=1 (abnormal)
# Symbol 'N' and '.' (unclassified normal-ish) → label 0
# Everything else → label 1
NORMAL_SYMBOLS = {'N', 'L', 'R', 'e', 'j'}  # L/R are bundle branch blocks but often grouped as "normal" in binary tasks
# For a stricter arrhythmia demo we label only pure normals as 0:
STRICTLY_NORMAL = {'N'}

# Non-beat annotation symbols to skip entirely (rhythm/noise markers)
NON_BEAT_SYMBOLS = {'+', '~', '|', 's', 'T', '*', 'D', '=', '"', '@', 'Q'}

DEFAULT_RECORDS = [
    '100', '101', '103', '105', '106', '108',
    '109', '111', '112', '113', '114', '115',
    '116', '117', '118', '119'
]


def bandpass_filter_360hz(signal):
    """0.5–40 Hz 3rd-order Butterworth bandpass at 360 Hz.

    Mirrors the existing bandpass_filter() in test-demo/predict_heart_problems.py,
    parameterized for 360 Hz.
    """
    nyq = 0.5 * FS
    low = 0.5 / nyq
    high = 40.0 / nyq
    b, a = butter(3, [low, high], btype='band')
    return filtfilt(b, a, signal)


def load_record(record_path):
    """Load channel 0 (MLII) from a MIT-BIH record.

    Returns:
        signal: float64 numpy array (full record length ~650,000 samples)
        fs: sampling frequency (always 360 for MIT-BIH)
    """
    import wfdb
    rec = wfdb.rdrecord(record_path)
    signal = rec.p_signal[:, 0].astype(np.float64)  # MLII lead
    fs = rec.fs
    return signal, fs


def load_annotations(record_path):
    """Load beat annotations from .atr file.

    Returns:
        sample_indices: int array of R-peak positions
        symbols: list of annotation symbols
    """
    import wfdb
    ann = wfdb.rdann(record_path, 'atr')
    return ann.sample, ann.symbol


def label_symbol(symbol):
    """Map annotation symbol to binary label.

    Returns:
        0 for normal beats (symbol 'N')
        1 for abnormal beats (all other beat symbols)
        None to skip (non-beat annotations)
    """
    if symbol in NON_BEAT_SYMBOLS:
        return None  # Skip non-beat markers
    if symbol in STRICTLY_NORMAL:
        return 0
    return 1


def extract_windows(signal, r_samples, symbols):
    """Extract 216-sample windows centered on each R-peak annotation.

    Args:
        signal: filtered 1D signal array
        r_samples: array of R-peak sample indices
        symbols: list of annotation symbols (same length as r_samples)

    Returns:
        windows: list of float64 arrays, each shape (216,)
        labels: list of int (0 or 1)
    """
    windows = []
    labels = []
    n = len(signal)

    for idx, sym in zip(r_samples, symbols):
        label = label_symbol(sym)
        if label is None:
            continue  # Skip non-beat annotations

        start = int(idx) - HALF
        end = int(idx) + HALF

        if start < 0 or end > n:
            continue  # Skip beats too close to record edges

        window = signal[start:end].copy()
        windows.append(window)
        labels.append(label)

    return windows, labels


def normalize_windows(X):
    """Per-row z-score normalization.

    Identical to the normalization in existing predict_heart_problems.py.
    """
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True) + 1e-8
    return (X - means) / stds


def process_all_records(records, raw_dir):
    """Run the full preprocessing pipeline over all records.

    Returns:
        X: float32 array, shape (total_beats, 216)
        y: int8 array, shape (total_beats,)
    """
    all_windows = []
    all_labels = []
    total_normal = 0
    total_abnormal = 0
    skipped_records = []

    for rec in records:
        rec_path = os.path.join(raw_dir, rec)
        # Check files exist
        if not all(os.path.exists(rec_path + ext) for ext in ['.dat', '.hea', '.atr']):
            print(f"  [SKIP] {rec} — files missing")
            skipped_records.append(rec)
            continue

        try:
            signal, fs = load_record(rec_path)
            r_samples, symbols = load_annotations(rec_path)

            filtered = bandpass_filter_360hz(signal)
            windows, labels = extract_windows(filtered, r_samples, symbols)

            n_normal = labels.count(0)
            n_abnormal = labels.count(1)
            total_normal += n_normal
            total_abnormal += n_abnormal

            all_windows.extend(windows)
            all_labels.extend(labels)

            print(f"  [OK]   {rec}: {len(labels)} beats "
                  f"(normal={n_normal}, abnormal={n_abnormal})")

        except Exception as e:
            print(f"  [FAIL] {rec}: {e}")
            skipped_records.append(rec)

    X = np.array(all_windows, dtype=np.float64)
    X = normalize_windows(X).astype(np.float32)
    y = np.array(all_labels, dtype=np.int8)

    print(f"\nTotal: {len(y)} beats | Normal: {total_normal} "
          f"({100*total_normal/max(1,len(y)):.1f}%) | "
          f"Abnormal: {total_abnormal} ({100*total_abnormal/max(1,len(y)):.1f}%)")

    return X, y, skipped_records


def save_dataset(X, y, output_dir, records_used, skipped):
    """Save arrays and metadata to disk."""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y.npy'), y)

    n_normal = int((y == 0).sum())
    n_abnormal = int((y == 1).sum())
    metadata = {
        'total_beats': int(len(y)),
        'n_normal': n_normal,
        'n_abnormal': n_abnormal,
        'class_balance_pct_abnormal': round(100 * n_abnormal / max(1, len(y)), 2),
        'window_samples': WINDOW_SAMPLES,
        'sampling_rate_hz': FS,
        'window_ms': round(1000 * WINDOW_SAMPLES / FS, 1),
        'records_used': records_used,
        'records_skipped': skipped,
        'label_scheme': 'binary: N=0 (normal), all others=1 (abnormal)',
        'normalization': 'per-beat z-score',
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved to: {output_dir}")
    print(f"  X.npy: {X.shape}  ({X.nbytes/1e6:.1f} MB)")
    print(f"  y.npy: {y.shape}")
    print(f"  metadata.json")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess MIT-BIH records into labeled beat windows',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo/preprocess_mitdb.py
  python demo/preprocess_mitdb.py --raw-dir data/mitdb_raw --output-dir data/mitdb_processed
  python demo/preprocess_mitdb.py --records 100 101 103
        """
    )
    parser.add_argument(
        '--raw-dir',
        default='data/mitdb_raw',
        help='Directory containing downloaded MIT-BIH records (default: data/mitdb_raw)'
    )
    parser.add_argument(
        '--output-dir',
        default='data/mitdb_processed',
        help='Output directory for processed numpy arrays (default: data/mitdb_processed)'
    )
    parser.add_argument(
        '--records', '-r',
        nargs='+',
        default=DEFAULT_RECORDS,
        help='Record numbers to process (default: all 16)'
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    raw_dir = os.path.join(project_root, args.raw_dir)
    output_dir = os.path.join(project_root, args.output_dir)

    try:
        import wfdb
    except ImportError:
        print("ERROR: wfdb not installed. Run: pip install wfdb")
        sys.exit(1)

    print(f"Processing MIT-BIH records from: {raw_dir}")
    print(f"Records: {', '.join(args.records)}\n")

    X, y, skipped = process_all_records(args.records, raw_dir)

    if len(y) == 0:
        print("\nERROR: No beats extracted. Check that records are downloaded.")
        sys.exit(1)

    used = [r for r in args.records if r not in skipped]
    save_dataset(X, y, output_dir, used, skipped)

    print(f"\nNext step — train models (can run in parallel):")
    print(f"  python models/cnn/train.py  --data-dir {args.output_dir}")
    print(f"  python models/lstm/train.py --data-dir {args.output_dir}")
    print(f"  python models/rnn/train.py  --data-dir {args.output_dir}")


if __name__ == '__main__':
    main()
