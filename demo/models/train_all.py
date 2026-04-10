"""
demo/models/train_all.py

Unified training script for all three demo models:
  - CNN ResNet  (demo/models/cnn_resnet.py)
  - LSTM + Attention  (demo/models/lstm_attention.py)
  - BiGRU  (demo/models/gru_model.py)

Trains each model on MIT-BIH processed data and saves to demo/models/saved/.

Usage:
    python demo/models/train_all.py
    python demo/models/train_all.py --data-dir data/mitdb_processed
    python demo/models/train_all.py --model cnn     # train only CNN
    python demo/models/train_all.py --model lstm    # train only LSTM
    python demo/models/train_all.py --model gru     # train only GRU
    python demo/models/train_all.py --epochs 50 --batch-size 64
"""

import argparse
import json
import os
import sys
import time
import numpy as np

# Resolve paths
_script_dir = os.path.dirname(os.path.abspath(__file__))
_demo_dir = os.path.dirname(_script_dir)
_project_root = os.path.dirname(_demo_dir)
sys.path.insert(0, _script_dir)

SAVE_DIR = os.path.join(_demo_dir, 'models', 'saved')

MODEL_SAVE_PATHS = {
    'cnn':  os.path.join(SAVE_DIR, 'cnn_resnet.h5'),
    'lstm': os.path.join(SAVE_DIR, 'lstm_attention.h5'),
    'gru':  os.path.join(SAVE_DIR, 'gru_model.h5'),
}


# ======================================================================
# Data loading

def load_data(data_dir):
    """Load preprocessed beat windows and labels from demo/preprocess_mitdb.py output."""
    X_path = os.path.join(data_dir, 'X.npy')
    y_path = os.path.join(data_dir, 'y.npy')
    meta_path = os.path.join(data_dir, 'metadata.json')

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print(f"ERROR: Data not found in {data_dir}")
        print("Run: python demo/preprocess_mitdb.py")
        sys.exit(1)

    X = np.load(X_path)   # (N, 216)
    y = np.load(y_path)   # (N,)

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  Dataset: {meta.get('total_beats', len(y))} beats | "
              f"Normal: {meta.get('n_normal', (y==0).sum())} | "
              f"Abnormal: {meta.get('n_abnormal', (y==1).sum())} | "
              f"Window: {meta.get('window_samples', 216)} samples @ "
              f"{meta.get('sampling_rate_hz', 360)} Hz")

    return X, y


def compute_class_weights(y_train):
    """Inverse-frequency class weights for the imbalanced MIT-BIH dataset (~75% normal)."""
    n_normal = (y_train == 0).sum()
    n_abnormal = (y_train == 1).sum()
    weight = n_normal / max(1, n_abnormal)
    return {0: 1.0, 1: float(weight)}


# ======================================================================
# Training

def train_model(name, build_fn, X_train, y_train, X_val, y_val,
                epochs, batch_size, save_path):
    """Train a single model with early stopping and LR scheduling."""
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    print(f"\n{'='*60}")
    print(f"  Training: {name.upper()}")
    print(f"  Save to:  {save_path}")
    print(f"{'='*60}")

    model = build_fn(input_shape=(X_train.shape[1], 1))
    model.summary(print_fn=lambda s: print(f"  {s}"))

    class_weights = compute_class_weights(y_train)
    print(f"\n  Class weights: {class_weights}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=10,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=0
        )
    ]

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    elapsed = time.time() - t0
    print(f"\n  Training time: {elapsed/60:.1f} min")

    # Evaluation
    y_prob = model.predict(X_val, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_val, y_prob)
    cm = confusion_matrix(y_val, y_pred)

    print(f"\n  === {name.upper()} Validation Results ===")
    print(f"  AUC-ROC: {auc:.4f}")
    print(classification_report(y_val, y_pred,
                                target_names=['Normal', 'Abnormal'],
                                digits=3))
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    # Save results alongside model
    results_path = save_path.replace('.h5', '_results.json')
    results = {
        'model': name,
        'val_auc': round(auc, 4),
        'val_accuracy': round(float((y_pred == y_val).mean()), 4),
        'confusion_matrix': cm.tolist(),
        'training_time_min': round(elapsed / 60, 1),
        'epochs_trained': len(history.history['loss']),
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    return model, results


# ======================================================================
# Main

def main():
    parser = argparse.ArgumentParser(
        description='Train demo ECG models (ResNet CNN, Attention LSTM, BiGRU)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all three models
  python demo/models/train_all.py

  # Train only the CNN
  python demo/models/train_all.py --model cnn

  # More epochs, larger batch
  python demo/models/train_all.py --epochs 50 --batch-size 64

  # Custom data dir
  python demo/models/train_all.py --data-dir data/mitdb_processed
        """
    )
    parser.add_argument('--data-dir', default='data/mitdb_processed',
                        help='Directory with X.npy, y.npy (default: data/mitdb_processed)')
    parser.add_argument('--model', choices=['cnn', 'lstm', 'gru', 'all'],
                        default='all', help='Which model to train (default: all)')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Max training epochs per model (default: 40)')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    # Resolve data dir relative to project root
    data_dir = os.path.join(_project_root, args.data_dir)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 60)
    print("  Demo Model Training — CNN ResNet / Attention LSTM / BiGRU")
    print("=" * 60)
    print(f"\nData: {data_dir}")
    print(f"Save: {SAVE_DIR}\n")

    X, y = load_data(data_dir)
    X = X.reshape(X.shape[0], X.shape[1], 1).astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train: {len(y_train)}  |  Val: {len(y_val)}")

    # Import model builders
    from cnn_resnet import build_cnn_resnet
    from lstm_attention import build_lstm_attention
    from gru_model import build_gru_model

    registry = {
        'cnn':  (build_cnn_resnet,    MODEL_SAVE_PATHS['cnn']),
        'lstm': (build_lstm_attention, MODEL_SAVE_PATHS['lstm']),
        'gru':  (build_gru_model,      MODEL_SAVE_PATHS['gru']),
    }

    to_train = list(registry.keys()) if args.model == 'all' else [args.model]
    all_results = {}

    for name in to_train:
        build_fn, save_path = registry[name]
        _, results = train_model(
            name, build_fn,
            X_train, y_train, X_val, y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_path=save_path
        )
        all_results[name] = results

    # Summary table
    print(f"\n{'='*60}")
    print("  TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<20}  {'AUC':>6}  {'Accuracy':>9}  {'Epochs':>6}")
    print(f"  {'-'*50}")
    for name, r in all_results.items():
        print(f"  {name:<20}  {r['val_auc']:>6.4f}  "
              f"{r['val_accuracy']:>9.4f}  {r['epochs_trained']:>6}")

    print(f"\nModels saved to: {SAVE_DIR}")
    print(f"\nNext — run the live demo:")
    print(f"  python demo/live_demo.py --record data/mitdb_raw/119 --speed 5.0 --loop")


if __name__ == '__main__':
    main()
