"""
models/cnn/train.py

Train a 1D CNN for arrhythmia detection on MIT-BIH beat windows.

Architecture:
    Conv1D(32,5) → BN → MaxPool
    Conv1D(64,5) → BN → MaxPool
    Conv1D(128,3) → BN → MaxPool
    GlobalAveragePooling1D
    Dense(64, relu) → Dropout(0.3)
    Dense(1, sigmoid)

Input:  (216, 1) beat window, z-score normalized
Output: probability of arrhythmia (0=normal, 1=abnormal)

Usage:
    python models/cnn/train.py
    python models/cnn/train.py --data-dir data/mitdb_processed --output models/cnn/cnn_mitdb.h5
    python models/cnn/train.py --epochs 50 --batch-size 64
"""

import argparse
import os
import sys
import numpy as np


def build_cnn_model(input_shape=(216, 1)):
    """Build 1D CNN for beat classification."""
    from tensorflow.keras import layers, models, metrics

    inp = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv1D(32, kernel_size=5, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)  # → (108, 32)

    # Block 2
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)  # → (54, 64)

    # Block 3
    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)  # → (27, 128)

    # Global average pooling collapses temporal dimension
    x = layers.GlobalAveragePooling1D()(x)  # → (128,)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', metrics.AUC(name='auc')]
    )
    return model


def load_data(data_dir):
    """Load preprocessed beat windows and labels."""
    X_path = os.path.join(data_dir, 'X.npy')
    y_path = os.path.join(data_dir, 'y.npy')

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print(f"ERROR: Data files not found in {data_dir}")
        print("Run: python demo/preprocess_mitdb.py first")
        sys.exit(1)

    X = np.load(X_path)  # (N, 216)
    y = np.load(y_path)  # (N,)
    return X, y


def compute_class_weights(y):
    """Compute class weights to handle ~75% normal class imbalance."""
    n_normal = (y == 0).sum()
    n_abnormal = (y == 1).sum()
    # Weight abnormal class proportionally higher
    weight_abnormal = n_normal / max(1, n_abnormal)
    return {0: 1.0, 1: float(weight_abnormal)}


def print_results(model, X_val, y_val):
    """Print validation metrics and confusion matrix."""
    from sklearn.metrics import confusion_matrix, classification_report
    import numpy as np

    y_prob = model.predict(X_val, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n=== Validation Results ===")
    print(classification_report(y_val, y_pred, target_names=['Normal', 'Abnormal']))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_val, y_pred)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")


def main():
    parser = argparse.ArgumentParser(
        description='Train 1D CNN arrhythmia classifier on MIT-BIH beats',
    )
    parser.add_argument('--data-dir', default='data/mitdb_processed',
                        help='Directory with X.npy, y.npy (default: data/mitdb_processed)')
    parser.add_argument('--output', '-o', default='models/cnn/cnn_mitdb.h5',
                        help='Output model path (default: models/cnn/cnn_mitdb.h5)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    # Resolve paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(project_root, args.data_dir)
    output_path = os.path.join(project_root, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    print("=== CNN Arrhythmia Classifier ===")
    print(f"Data:   {data_dir}")
    print(f"Output: {output_path}\n")

    X, y = load_data(data_dir)
    print(f"Dataset: {X.shape[0]} beats | Normal: {(y==0).sum()} | Abnormal: {(y==1).sum()}")

    # Reshape for Keras: (N, 216, 1)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(y_train)} | Val: {len(y_val)}\n")

    class_weights = compute_class_weights(y_train)
    print(f"Class weights: {class_weights}\n")

    model = build_cnn_model(input_shape=(X.shape[1], 1))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=8, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4, verbose=1
        )
    ]

    print("\nTraining...")
    model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    model.save(output_path)
    print(f"\nModel saved to: {output_path}")

    print_results(model, X_val, y_val)

    print(f"\nNext: python demo/live_demo.py --cnn {args.output}")


if __name__ == '__main__':
    main()
