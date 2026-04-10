"""
demo/models/cnn_resnet.py

Residual 1D CNN for ECG beat classification.

Why ResNet over plain CNN:
- Residual connections prevent vanishing gradients in deeper networks
- Skip connections let the model learn "what to change" rather than
  a full transformation — works especially well on ECG where small
  waveform deviations (ST changes, QRS width) are the signal
- State-of-the-art for 1D physiological time series (Ribeiro et al. 2020)

Architecture:
    Initial conv → 4 residual blocks with increasing filters → GAP → Dense

Input:  (216, 1) — 600ms beat window at 360Hz, z-score normalized
Output: sigmoid probability (0=normal, 1=arrhythmia)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, metrics


def _residual_block(x, filters, kernel_size=5, downsample=False):
    """Single residual block with optional downsampling via stride.

    Structure:
        Conv → BN → ReLU → Dropout → Conv → BN
        + shortcut (projection if filters changed)
        → Add → ReLU
    """
    stride = 2 if downsample else 1
    shortcut = x

    # Main path
    x = layers.Conv1D(filters, kernel_size, strides=stride,
                      padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1)(x)  # Light dropout inside block for regularisation

    x = layers.Conv1D(filters, kernel_size,
                      padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Projection shortcut — align dimensions when filters change or downsampling
    if shortcut.shape[-1] != filters or downsample:
        shortcut = layers.Conv1D(filters, 1, strides=stride,
                                 padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def build_cnn_resnet(input_shape=(216, 1)):
    """Build Residual 1D CNN.

    Block structure (filters × length after stride):
        Initial 32×216
        Block 1: 32×216   (no downsample — preserve fine-grained P/Q/S features)
        Block 2: 64×108   (downsample → broader context)
        Block 3: 64×108
        Block 4: 128×54   (downsample → high-level pattern)
        GAP → Dense(64) → Dense(1)
    """
    inp = layers.Input(shape=input_shape)

    # Initial feature extraction — wider kernel to capture QRS shape
    x = layers.Conv1D(32, kernel_size=15, padding='same', use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Residual blocks
    x = _residual_block(x, filters=32, kernel_size=5, downsample=False)
    x = _residual_block(x, filters=64, kernel_size=5, downsample=True)   # 108
    x = _residual_block(x, filters=64, kernel_size=5, downsample=False)
    x = _residual_block(x, filters=128, kernel_size=3, downsample=True)  # 54

    # Global average pooling collapses the temporal dimension without
    # throwing away positional information (better than Flatten here)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inp, outputs=out, name='cnn_resnet')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', metrics.AUC(name='auc')]
    )
    return model
