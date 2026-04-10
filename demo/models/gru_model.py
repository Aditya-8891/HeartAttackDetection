"""
demo/models/gru_model.py

Bidirectional GRU for ECG beat classification.

Why GRU over SimpleRNN:
- SimpleRNN suffers from vanishing gradients on sequences > ~30 steps;
  a 216-sample beat window is well beyond that
- GRU uses reset/update gates (similar to LSTM) to selectively remember
  context, making it viable for full-beat analysis
- GRU has fewer parameters than LSTM (no separate cell state) — trains
  faster while matching LSTM accuracy on ECG tasks

Why this instead of the existing models/rnn/train.py SimpleRNN:
- SimpleRNN on 216 samples will struggle to connect the P-wave context
  to the ST segment; GRU maintains that longer-range dependency

Architecture:
    BiGRU(64, return_sequences=True) → Dropout
    BiGRU(32)
    Dense(32, relu) → Dropout
    Dense(1, sigmoid)

Input:  (216, 1) — 600ms beat window at 360Hz, z-score normalized
Output: sigmoid probability (0=normal, 1=arrhythmia)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, metrics


def build_gru_model(input_shape=(216, 1)):
    """Build Bidirectional GRU classifier."""
    inp = layers.Input(shape=input_shape)

    # First BiGRU — extract per-step bidirectional context
    x = layers.Bidirectional(
        layers.GRU(64, return_sequences=True, dropout=0.1, recurrent_dropout=0.0)
    )(inp)
    x = layers.Dropout(0.2)(x)

    # Second BiGRU — compress to a single vector
    x = layers.Bidirectional(
        layers.GRU(32, return_sequences=False, dropout=0.1, recurrent_dropout=0.0)
    )(x)

    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inp, outputs=out, name='bigru')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', metrics.AUC(name='auc')]
    )
    return model
