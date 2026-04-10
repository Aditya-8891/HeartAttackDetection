"""
demo/models/lstm_attention.py

Bidirectional LSTM with additive (Bahdanau-style) self-attention.

Why attention over plain BiLSTM:
- A plain BiLSTM treats all time steps equally when summarising the sequence
- In an ECG beat, the QRS complex (~40ms) and the ST segment (~80ms) carry
  most of the diagnostic information; the isoelectric baseline is mostly noise
- Attention learns to weight those clinically relevant windows automatically
  without being told where they are

Architecture:
    BiLSTM(64, return_sequences=True)
    → Additive attention (Dense(1,tanh) → softmax over time → weighted sum)
    → Dropout(0.3)
    → Dense(64, relu)
    → Dropout(0.2)
    → Dense(1, sigmoid)

Input:  (216, 1) — 600ms beat window at 360Hz, z-score normalized
Output: sigmoid probability (0=normal, 1=arrhythmia)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, metrics


class AdditiveAttention(layers.Layer):
    """Bahdanau-style additive attention over the time dimension.

    Computes a scalar attention weight for each time step, normalises with
    softmax, then returns a weighted sum of the LSTM hidden states.

    This is equivalent to:
        score_t = tanh(W * h_t + b)
        alpha_t = softmax(score_t)
        context = sum_t(alpha_t * h_t)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = layers.Dense(1, activation='tanh')
        self.softmax = layers.Softmax(axis=1)

    def call(self, lstm_output):
        # lstm_output: (batch, time_steps, hidden_dim)
        scores = self.score_dense(lstm_output)         # (batch, time_steps, 1)
        weights = self.softmax(scores)                  # (batch, time_steps, 1)
        context = tf.reduce_sum(weights * lstm_output, axis=1)  # (batch, hidden_dim)
        return context, weights

    def get_config(self):
        return super().get_config()


def build_lstm_attention(input_shape=(216, 1)):
    """Build Attention BiLSTM model."""
    inp = layers.Input(shape=input_shape)

    # BiLSTM over the full beat sequence
    # return_sequences=True so attention can operate on all time steps
    lstm_out = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.1, recurrent_dropout=0.0)
    )(inp)  # shape: (batch, 216, 128)

    # Additive attention — learn which time steps (QRS/ST) matter most
    context, _ = AdditiveAttention(name='beat_attention')(lstm_out)
    # context shape: (batch, 128)

    x = layers.Dropout(0.3)(context)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inp, outputs=out, name='lstm_attention')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', metrics.AUC(name='auc')]
    )
    return model
