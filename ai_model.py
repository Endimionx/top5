import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, Layer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger

# ========== Positional Encoding ==========
class PositionalEncoding(Layer):
    def __init__(self, max_len=5000, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_len = max_len

    def build(self, input_shape):
        d_model = input_shape[-1]
        position = tf.range(self.max_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * (-tf.math.log(10000.0) / d_model))
        pe = tf.zeros((self.max_len, d_model))
        pe = tf.tensor_scatter_nd_update(
            pe,
            indices=tf.range(0, d_model, 2)[:, tf.newaxis],
            updates=tf.sin(position * div_term)
        )
        pe = tf.tensor_scatter_nd_update(
            pe,
            indices=tf.range(1, d_model, 2)[:, tf.newaxis],
            updates=tf.cos(position * div_term)
        )
        self.pe = pe[tf.newaxis, ...]

    def call(self, x):
        length = tf.shape(x)[1]
        return x + self.pe[:, :length, :]

# ========== Attention Layer ==========
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def call(self, inputs):
        score = tf.matmul(inputs, inputs, transpose_b=True)
        weights = tf.nn.softmax(score, axis=-1)
        return tf.matmul(weights, inputs)

# ========== Data Preparation ==========
def prepare_lstm_data(df):
    data = df['angka'].dropna().apply(lambda x: [int(d) for d in f"{int(x):04d}"]).tolist()
    data = np.array(data)

    if len(data) < 11:
        return None, None

    X, y = [], []
    for i in range(len(data) - 10):
        X.append(data[i:i+10])
        y.append(data[i+10])
    X, y = np.array(X), np.array(y)

    if len(X) == 0 or len(y) == 0:
        return None, None

    y_encoded = [to_categorical(y[:, i], num_classes=10) for i in range(4)]
    return X, y_encoded

# ========== Check Model ==========
def model_exists(lokasi):
    filename = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"
    return os.path.exists(filename)

# ========== Train & Save ==========
def train_and_save_lstm(df, lokasi):
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("training_logs", exist_ok=True)
    filename = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"
    log_file = f"training_logs/history_{lokasi.lower().replace(' ', '_')}.csv"

    X, y = prepare_lstm_data(df)
    if X is None or y is None:
        return

    if os.path.exists(filename):
        print(f"🔁 Fine-tuning model untuk {lokasi}")
        model = load_model(filename, custom_objects={
            "PositionalEncoding": PositionalEncoding,
            "AttentionLayer": AttentionLayer
        })
    else:
        print(f"🆕 Membuat model baru untuk {lokasi}")
        inputs = Input(shape=(10, 4))
        x = PositionalEncoding()(inputs)
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Dropout(0.2)(x)
        x = AttentionLayer()(x)
        x = Bidirectional(LSTM(64))(x)
        outputs = [Dense(10, activation='softmax', name=f'output_{i}')(x) for i in range(4)]
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3),
        CSVLogger(log_file, append=False)
    ]

    model.fit(X, y, epochs=50, batch_size=16, verbose=0, callbacks=callbacks)
    model.save(filename)

# ========== Predict Top-N per Digit ==========
def top6_lstm(df, lokasi=None, return_probs=False, top_n=6):
    filename = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"
    if not os.path.exists(filename):
        return None

    data = df['angka'].dropna().apply(lambda x: [int(d) for d in f"{int(x):04d}"]).tolist()
    data = np.array(data)
    if len(data) < 10:
        return None

    model = load_model(filename, custom_objects={
        "PositionalEncoding": PositionalEncoding,
        "AttentionLayer": AttentionLayer
    })
    input_seq = np.array(data[-10:]).reshape(1, 10, 4)
    preds = model.predict(input_seq, verbose=0)

    if return_probs:
        return preds

    result = []
    for i in range(4):
        top = list(np.argsort(-preds[i][0])[:top_n])
        result.append(top)
    return result

# ========== Kombinasi 4D dengan Confidence ==========
def prediksi_kombinasi_4d(df, lokasi, top_n=10):
    probs = top6_lstm(df, lokasi=lokasi, return_probs=True)
    if probs is None:
        return None

    top_digit = [np.argsort(-probs[i][0])[:4] for i in range(4)]
    hasil = []

    for a in top_digit[0]:
        for b in top_digit[1]:
            for c in top_digit[2]:
                for d in top_digit[3]:
                    conf = probs[0][0][a] * probs[1][0][b] * probs[2][0][c] * probs[3][0][d]
                    hasil.append((f"{a}{b}{c}{d}", conf))

    hasil.sort(key=lambda x: -x[1])
    return hasil[:top_n]

# ========== Prediksi Rendah ==========
def anti_top6_lstm(df, lokasi=None):
    probs = top6_lstm(df, lokasi=lokasi, return_probs=True)
    if probs is None:
        return None
    result = []
    for i in range(4):
        low = list(np.argsort(probs[i][0])[:6])
        result.append(low)
    return result

def low6_lstm(df, lokasi=None):
    return anti_top6_lstm(df, lokasi)
