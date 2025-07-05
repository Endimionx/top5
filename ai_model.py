import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Layer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from markov_model import top6_markov

class PositionalEncoding(Layer):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def call(self, x):
        seq_len = tf.shape(x)[1]
        d_model = tf.shape(x)[2]
        position = tf.cast(tf.range(seq_len)[:, tf.newaxis], dtype=tf.float32)
        div_term = tf.cast(tf.range(d_model)[tf.newaxis, :], dtype=tf.float32)
        angle_rates = 1 / tf.pow(10000.0, (2 * (div_term // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = position * angle_rates
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return x + pos_encoding

class AttentionLayer(Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def call(self, x):
        query = tf.keras.layers.Dense(x.shape[-1])(x)
        key = tf.keras.layers.Dense(x.shape[-1])(x)
        value = tf.keras.layers.Dense(x.shape[-1])(x)
        scores = tf.matmul(query, key, transpose_b=True)
        weights = tf.nn.softmax(scores / tf.math.sqrt(tf.cast(x.shape[-1], tf.float32)), axis=-1)
        context = tf.matmul(weights, value)
        return context[:, -1, :]

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
    y_encoded = [to_categorical(y[:, i], num_classes=10) for i in range(4)]
    return X, y_encoded

def model_exists(lokasi):
    filename = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"
    return os.path.exists(filename)

def train_and_save_lstm(df, lokasi):
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("training_logs", exist_ok=True)
    filename = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"
    log_file = f"training_logs/history_{lokasi.lower().replace(' ', '_')}.csv"

    X, y = prepare_lstm_data(df)
    if X is None or y is None:
        return

    if os.path.exists(filename):
        model = load_model(filename, custom_objects={
            "AttentionLayer": AttentionLayer,
            "PositionalEncoding": PositionalEncoding
        })
    else:
        inputs = Input(shape=(10, 4))
        x = PositionalEncoding()(inputs)
        x = LSTM(128, return_sequences=True)(x)
        x = Dropout(0.3)(x)
        x = LSTM(64, return_sequences=True)(x)
        x = AttentionLayer()(x)
        outputs = [Dense(10, activation='softmax', name=f'output_{i}')(x) for i in range(4)]
        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3),
        CSVLogger(log_file, append=False)
    ]

    model.fit(X, y, epochs=50, batch_size=16, verbose=0, callbacks=callbacks)
    model.save(filename)

def top6_lstm(df, lokasi=None, return_probs=False):
    filename = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"
    if not os.path.exists(filename):
        return None
    data = df['angka'].dropna().apply(lambda x: [int(d) for d in f"{int(x):04d}"]).tolist()
    if len(data) < 10:
        return None
    input_seq = np.array(data[-10:]).reshape(1, 10, 4)
    model = load_model(filename, custom_objects={
        "AttentionLayer": AttentionLayer,
        "PositionalEncoding": PositionalEncoding
    })
    preds = model.predict(input_seq, verbose=0)
    top6 = []
    probs = []
    for i in range(4):
        top_idx = np.argsort(-preds[i][0])[:6]
        top6.append(list(top_idx))
        probs.append(preds[i][0])
    return (top6, probs) if return_probs else top6

def kombinasi_4d(df, lokasi=None, top_n=10):
    result = top6_lstm(df, lokasi=lokasi, return_probs=True)
    if result is None:
        return None
    _, probs = result
    top_digits = [np.argsort(-p)[:4] for p in probs]
    kombinasi = []
    for a in top_digits[0]:
        for b in top_digits[1]:
            for c in top_digits[2]:
                for d in top_digits[3]:
                    skor = probs[0][a] * probs[1][b] * probs[2][c] * probs[3][d]
                    kombinasi.append((f"{a}{b}{c}{d}", skor))
    kombinasi.sort(key=lambda x: -x[1])
    return kombinasi[:top_n]

def ensemble_lstm_markov(df, lokasi=None):
    lstm_result = top6_lstm(df, lokasi)
    markov_result = top6_markov(df)[0] if isinstance(top6_markov(df), tuple) else top6_markov(df)
    if lstm_result is None or markov_result is None:
        return None
    hasil = []
    for i in range(4):
        gabung = lstm_result[i] + markov_result[i]
        freq = {x: gabung.count(x) for x in set(gabung)}
        top6 = sorted(freq.items(), key=lambda x: -x[1])
        top6 = [k for k, _ in top6[:6]]
        hasil.append(top6)
    return hasil
