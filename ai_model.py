import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, Attention, Concatenate, Layer
from tensorflow.keras.callbacks import CSVLogger
import os
import pandas as pd
from markov_model import top6_markov
from itertools import product

# ===== Positional Encoding Layer =====
class PositionalEncoding(Layer):
    def __init__(self, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)

    def call(self, inputs):
        batch_size, seq_len, d_model = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
        position = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(np.log(10000.0) / d_model))
        pe = tf.zeros((seq_len, d_model))
        pe = tf.tensor_scatter_nd_update(pe, [[i, j] for i in range(seq_len) for j in range(0, d_model, 2)],
                                         tf.reshape(tf.math.sin(position * div_term), [-1]))
        pe = tf.tensor_scatter_nd_update(pe, [[i, j + 1] for i in range(seq_len) for j in range(0, d_model - 1, 2)],
                                         tf.reshape(tf.math.cos(position * div_term), [-1]))
        pe = tf.expand_dims(pe, 0)
        return inputs + pe

# ===== Preprocessing =====
def preprocess_data(df):
    sequences = []
    targets = [[] for _ in range(4)]
    for angka in df["angka"]:
        digits = [int(d) for d in f"{int(angka):04d}"]
        sequences.append(digits[:-1])
        for i in range(4):
            targets[i].append(tf.keras.utils.to_categorical(digits[i], num_classes=10))
    X = np.array(sequences)
    y = [np.array(t) for t in targets]
    return X, y

# ===== Build LSTM Model =====
def build_lstm_model(attention=True, positional=True):
    inputs = Input(shape=(3,))
    x = tf.keras.layers.Embedding(input_dim=10, output_dim=16)(inputs)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    if positional:
        x = PositionalEncoding()(x)
    if attention:
        attn = Attention()([x, x])
        x = Concatenate()([x, attn])
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.3)(x)
    outputs = [Dense(10, activation="softmax", name=f"output_{i}")(x) for i in range(4)]
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# ===== Training =====
def train_and_save_lstm(df, lokasi):
    if len(df) < 20:
        return
    X, y = preprocess_data(df)
    model = build_lstm_model(attention=True, positional=True)
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("training_logs", exist_ok=True)
    log_path = f"training_logs/history_{lokasi.lower().replace(' ', '_')}.csv"
    csv_logger = CSVLogger(log_path)
    model.fit(X, y, epochs=30, batch_size=16, verbose=0, callbacks=[csv_logger])
    model.save(f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5")

# ===== Model Exists =====
def model_exists(lokasi):
    return os.path.exists(f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5")

# ===== Top6 LSTM =====
def top6_lstm(df, lokasi=None, return_probs=False):
    try:
        model_path = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"
        if not os.path.exists(model_path):
            return None
        model = load_model(model_path, compile=False)
        sequences = []
        for angka in df["angka"]:
            digits = [int(d) for d in f"{int(angka):04d}"]
            sequences.append(digits[:-1])
        if len(sequences) == 0:
            return None
        X = np.array(sequences)
        y_pred = model.predict(X, verbose=0)
        top6 = []
        probs = []
        for i in range(4):
            avg_probs = np.mean(y_pred[i], axis=0)
            top_idx = avg_probs.argsort()[-6:][::-1]
            top6.append(list(top_idx))
            probs.append(avg_probs[top_idx])
        if return_probs:
            return top6, probs
        return top6
    except Exception as e:
        print(f"[ERROR top6_lstm] {e}")
        return None

# ===== Kombinasi 4D dari Top6 =====
def kombinasi_4d(df, lokasi, top_n=10):
    result, probs = top6_lstm(df, lokasi=lokasi, return_probs=True)
    if result is None:
        return []
    combinations = list(product(*result))
    scores = []
    for combo in combinations:
        score = 1
        for i in range(4):
            if combo[i] in result[i]:
                idx = result[i].index(combo[i])
                score *= probs[i][idx]
            else:
                score *= 0
        scores.append(("".join(map(str, combo)), score))
    topk = sorted(scores, key=lambda x: -x[1])[:top_n]
    return topk

# ===== Ensemble LSTM + Markov =====
def top6_ensemble(df, lokasi):
    lstm_result = top6_lstm(df, lokasi=lokasi)
    markov_result, _ = top6_markov(df)
    if lstm_result is None or markov_result is None:
        return None
    ensemble = []
    for i in range(4):
        gabung = lstm_result[i] + markov_result[i]
        frek = {x: gabung.count(x) for x in set(gabung)}
        top6 = sorted(frek.items(), key=lambda x: -x[1])[:6]
        ensemble.append([x[0] for x in top6])
    return ensemble

# ===== Anti Top6 (digit yang tidak masuk) =====
def anti_top6_lstm(df, lokasi):
    hasil = top6_lstm(df, lokasi)
    return [[x for x in range(10) if x not in hasil[i]] for i in range(4)] if hasil else None

# ===== Low6 (digit dengan confidence terendah) =====
def low6_lstm(df, lokasi):
    model_path = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"
    if not os.path.exists(model_path):
        return None
    model = load_model(model_path, compile=False)
    sequences = []
    for angka in df["angka"]:
        digits = [int(d) for d in f"{int(angka):04d}"]
        sequences.append(digits[:-1])
    if len(sequences) == 0:
        return None
    X = np.array(sequences)
    y_pred = model.predict(X, verbose=0)
    low6 = []
    for i in range(4):
        avg_probs = np.mean(y_pred[i], axis=0)
        low_idx = avg_probs.argsort()[:6]
        low6.append(list(low_idx))
    return low6
