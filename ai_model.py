import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, Bidirectional, 
                                     Layer, Embedding, MultiHeadAttention, Concatenate)
from tensorflow.keras.callbacks import CSVLogger
import os
import pandas as pd
from markov_model import top6_markov

TEMPERATURE = 1.2  # untuk scaling softmax

class PositionalEncoding(Layer):
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        d_model = tf.shape(inputs)[2]
        position = tf.cast(tf.range(seq_len)[:, tf.newaxis], dtype=tf.float32)
        i = tf.cast(tf.range(d_model)[tf.newaxis, :], dtype=tf.float32)
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = position * angle_rates
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = tf.expand_dims(pos_encoding, 0)
        return inputs + tf.cast(pos_encoding, tf.float32)

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

def build_lstm_model(attention=True, positional=True):
    inputs = Input(shape=(3,))
    x = Embedding(input_dim=10, output_dim=8)(inputs)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    if positional:
        x = PositionalEncoding()(x)
    if attention:
        attn = MultiHeadAttention(num_heads=2, key_dim=64)(x, x)
        x = Concatenate()([x, attn])
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.2)(x)

    # temperature scaling
    def scaled_dense(name):
        return Dense(10, activation=lambda x: tf.nn.softmax(x / TEMPERATURE), name=name)

    outputs = [scaled_dense(f"output_{i}")(x) for i in range(4)]
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_and_save_lstm(df, lokasi):
    if len(df) < 20: return
    X, y = preprocess_data(df)
    model = build_lstm_model()
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("training_logs", exist_ok=True)
    log_path = f"training_logs/history_{lokasi.lower().replace(' ', '_')}.csv"
    csv_logger = CSVLogger(log_path)
    model.fit(X, y, epochs=30, batch_size=16, verbose=0, callbacks=[csv_logger])
    model.save(f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5")

def model_exists(lokasi):
    return os.path.exists(f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5")

def top6_lstm(df, lokasi=None, return_probs=False, return_accuracy=False):
    try:
        model = load_model(f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5",
                           compile=False,
                           custom_objects={"PositionalEncoding": PositionalEncoding})
        sequences = []
        true_digits = [[] for _ in range(4)]
        for angka in df["angka"]:
            digits = [int(d) for d in f"{int(angka):04d}"]
            sequences.append(digits[:-1])
            for i in range(4):
                true_digits[i].append(digits[i])

        X = np.array(sequences)
        y_pred = model.predict(X, verbose=0)
        top6 = []
        probs = []
        accs = []

        for i in range(4):
            avg_probs = np.mean(y_pred[i], axis=0)
            top_idx = avg_probs.argsort()[-6:][::-1]
            top6.append(list(top_idx))
            probs.append(avg_probs[top_idx])

            if return_accuracy:
                # Akurasi per digit posisi i
                correct = sum(np.argmax(y_pred[i], axis=1) == np.array(true_digits[i]))
                acc = 100 * correct / len(true_digits[i])
                accs.append(acc)

        if return_probs and return_accuracy:
            return top6, probs, accs
        if return_probs:
            return top6, probs
        if return_accuracy:
            return top6, accs
        return top6
    except:
        return None

def kombinasi_4d(df, lokasi, top_n=10):
    result, probs = top6_lstm(df, lokasi=lokasi, return_probs=True)
    if result is None: return []
    from itertools import product
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
