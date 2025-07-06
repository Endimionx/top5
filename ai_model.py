import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Embedding
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
import os
import pandas as pd
from markov_model import top6_markov
from tensorflow.keras.utils import to_categorical

class PositionalEncoding(tf.keras.layers.Layer):
    def call(self, x):
        seq_len = tf.shape(x)[1]
        d_model = tf.shape(x)[2]
        pos = tf.cast(tf.range(seq_len)[:, tf.newaxis], dtype=tf.float32)
        i = tf.cast(tf.range(d_model)[tf.newaxis, :], dtype=tf.float32)
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = pos * angle_rates
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = tf.expand_dims(pos_encoding, 0)
        return x + tf.cast(pos_encoding, tf.float32)

def preprocess_data(df):
    sequences = []
    targets = [[] for _ in range(4)]
    for angka in df["angka"]:
        digits = [int(d) for d in f"{int(angka):04d}"]
        sequences.append(digits[:-1])  # input: 3 digit pertama
        for i in range(4):
            targets[i].append(to_categorical(digits[i], num_classes=10))
    X = np.array(sequences)
    y = [np.array(t) for t in targets]
    return X, y

def build_model(input_dim=10, embed_dim=8, lstm_units=64, attention_heads=2, use_positional_encoding=True):
    inputs = Input(shape=(3,))
    x = Embedding(input_dim=input_dim, output_dim=embed_dim)(inputs)
    if use_positional_encoding:
        x = PositionalEncoding()(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = MultiHeadAttention(num_heads=attention_heads, key_dim=embed_dim)(x, x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    model = Model(inputs, x)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_and_save_lstm(df, lokasi):
    if len(df) < 20: return
    X, y_all = preprocess_data(df)
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("training_logs", exist_ok=True)

    for i in range(4):
        y = y_all[i]
        model = build_model()
        log_path = f"training_logs/history_{lokasi.lower().replace(' ', '_')}_digit{i}.csv"
        callbacks = [
            CSVLogger(log_path),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
        model.fit(X, y, epochs=50, batch_size=16, verbose=0, callbacks=callbacks, validation_split=0.1)
        model.save(f"saved_models/{lokasi.lower().replace(' ', '_')}_digit{i}.h5")

def model_exists(lokasi):
    return all(os.path.exists(f"saved_models/{lokasi.lower().replace(' ', '_')}_digit{i}.h5") for i in range(4))

def top6_lstm(df, lokasi=None, return_probs=False):
    X, _ = preprocess_data(df)
    results, probs = [], []
    for i in range(4):
        path = f"saved_models/{lokasi.lower().replace(' ', '_')}_digit{i}.h5"
        if not os.path.exists(path):
            return None
        try:
            model = load_model(path, compile=False, custom_objects={"PositionalEncoding": PositionalEncoding})
            pred = model.predict(X, verbose=0)
            avg = np.mean(pred, axis=0)
            top6 = avg.argsort()[-6:][::-1]
            results.append(list(top6))
            probs.append(avg[top6])
        except:
            return None
    if return_probs:
        return results, probs
    return results

def kombinasi_4d(df, lokasi, top_n=10):
    result, probs = top6_lstm(df, lokasi=lokasi, return_probs=True)
    if result is None: return []
    from itertools import product
    combinations = list(product(*result))
    scores = []
    for combo in combinations:
        score = 1.0
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
        combined = lstm_result[i] + markov_result[i]
        freq = {x: combined.count(x) for x in set(combined)}
        top6 = sorted(freq.items(), key=lambda x: -x[1])[:6]
        ensemble.append([x[0] for x in top6])
    return ensemble
