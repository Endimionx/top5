import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM, Dropout,
    Dense, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, Add
)
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
from tensorflow.keras.utils import to_categorical
import os
import pandas as pd
from itertools import product
from markov_model import top6_markov


# 🔷 Trainable Positional Encoding
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.pos_encoding = self.add_weight(
            shape=(1, input_shape[1], input_shape[2]),
            initializer="random_normal",
            trainable=True,
            name="pos_encoding"
        )

    def call(self, x):
        return x + self.pos_encoding


# 🔷 Preprocess untuk semua digit
def preprocess_data(df, window_size=5):
    sequences = []
    targets = [[] for _ in range(4)]
    angka = df["angka"].values
    for i in range(len(angka) - window_size):
        window = angka[i:i+window_size]
        if any(len(x) != 4 or not x.isdigit() for x in window):
            continue
        seq = [int(d) for num in window[:-1] for d in f"{int(num):04d}"]
        sequences.append(seq)
        target_digits = [int(d) for d in f"{int(window[-1]):04d}"]
        for j in range(4):
            targets[j].append(to_categorical(target_digits[j], num_classes=10))
    X = np.array(sequences)
    y = [np.array(t) for t in targets]
    return X, y


# 🔷 Build model per digit
def build_model(input_len, embed_dim=16, lstm_units=64, attention_heads=2, temperature=0.5):
    inputs = Input(shape=(input_len,))
    x = Embedding(input_dim=10, output_dim=embed_dim)(inputs)
    x = PositionalEncoding()(x)

    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.5)(x)

    x2 = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    attn_out = MultiHeadAttention(num_heads=attention_heads, key_dim=embed_dim)(x2, x2)
    x = Add()([x2, attn_out])  # Residual
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    logits = Dense(10)(x)
    outputs = tf.keras.layers.Activation('softmax')(logits / temperature)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# 🔷 Train semua digit & simpan
def train_and_save_lstm(df, lokasi, window_size=5):
    if len(df) < window_size + 5:
        return
    X, y_all = preprocess_data(df, window_size=window_size)
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("training_logs", exist_ok=True)
    for i in range(4):
        y = y_all[i]
        model = build_model(input_len=X.shape[1])
        log_path = f"training_logs/history_{lokasi.lower().replace(' ', '_')}_digit{i}.csv"
        callbacks = [
            CSVLogger(log_path),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
        model.fit(X, y, epochs=50, batch_size=16, verbose=0, validation_split=0.2, callbacks=callbacks)
        model.save(f"saved_models/{lokasi.lower().replace(' ', '_')}_digit{i}.h5")


# 🔷 Cek semua model digit tersedia
def model_exists(lokasi):
    return all(os.path.exists(f"saved_models/{lokasi.lower().replace(' ', '_')}_digit{i}.h5") for i in range(4))


# 🔷 Prediksi top 6 per digit
def top6_lstm(df, lokasi=None, return_probs=False, temperature=0.5):
    X, _ = preprocess_data(df)
    if len(X) == 0:
        return None
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
        except Exception:
            return None
    return (results, probs) if return_probs else results


# 🔷 Kombinasi 4D Confidence Voting
def kombinasi_4d(df, lokasi, top_n=10, min_conf=0.0001, power=1.5, mode='product'):
    result, probs = top6_lstm(df, lokasi=lokasi, return_probs=True)
    if result is None or probs is None:
        return []
    combinations = list(product(*result))
    scores = []
    for combo in combinations:
        digit_scores = []
        valid = True
        for i in range(4):
            try:
                idx = result[i].index(combo[i])
                digit_scores.append(probs[i][idx] ** power)
            except:
                valid = False
                break
        if not valid:
            continue
        score = np.prod(digit_scores) if mode == 'product' else np.mean(digit_scores)
        if score >= min_conf:
            scores.append(("".join(map(str, combo)), score))
    topk = sorted(scores, key=lambda x: -x[1])[:top_n]
    return topk


# 🔷 Ensemble LSTM + Markov
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
