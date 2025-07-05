import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, Concatenate, Layer, Embedding, LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.callbacks import CSVLogger
import os
import pandas as pd
from markov_model import top6_markov

TEMPERATURE = 1.5  # Untuk scaling confidence

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

def preprocess_data(df, window=5):
    sequences = []
    targets = [[] for _ in range(4)]
    padded = ["0000"] * (window - 1) + list(df["angka"])
    for i in range(len(df)):
        window_digits = []
        for j in range(window):
            digits = [int(d) for d in f"{int(padded[i + j]):04d}"]
            window_digits.extend(digits)
        sequences.append(window_digits[:-1])  # buang 1 digit terakhir
        full_digits = [int(d) for d in f"{int(df.iloc[i]['angka']):04d}"]
        for k in range(4):
            targets[k].append(tf.keras.utils.to_categorical(full_digits[k], num_classes=10))
    X = np.array(sequences)
    y = [np.array(t) for t in targets]
    return X, y

def build_digit_model(attention=True, positional=True, input_len=19):
    inputs = Input(shape=(input_len,))
    x = Embedding(input_dim=10, output_dim=8)(inputs)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    if positional:
        x = PositionalEncoding()(x)
    if attention:
        x = MultiHeadAttention(num_heads=4, key_dim=8)(x, x)
        x = LayerNormalization()(x)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.3)(x)
    x = Dense(10)(x)
    output = tf.keras.layers.Activation(lambda z: tf.nn.softmax(z / TEMPERATURE))(x)
    model = Model(inputs, output)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_and_save_lstm(df, lokasi):
    if len(df) < 20:
        print("❌ Data terlalu sedikit untuk pelatihan.")
        return

    X, y = preprocess_data(df)
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("training_logs", exist_ok=True)

    for i in range(4):
        try:
            print(f"🔄 Melatih model digit ke-{i}...")
            name = f"{lokasi.lower().replace(' ', '_')}_digit{i}.h5"
            path = f"saved_models/{name}"
            model = build_digit_model()
            log_path = f"training_logs/history_{name.replace('.h5', '')}.csv"
            csv_logger = CSVLogger(log_path)
            model.fit(X, y[i], epochs=30, batch_size=16, verbose=0, callbacks=[csv_logger])
            model.save(path)
            print(f"✅ Model digit {i} disimpan ke {path}")
        except Exception as e:
            print(f"❌ Gagal melatih digit ke-{i}: {e}")

def model_exists(lokasi):
    for i in range(4):
        if not os.path.exists(f"saved_models/{lokasi.lower().replace(' ', '_')}_digit{i}.h5"):
            return False
    return True

def top6_lstm(df, lokasi=None, return_probs=False):
    try:
        X, _ = preprocess_data(df)
        preds = []
        probs = []
        for i in range(4):
            path = f"saved_models/{lokasi.lower().replace(' ', '_')}_digit{i}.h5"
            model = load_model(path, compile=False, custom_objects={"PositionalEncoding": PositionalEncoding})
            y = model.predict(X, verbose=0)
            avg = np.mean(y, axis=0)
            top_idx = avg.argsort()[-6:][::-1]
            preds.append(list(top_idx))
            probs.append(avg[top_idx])
        if return_probs:
            return preds, probs
        return preds
    except Exception as e:
        print(f"❌ Gagal prediksi LSTM: {e}")
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
