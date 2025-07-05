import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, Attention, Concatenate, Embedding, Layer
from tensorflow.keras.callbacks import CSVLogger
import os
import pandas as pd
from markov_model import top6_markov

# Positional Encoding layer
class PositionalEncoding(Layer):
    def call(self, inputs):
        position = tf.range(tf.shape(inputs)[1], dtype=tf.float32)
        pos_enc = tf.expand_dims(position, axis=0)
        return inputs + pos_enc

def augment_data(df, n_aug=2):
    angka = df["angka"].tolist()
    augmented = []
    for a in angka:
        a = f"{int(a):04d}"
        digits = list(a)
        for _ in range(n_aug):
            idx = np.random.randint(0, 4)
            new_digit = str(np.random.randint(0, 10))
            digits_aug = digits.copy()
            digits_aug[idx] = new_digit
            augmented.append("".join(digits_aug))
    df_aug = pd.DataFrame({"angka": angka + augmented})
    return df_aug

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

def build_lstm_model(attention=False, positional=False):
    inputs = Input(shape=(3,))
    x = Embedding(input_dim=10, output_dim=8)(inputs)
    if positional:
        x = PositionalEncoding()(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    if attention:
        attn = Attention()([x, x])
        x = Concatenate()([x, attn])
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.2)(x)
    outputs = [Dense(10, activation="softmax", name=f"output_{i}")(x) for i in range(4)]
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_and_save_lstm(df, lokasi):
    if len(df) < 20:
        return
    df_aug = augment_data(df)
    X, y = preprocess_data(df_aug)
    model = build_lstm_model(attention=True, positional=True)
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("training_logs", exist_ok=True)
    log_path = f"training_logs/history_{lokasi.lower().replace(' ', '_')}.csv"
    csv_logger = CSVLogger(log_path)
    model.fit(X, y, epochs=30, batch_size=16, verbose=0, callbacks=[csv_logger])
    model.save(f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5")

def model_exists(lokasi):
    return os.path.exists(f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5")

def apply_temperature_scaling(probs, temp=1.5):
    scaled = np.log(probs + 1e-9) / temp
    scaled = np.exp(scaled) / np.sum(np.exp(scaled))
    return scaled

def top6_lstm(df, lokasi=None, return_probs=False, temp=1.5):
    try:
        model = load_model(f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5", compile=False)
        sequences = []
        for angka in df["angka"]:
            digits = [int(d) for d in f"{int(angka):04d}"]
            sequences.append(digits[:-1])
        X = np.array(sequences)
        y_pred = model.predict(X, verbose=0)
        top6 = []
        probs = []
        for out in y_pred:
            avg_probs = np.mean(out, axis=0)
            temp_scaled = apply_temperature_scaling(avg_probs, temp=temp)
            top_idx = temp_scaled.argsort()[-6:][::-1]
            top6.append(list(top_idx))
            probs.append(temp_scaled[top_idx])
        if return_probs:
            return top6, probs
        return top6
    except:
        return None

def kombinasi_4d(df, lokasi, top_n=10):
    result, probs = top6_lstm(df, lokasi=lokasi, return_probs=True)
    if result is None:
        return []
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

def top6_ensemble(df, lokasi, weight_lstm=0.6, weight_markov=0.4):
    lstm_result = top6_lstm(df, lokasi=lokasi)
    markov_result, _ = top6_markov(df)
    if lstm_result is None or markov_result is None:
        return None
    ensemble = []
    for i in range(4):
        score_dict = {}
        for idx, digit in enumerate(lstm_result[i]):
            score_dict[digit] = score_dict.get(digit, 0) + weight_lstm * (6 - idx)
        for idx, digit in enumerate(markov_result[i]):
            score_dict[digit] = score_dict.get(digit, 0) + weight_markov * (6 - idx)
        top = sorted(score_dict.items(), key=lambda x: -x[1])[:6]
        ensemble.append([x[0] for x in top])
    return ensemble

def anti_top6_lstm(df, lokasi):
    hasil = top6_lstm(df, lokasi)
    return [[x for x in range(10) if x not in hasil[i]] for i in range(4)]

def low6_lstm(df, lokasi):
    model = load_model(f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5", compile=False)
    sequences = []
    for angka in df["angka"]:
        digits = [int(d) for d in f"{int(angka):04d}"]
        sequences.append(digits[:-1])
    X = np.array(sequences)
    y_pred = model.predict(X, verbose=0)
    low6 = []
    for out in y_pred:
        avg_probs = np.mean(out, axis=0)
        low_idx = avg_probs.argsort()[:6]
        low6.append(list(low_idx))
    return low6
