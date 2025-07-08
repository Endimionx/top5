import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

def preprocess_data(df, window_size=5):
    angka_list = [int(x) for x in df["angka"] if len(x) == 4]
    X, y = [], [[], [], [], []]
    for i in range(len(angka_list) - window_size):
        window = angka_list[i:i + window_size]
        target = f"{angka_list[i + window_size]:04d}"
        X.append([list(map(int, f"{num:04d}")) for num in window])
        for d in range(4):
            y[d].append(int(target[d]))
    X = np.array(X)
    y = [to_categorical(np.array(yd), num_classes=10) for yd in y]
    return X, y

def build_model(input_len):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(input_len, 4)))
    model.add(Dropout(0.3))
    model.add(LayerNormalization())
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer=Adam(0.001), metrics=["accuracy"])
    return model

def model_exists(lokasi):
    for i in range(4):
        path = f"saved_models/{lokasi.lower().replace(' ', '_')}_digit{i}.h5"
        if not os.path.exists(path):
            return False
    return True

def load_digit_model(lokasi, digit_index):
    path = f"saved_models/{lokasi.lower().replace(' ', '_')}_digit{digit_index}.h5"
    return load_model(path) if os.path.exists(path) else None

def top6_lstm(df, lokasi, return_probs=False):
    if len(df) < 30:
        return [[0]*6 for _ in range(4)]
    X, _ = preprocess_data(df)
    X_last = X[-1:]
    hasil = []
    confidences = []

    for i in range(4):
        model = load_digit_model(lokasi, i)
        if model is None:
            hasil.append([0]*6)
            confidences.append([0]*10)
            continue
        probas = model.predict(X_last, verbose=0)[0]
        top_indices = probas.argsort()[-6:][::-1]
        hasil.append(top_indices.tolist())
        confidences.append(probas.tolist())

    return (hasil, confidences) if return_probs else hasil

def kombinasi_4d(df, lokasi, top_n=10, mode="average"):
    pred, confs = top6_lstm(df, lokasi, return_probs=True)
    kombinasi = []
    for a in pred[0]:
        for b in pred[1]:
            for c in pred[2]:
                for d in pred[3]:
                    angka = f"{a}{b}{c}{d}"
                    if mode == "product":
                        score = (
                            confs[0][a] *
                            confs[1][b] *
                            confs[2][c] *
                            confs[3][d]
                        )
                    else:
                        score = (
                            confs[0][a] +
                            confs[1][b] +
                            confs[2][c] +
                            confs[3][d]
                        ) / 4
                    kombinasi.append((angka, score))

    kombinasi.sort(key=lambda x: -x[1])
    return kombinasi[:top_n]

def top6_ensemble(df, lokasi):
    hasil_lstm = top6_lstm(df, lokasi)
    hasil_markov, _ = top6_markov(df)
    hasil = []
    for i in range(4):
        combined = hasil_lstm[i] + hasil_markov[i]
        counter = {}
        for d in combined:
            counter[d] = counter.get(d, 0) + 1
        sorted_digit = sorted(counter.items(), key=lambda x: -x[1])
        top6 = [d for d, _ in sorted_digit[:6]]
        if len(top6) < 6:
            top6 += [x for x in range(10) if x not in top6][:6 - len(top6)]
        hasil.append(top6)
    return hasil
