import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from collections import Counter

window_size = 5

def preprocess_data(df):
    angka = df["angka"].astype(str).str.zfill(4).tolist()
    sequences = [list(map(int, a)) for a in angka]
    X, y_digits = [], [[] for _ in range(4)]

    for i in range(window_size, len(sequences)):
        window = sequences[i - window_size:i]
        flat_window = [d for seq in window for d in seq]
        target = sequences[i]
        X.append(flat_window)
        for j in range(4):
            y_digits[j].append(target[j])

    X = np.array(X) % 10
    X = X.reshape(-1, window_size * 4, 1)
    y_digits = [to_categorical(y, num_classes=10) for y in y_digits]
    return X, y_digits

def build_model(input_len=20):
    model = Sequential([
        LSTM(64, input_shape=(input_len, 1), return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def model_exists(lokasi):
    lokasi = lokasi.lower().replace(' ', '_')
    return all(os.path.exists(f"saved_models/{lokasi}_digit{i}.h5") for i in range(4))

def top6_lstm(df, lokasi):
    lokasi = lokasi.lower().replace(' ', '_')
    X, _ = preprocess_data(df)
    X_last = X[-1:].reshape(1, X.shape[1], 1)
    hasil = []

    for i in range(4):
        model_path = f"saved_models/{lokasi}_digit{i}.h5"
        if not os.path.exists(model_path):
            hasil.append(list(range(10)))
            continue
        model = load_model(model_path)
        probas = model.predict(X_last, verbose=0)[0]
        top_indices = np.argsort(probas)[-6:][::-1]
        hasil.append(list(top_indices))

    return hasil

def kombinasi_4d(df, lokasi, top_n=10, mode="average"):
    top6 = top6_lstm(df, lokasi=lokasi)
    lokasi = lokasi.lower().replace(' ', '_')
    X, _ = preprocess_data(df)
    X_last = X[-1:].reshape(1, X.shape[1], 1)

    probas_list = []
    for i in range(4):
        model_path = f"saved_models/{lokasi}_digit{i}.h5"
        if not os.path.exists(model_path):
            probas_list.append(np.ones(10) / 10)
            continue
        model = load_model(model_path)
        probas = model.predict(X_last, verbose=0)[0]
        probas_list.append(probas)

    kombinasi = []
    for a in top6[0]:
        for b in top6[1]:
            for c in top6[2]:
                for d in top6[3]:
                    score = (
                        probas_list[0][a] *
                        probas_list[1][b] *
                        probas_list[2][c] *
                        probas_list[3][d]
                        if mode == "product"
                        else np.mean([
                            probas_list[0][a],
                            probas_list[1][b],
                            probas_list[2][c],
                            probas_list[3][d]
                        ])
                    )
                    kombinasi.append((f"{a}{b}{c}{d}", score))

    kombinasi.sort(key=lambda x: x[1], reverse=True)
    return kombinasi[:top_n]

def top6_ensemble(df, lokasi):
    from markov_model import top6_markov_hybrid
    pred_lstm = top6_lstm(df, lokasi)
    pred_markov = top6_markov_hybrid(df)
    hasil = []

    for i in range(4):
        gabung = pred_lstm[i] + pred_markov[i]
        top = [x for x, _ in Counter(gabung).most_common(6)]
        hasil.append(top)

    return hasil
