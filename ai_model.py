import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.utils import to_categorical

# Preprocess untuk model per digit, sliding window input
def preprocess_data(df, window_size=5):
    angka_list = [int(x) for x in df['angka'] if len(x) == 4 and x.isdigit()]
    X, y = [], [[] for _ in range(4)]
    for i in range(len(angka_list) - window_size):
        window = [int(d) for n in angka_list[i:i+window_size] for d in f"{n:04d}"]
        label = f"{angka_list[i+window_size]:04d}"
        X.append(window)
        for j in range(4):
            y[j].append(int(label[j]))
    X = np.array(X) % 10  # pastikan hanya digit 0-9
    y = [to_categorical(np.array(yy), num_classes=10) for yy in y]
    return X, y

# Bangun model LSTM sederhana untuk prediksi digit
def build_model(input_len=20):
    model = Sequential([
        Embedding(input_dim=10, output_dim=16, input_length=input_len),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Cek apakah model digit sudah ada
def model_exists(lokasi, digit_index):
    model_path = f"saved_models/{lokasi.lower().replace(' ', '_')}_digit{digit_index}.h5"
    return os.path.exists(model_path)

# Fungsi prediksi top-6 untuk LSTM per digit
def top6_lstm(df, lokasi, return_probs=False):
    X, _ = preprocess_data(df)
    X_last = X[-1:].reshape(1, -1)
    hasil = []
    probas = []

    for i in range(4):
        model_path = f"saved_models/{lokasi.lower().replace(' ', '_')}_digit{i}.h5"
        if not os.path.exists(model_path):
            hasil.append([0])
            probas.append([0.0]*10)
            continue
        model = load_model(model_path)
        pred = model.predict(X_last, verbose=0)[0]
        top6 = np.argsort(pred)[-6:][::-1].tolist()
        hasil.append(top6)
        probas.append(pred.tolist())
    
    if return_probs:
        return hasil, probas
    return hasil

# Kombinasi 4D dari hasil prediksi 4 digit
def kombinasi_4d(df, lokasi, top_n=10, mode="average"):
    hasil, probas = top6_lstm(df, lokasi, return_probs=True)
    from itertools import product

    kombinasi = []
    for i, r in enumerate(product(*hasil)):
        nilai = 0.0
        for j in range(4):
            nilai += probas[j][r[j]] if mode == "average" else np.log(probas[j][r[j]] + 1e-9)
        kombinasi.append((''.join(map(str, r)), nilai))
    
    kombinasi.sort(key=lambda x: x[1], reverse=True)
    return kombinasi[:top_n]

# Ensemble prediksi (gabungan Markov + LSTM jika ingin dikembangkan)
def top6_ensemble(df, lokasi):
    # Sementara: hanya gunakan LSTM (karena Markov digabung di app.py)
    return top6_lstm(df, lokasi)
