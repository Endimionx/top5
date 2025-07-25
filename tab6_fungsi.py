# tab6_fungsi.py

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.utils import to_categorical
from collections import Counter
from datetime import datetime
import os

DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

def build_lstm_model(window_size=10):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(window_size, 1)),
        Bidirectional(LSTM(64)),
        Dense(32, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def prepare_data_for_position(df, pos, window_size=10):
    sequences = df['angka'].astype(str).apply(lambda x: int(x[pos])).tolist()
    X, y = [], []
    for i in range(len(sequences) - window_size):
        X.append(sequences[i:i+window_size])
        y.append(sequences[i+window_size])
    X = np.array(X).reshape(-1, window_size, 1)
    y = to_categorical(y, num_classes=10)
    return X, y

def train_lstm_for_position(df, pos, window_size=10, epochs=15, batch_size=16):
    X, y = prepare_data_for_position(df, pos, window_size)
    model = build_lstm_model(window_size)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

def predict_top8_per_position(model, sequence, window_size=10):
    if len(sequence) < window_size:
        return []
    X_input = np.array(sequence[-window_size:]).reshape(1, window_size, 1)
    probs = model.predict(X_input, verbose=0)[0]
    top8 = np.argsort(probs)[::-1][:8].tolist()
    return top8, probs.tolist()

def parse_manual_8digit_input(textarea_input):
    """
    Mengubah input textarea menjadi list of list of digits.
    Setiap baris harus 8 digit.
    """
    lines = textarea_input.strip().splitlines()
    digits = []
    for line in lines:
        line = line.strip()
        if len(line) == 8 and line.isdigit():
            digits.append([int(d) for d in line])
    return digits if len(digits) >= 49 else None

def extract_frequencies_8digit(data_8digit, pos):
    """
    Ambil frekuensi digit di posisi tertentu dari 49 baris awal.
    """
    kolom = [baris[pos] for baris in data_8digit[:49]]
    return Counter(kolom)

def refine_prediction(top8_list, prob_list, freq_counter, freq_weight=0.3, prob_weight=0.7):
    """
    Gabungkan skor berdasarkan probabilitas dan frekuensi referensi.
    """
    scores = {}
    for i, d in enumerate(top8_list):
        prob_score = prob_list[d]
        freq_score = freq_counter.get(d, 0)
        scores[d] = (prob_score * prob_weight) + (freq_score * freq_weight)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:6]]

def save_prediction_log(result_dict, lokasi):
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"prediksi_tab6_{lokasi}_{today}.txt"
    with open(filename, "w") as f:
        f.write(f"Prediksi 4D - Lokasi: {lokasi} - Tanggal: {today}\n\n")
        for label, values in result_dict.items():
            f.write(f"{label.upper()}: {', '.join(str(v) for v in values)}\n")
    return filename
