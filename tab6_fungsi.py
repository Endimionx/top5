# tab6_fungsi.py

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Reshape
from tensorflow.keras.utils import to_categorical
from datetime import datetime
from collections import Counter
import os

DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

def build_lstm4d_model(window_size=10):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(window_size, 4)),
        Bidirectional(LSTM(64)),
        Dense(40, activation='relu'),
        Dense(4 * 10, activation='softmax'),
        Reshape((4, 10))
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def prepare_lstm4d_data(df, window_size=10):
    sequences = df['angka'].astype(str).apply(lambda x: [int(d) for d in x]).tolist()
    X, y = [], []
    for i in range(len(sequences) - window_size):
        window = sequences[i:i + window_size]
        label = sequences[i + window_size]
        X.append(window)
        y.append(label)
    X = np.array(X)
    y = to_categorical(y, num_classes=10)
    return X, y

def train_lstm4d(df, window_size=10, epochs=15, batch_size=16):
    X, y = prepare_lstm4d_data(df, window_size)
    model = build_lstm4d_model(window_size)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

def predict_lstm4d_top8(model, df, window_size=10):
    sequences = df['angka'].astype(str).apply(lambda x: [int(d) for d in x]).tolist()
    if len(sequences) < window_size:
        return None, None
    latest_window = sequences[-window_size:]
    X_input = np.array([latest_window])
    preds = model.predict(X_input, verbose=0)[0]  # shape: (4, 10)
    top8_per_digit = [np.argsort(p)[::-1][:8].tolist() for p in preds]
    full_probs = preds.tolist()
    return top8_per_digit, full_probs

def parse_manual_input(textarea_input):
    """
    Mengubah textarea string ke list of list (masing-masing 8 digit per baris).
    """
    lines = textarea_input.strip().splitlines()
    digits = []
    for line in lines:
        line = line.strip()
        if len(line) == 8 and line.isdigit():
            digits.append([int(d) for d in line])
    return digits if len(digits) >= 49 else None

def extract_digit_patterns_flat_per_digit(digits_49):
    """
    Ambil pola frekuensi untuk setiap posisi (ribuan, ratusan, puluhan, satuan)
    dari 8 digit prediksi yang tepat selama 49 hari terakhir.
    """
    pola_list = []
    for i in range(4):  # ribuan sampai satuan
        digits_i = []
        for baris in digits_49[:49]:  # hanya 49 baris pertama
            digits_i.append(baris[i])  # ambil posisi ke-i
        pola_list.append(Counter(digits_i))
    return pola_list

def refine_top8_with_patterns(top8, pola_refs, extra_score=2.0):
    """
    Tambahkan bobot jika cocok dengan pola referensi.
    """
    refined = []
    for i in range(4):
        digit_scores = {}
        for rank, d in enumerate(top8[i]):
            score = (8 - rank)
            score += pola_refs[i].get(d, 0) * 0.2
            digit_scores[d] = score
        ranked = sorted(digit_scores.items(), key=lambda x: x[1], reverse=True)
        refined_digits = [d for d, _ in ranked[:6]]
        refined.append(refined_digits)
    return refined

def save_prediction_log(result_dict, lokasi):
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"prediksi_tab6_{lokasi}_{today}.txt"
    with open(filename, "w") as f:
        f.write(f"Prediksi 4D - Lokasi: {lokasi} - Tanggal: {today}\n\n")
        for label, values in result_dict.items():
            f.write(f"{label.upper()}: {', '.join(str(v) for v in values)}\n")
    return filename
