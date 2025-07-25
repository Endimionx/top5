# tab6_fungsi.py

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Reshape
from tensorflow.keras.utils import to_categorical
from datetime import datetime
import os
from collections import Counter

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
    Mengubah textarea string ke list of list of digits (50 baris, 8 digit per baris).
    """
    lines = textarea_input.strip().splitlines()
    digits = []
    for line in lines:
        line = line.strip()
        if len(line) == 8 and line.isdigit():
            digits.append([int(d) for d in line])
    return digits if len(digits) >= 49 else None

def extract_digit_pattern_from_8digit_block(data8digit):
    """
    Ambil 49 baris pertama dan hitung frekuensi digit per posisi prediksi (misalnya ribuan saja).
    """
    ref_49 = data8digit[:49]  # exclude baris terakhir
    pattern_counter = Counter()
    for row in ref_49:
        for digit in row:
            pattern_counter[digit] += 1
    return pattern_counter

def refine_top8_with_patterns(top8, pattern_refs, extra_score=1.5):
    """
    Tambahkan bobot berdasarkan pola dari prediksi tepat 49 hari untuk masing-masing posisi.
    """
    refined = []
    for i in range(4):  # untuk ribuan, ratusan, dst
        digit_scores = {}
        for rank, d in enumerate(top8[i]):
            score = (8 - rank)  # base score
            score += pattern_refs[i].get(d, 0) * 0.25  # bobot dari frekuensi kemunculan
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
