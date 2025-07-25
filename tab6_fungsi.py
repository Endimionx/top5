# tab6_fungsi.py

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Reshape
from tensorflow.keras.utils import to_categorical
from datetime import datetime
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

def refine_top8_with_manual(top8, manual_refs, extra_score=2.0):
    """
    Tambahkan bobot jika digit dari prediksi manual cocok dengan top8.
    """
    refined = []
    for i in range(4):  # untuk setiap posisi
        digit_scores = {}
        for rank, d in enumerate(top8[i]):
            score = (8 - rank)  # skor dasar berdasarkan posisi
            if d == manual_refs[i]:
                score += extra_score  # boost jika cocok
            digit_scores[d] = score
        # urutkan ulang berdasarkan skor tertinggi
        ranked = sorted(digit_scores.items(), key=lambda x: x[1], reverse=True)
        refined_digits = [d for d, _ in ranked[:6]]
        refined.append(refined_digits)
    return refined

def parse_manual_input(textarea_input):
    """
    Mengubah textarea string ke list digit (harus 50 baris = 49 + 1).
    """
    lines = textarea_input.strip().splitlines()
    digits = []
    for line in lines:
        try:
            d = int(line.strip())
            if 0 <= d <= 9:
                digits.append(d)
        except:
            continue
    return digits if len(digits) == 50 else None

def extract_manual_ref_per_digit(textarea_dict):
    """
    Mengembalikan list per posisi: [list_49digit, digit_ke50]
    """
    refs_49 = {}
    digit_50 = {}
    for pos in DIGIT_LABELS:
        parsed = parse_manual_input(textarea_dict.get(pos, ""))
        if parsed is None or len(parsed) != 50:
            raise ValueError(f"Posisi '{pos}' harus berisi 50 baris angka 0-9.")
        refs_49[pos] = parsed[:49]
        digit_50[pos] = parsed[49]
    return refs_49, digit_50

def save_prediction_log(result_dict, lokasi):
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"prediksi_tab6_{lokasi}_{today}.txt"
    with open(filename, "w") as f:
        f.write(f"Prediksi 4D - Lokasi: {lokasi} - Tanggal: {today}\n\n")
        for label, values in result_dict.items():
            f.write(f"{label.upper()}: {', '.join(str(v) for v in values)}\n")
    return filename
