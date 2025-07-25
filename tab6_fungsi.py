# tab6_fungsi.py

import numpy as np
from collections import Counter
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os

DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

def parse_reference_input(textarea_input):
    """
    Mengubah textarea 8-digit per baris menjadi array shape (49, 8).
    """
    lines = textarea_input.strip().splitlines()
    digits = []
    for line in lines:
        line = line.strip()
        if len(line) == 8 and line.isdigit():
            digits.append([int(d) for d in line])
    return np.array(digits) if len(digits) >= 49 else None

def prepare_X_y_from_ref_and_df(ref_array, df, posisi_digit):
    """
    Dari ref (49 x 8) dan df (target), siapkan X, y.
    posisi_digit: 0=ribuan, dst
    """
    if len(df) < 49:
        return None, None
    X = ref_array[:49]
    y_all = df["angka"].astype(str).apply(lambda x: int(x[posisi_digit])).values[-49:]
    return np.array(X), np.array(y_all)

def train_digit_model(X, y):
    """
    Membangun model sederhana MLP dan melatihnya.
    """
    y_cat = np.eye(10)[y]
    model = Sequential([
        Dense(64, activation='relu', input_shape=(8,)),
        Dense(32, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y_cat, epochs=25, verbose=0)
    return model

def predict_top6(model, ref_array):
    """
    Prediksi digit besok berdasarkan baris ke-49 dari ref_array.
    """
    input_pred = ref_array[-1].reshape(1, 8)
    probs = model.predict(input_pred, verbose=0)[0]
    top6 = np.argsort(probs)[::-1][:6]
    return top6.tolist(), probs.tolist()

def save_prediction_log(result_dict, lokasi):
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"prediksi_tab6_{lokasi}_{today}.txt"
    with open(filename, "w") as f:
        f.write(f"Prediksi 4D - Lokasi: {lokasi} - Tanggal: {today}\n\n")
        for label, values in result_dict.items():
            f.write(f"{label.upper()}: {', '.join(str(v) for v in values)}\n")
    return filename
