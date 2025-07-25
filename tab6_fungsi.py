# tab6_fungsi.py

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.utils import to_categorical
from datetime import datetime
import os

DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

def parse_reference_input(text):
    """
    Mengubah text area menjadi array shape (49, 8)
    """
    lines = text.strip().splitlines()
    digits = []
    for line in lines:
        line = line.strip()
        if len(line) == 8 and line.isdigit():
            digits.append([int(c) for c in line])
    return np.array(digits) if len(digits) == 49 else None

def get_target_digit_from_df(df, posisi):
    """
    Ambil target digit dari df[-1], posisi: 0=ribuan, 1=ratusan, dst
    """
    target = str(df.iloc[-1]["angka"]).zfill(4)
    if len(target) != 4:
        return None
    return int(target[posisi])

def build_model(input_shape=(49, 8)):
    model = Sequential([
        Bidirectional(LSTM(32, return_sequences=False), input_shape=input_shape),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_and_predict_top6(X_raw, y_digit, epochs=50, batch_size=16):
    """
    Melatih model dan menghasilkan top6 prediksi digit
    """
    # X shape: (49, 8) → tambah batch dimension: (1, 49, 8)
    X = np.expand_dims(X_raw, axis=0)
    y_cat = to_categorical([y_digit], num_classes=10)
    model = build_model(input_shape=(49, 8))
    model.fit(X, y_cat, epochs=epochs, batch_size=1, verbose=0)
    pred = model.predict(X, verbose=0)[0]
    top6 = np.argsort(pred)[::-1][:6].tolist()
    return top6, pred.tolist()

def save_prediction_log(result_dict, lokasi):
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"prediksi_tab6_{lokasi}_{today}.txt"
    with open(filename, "w") as f:
        f.write(f"Prediksi 4D - Lokasi: {lokasi} - Tanggal: {today}\n\n")
        for label, values in result_dict.items():
            f.write(f"{label.upper()}: {', '.join(str(v) for v in values)}\n")
    return filename
