# tab6_fungsi.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Reshape
from tensorflow.keras.utils import to_categorical
from itertools import product
import os
from datetime import datetime

DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

def build_lstm4d_model(window_size):
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

def train_lstm4d(model, X, y, epochs=20, batch_size=32):
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

def predict_lstm4d_top6_per_digit(model, df, window_size):
    sequences = df['angka'].astype(str).apply(lambda x: [int(d) for d in x]).tolist()
    if len(sequences) < window_size:
        return None, None
    latest_window = sequences[-window_size:]
    X_input = np.array([latest_window])
    preds = model.predict(X_input, verbose=0)[0]  # shape: (4, 10)
    top6_per_digit = [np.argsort(p)[::-1][:6].tolist() for p in preds]
    full_probs = preds.tolist()
    return top6_per_digit, full_probs

def generate_all_4d_combinations(prediksi_top6):
    all_4d = list(product(*prediksi_top6))  # list of tuple (r, r, p, s)
    return [''.join(str(d) for d in tup) for tup in all_4d]

def filter_by_reference_8digit(all_4d, data_ref):
    if not data_ref:
        return all_4d
    return [angka for angka in all_4d if any(angka in ref for ref in data_ref)]

def save_prediction_to_txt(hasil_angka4d, lokasi="lokasi_default", note=None):
    """Simpan hasil prediksi ke file txt berdasarkan lokasi dan tanggal"""
    if not hasil_angka4d:
        return

    today = datetime.today().strftime("%Y-%m-%d")
    os.makedirs("prediksi_output", exist_ok=True)
    filename = f"prediksi_output/{lokasi.lower()}_{today}.txt"

    with open(filename, "w") as f:
        f.write(f"📌 Prediksi 4D\nLokasi: {lokasi}\nTanggal: {today}\n")
        if note:
            f.write(f"Catatan: {note}\n")
        f.write(f"Total Kombinasi: {len(hasil_angka4d)}\n")
        f.write("=" * 40 + "\n")
        for angka in hasil_angka4d:
            f.write(f"{angka}\n")
    print(f"[Saved] {filename}")
