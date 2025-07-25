import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict, Counter
from datetime import datetime

DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

def parse_reference_input(textarea):
    """
    Konversi TextArea (50 baris, masing-masing 8 digit) ke list of list.
    """
    lines = textarea.strip().splitlines()
    data = []
    for line in lines:
        line = line.strip()
        if len(line) == 8 and line.isdigit():
            data.append([int(d) for d in line])
    return data if len(data) == 50 else None

def extract_target_from_df(df, posisi_index):
    """
    Mengambil target digit dari df berdasarkan posisi:
    0=ribuan, 1=ratusan, 2=puluhan, 3=satuan
    """
    digits = df["angka"].astype(str).apply(lambda x: int(x[posisi_index])).values
    return digits[-49:]  # hanya 49 baris terakhir

def train_model_per_posisi(X, y):
    """
    Model sederhana: RandomForestClassifier
    """
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model

def predict_last_row(reference_50, model):
    """
    Prediksi baris ke-50 dari referensi.
    """
    last_row = [reference_50[-1]]  # hanya 1 sample
    return model.predict(last_row)[0]

def save_prediction_log(result_dict, lokasi):
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"prediksi_tab6_modeB_{lokasi}_{today}.txt"
    with open(filename, "w") as f:
        f.write(f"Prediksi 4D (Mode B) - Lokasi: {lokasi} - Tanggal: {today}\n\n")
        for label, value in result_dict.items():
            f.write(f"{label.upper()}: {value}\n")
    return filename
