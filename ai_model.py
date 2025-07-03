import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

MODEL_DIR = "saved_models"

def top5_lstm(df, lokasi="default"):
    try:
        data = df['angka'].dropna().apply(lambda x: [int(d) for d in f"{int(x):04d}"]).tolist()
        data = np.array(data)
        if len(data) < 11:
            print("[DEBUG] Data kurang dari 11")
            return None

        X, y = [], []
        for i in range(len(data) - 10):
            X.append(data[i:i+10])
            y.append(data[i+10])
        X = np.array(X)
        y = np.array(y)

        if len(X) == 0 or len(y) == 0:
            print("[DEBUG] X atau y kosong")
            return None

        model_path = os.path.join(MODEL_DIR, f"lstm_{lokasi.lower().replace(' ', '_')}.h5")
        if not os.path.exists(model_path):
            print(f"[DEBUG] Model tidak ditemukan: {model_path}")
            return None

        model = load_model(model_path)
        input_seq = np.array(data[-10:]).reshape(1, 10, 4)
        pred = model.predict(input_seq, verbose=0)
        pred = pred.reshape(4, 10)

        top5 = []
        for i in range(4):
            top = list(np.argsort(-pred[i])[:5])
            top5.append(top)

        print("[DEBUG] Prediksi berhasil:", top5)
        return top5

    except Exception as e:
        print(f"[ERROR] top5_lstm gagal: {e}")
        return None
