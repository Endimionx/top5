import numpy as np
from itertools import product
from datetime import datetime
import os

DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

def build_lstm4d_model(window_size):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Reshape
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(window_size, 4)),
        Bidirectional(LSTM(64)),
        Dense(40, activation='relu'),
        Dense(4 * 10, activation='softmax'),
        Reshape((4, 10))
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def prepare_lstm4d_data(df, window_size):
    from tensorflow.keras.utils import to_categorical
    sequences = df['angka'].astype(str).apply(lambda x: [int(d) for d in x]).tolist()
    X, y = [], []
    for i in range(len(sequences) - window_size):
        window = sequences[i:i + window_size]
        label = sequences[i + window_size]
        X.append(window)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    y = np.array([to_categorical(label, num_classes=10) for label in y])
    return X, y

def train_lstm4d(model, X, y, epochs=20, batch_size=32):
    return model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0).model

def predict_lstm4d_topk_per_digit(model, df, window_size, top_k=8):
    sequences = df['angka'].astype(str).apply(lambda x: [int(d) for d in x]).tolist()
    if len(sequences) < window_size:
        return None, None
    latest_window = sequences[-window_size:]
    X_input = np.array([latest_window])
    preds = model.predict(X_input, verbose=0)[0]  # shape: (4, 10)
    topk_digits = [np.argsort(p)[::-1][:top_k].tolist() for p in preds]
    return topk_digits, preds.tolist()

def generate_all_4d_combinations_with_probs(topk_digits, probs):
    all_4d = []
    for comb in product(*topk_digits):  # (d1, d2, d3, d4)
        prob_sum = sum([probs[i][d] for i, d in enumerate(comb)])
        angka = "".join(str(d) for d in comb)
        all_4d.append((angka, prob_sum))
    return all_4d

def filter_and_rank_by_reference(all_4d_with_scores, ref_dict):
    result = []
    for angka, score in all_4d_with_scores:
        cocok = True
        for i, label in enumerate(DIGIT_LABELS):
            if ref_dict[label] and int(angka[i]) not in ref_dict[label]:
                cocok = False
                break
        if cocok:
            result.append((angka, score))
    return sorted(result, key=lambda x: -x[1])

def save_prediction_to_txt(pred_list, lokasi, note=""):
    os.makedirs("prediksi_output", exist_ok=True)
    tgl = datetime.now().strftime("%Y-%m-%d")
    filename = f"prediksi_output/{lokasi}_{tgl}.txt"
    with open(filename, "w") as f:
        f.write(f"Prediksi {tgl} - Lokasi: {lokasi}\n")
        if note:
            f.write(note + "\n")
        for angka in pred_list:
            f.write(f"{angka}\n")
