import numpy as np
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, LayerNormalization
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df, window_size=5):
    angka_list = [int(x) for x in df['angka']]
    X, y = [], [[] for _ in range(4)]
    for i in range(len(angka_list) - window_size):
        window = angka_list[i:i+window_size]
        target = f"{angka_list[i + window_size]:04d}"
        X.append(window)
        for j in range(4):
            y[j].append(int(target[j]))
    X = np.array(X)
    y = [np.array(label) for label in y]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled.reshape(-1, window_size, 1), y

def build_model(input_len):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(input_len, 1)))
    model.add(LayerNormalization())
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def top6_lstm(df, lokasi):
    X, _ = preprocess_data(df)
    hasil = []
    for i in range(4):
        path = f"saved_models/{lokasi.lower().replace(' ', '_')}_digit{i}.h5"
        if not os.path.exists(path):
            hasil.append([0,1,2,3,4,5])
            continue
        model = load_model(path)
        probs = model.predict(X[-1].reshape(1, X.shape[1], 1))[0]
        top6 = np.argsort(probs)[-6:][::-1]
        hasil.append(top6.tolist())
    return hasil

def kombinasi_4d(df, lokasi, top_n=10, mode="average"):
    X, _ = preprocess_data(df)
    if len(X) == 0: return []
    pred_per_digit = []
    for i in range(4):
        path = f"saved_models/{lokasi.lower().replace(' ', '_')}_digit{i}.h5"
        if not os.path.exists(path):
            pred_per_digit.append(np.ones(10)/10)
            continue
        model = load_model(path)
        prob = model.predict(X[-1].reshape(1, X.shape[1], 1))[0]
        pred_per_digit.append(prob)

    kombinasi = []
    for a in np.argsort(pred_per_digit[0])[-6:]:
        for b in np.argsort(pred_per_digit[1])[-6:]:
            for c in np.argsort(pred_per_digit[2])[-6:]:
                for d in np.argsort(pred_per_digit[3])[-6:]:
                    angka = f"{a}{b}{c}{d}"
                    if mode == "average":
                        score = (pred_per_digit[0][a] + pred_per_digit[1][b] + pred_per_digit[2][c] + pred_per_digit[3][d]) / 4
                    else:
                        score = pred_per_digit[0][a] * pred_per_digit[1][b] * pred_per_digit[2][c] * pred_per_digit[3][d]
                    kombinasi.append((angka, score))
    kombinasi = sorted(kombinasi, key=lambda x: x[1], reverse=True)[:top_n]
    return kombinasi

def top6_ensemble(df, lokasi):
    from markov_model import top6_markov_hybrid
    lstm = top6_lstm(df, lokasi)
    markov = top6_markov_hybrid(df)
    hasil = []
    for i in range(4):
        gabung = lstm[i] + markov[i]
        counter = {x: gabung.count(x) for x in set(gabung)}
        top = sorted(counter, key=lambda x: (-counter[x], x))[:6]
        hasil.append(top)
    return hasil

def model_exists(lokasi):
    for i in range(4):
        path = f"saved_models/{lokasi.lower().replace(' ', '_')}_digit{i}.h5"
        if not os.path.exists(path):
            return False
    return True
