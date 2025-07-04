import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

def prepare_lstm_data(df):
    data = df['angka'].dropna().apply(lambda x: [int(d) for d in f"{int(x):04d}"]).tolist()
    data = np.array(data)

    if len(data) < 11:
        return None, None

    X, y = [], []
    for i in range(len(data) - 10):
        X.append(data[i:i+10])
        y.append(data[i+10])
    X, y = np.array(X), np.array(y)

    if len(X) == 0 or len(y) == 0:
        return None, None

    y_encoded = np.array([to_categorical(d, num_classes=10) for d in y])
    y_encoded = y_encoded.reshape(-1, 40)

    return X, y_encoded

def model_exists(lokasi):
    filename = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"
    return os.path.exists(filename)

def train_and_save_lstm(df, lokasi):
    os.makedirs("saved_models", exist_ok=True)
    filename = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"

    X, y = prepare_lstm_data(df)
    if X is None or y is None:
        return

    if os.path.exists(filename):
        print(f"🧠 Fine-tuning model untuk {lokasi}")
        model = load_model(filename)
    else:
        print(f"🧠 Membuat model baru untuk {lokasi}")
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(10, 4)))
        model.add(Dropout(0.2))
        model.add(LSTM(64))
        model.add(Dense(40, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy')

    callbacks = [EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)]

    model.fit(X, y, epochs=50, batch_size=16, verbose=0, callbacks=callbacks)

    model.save(filename)

def top6_lstm(df, lokasi=None):
    filename = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"
    if not os.path.exists(filename):
        return None

    data = df['angka'].dropna().apply(lambda x: [int(d) for d in f"{int(x):04d}"]).tolist()
    data = np.array(data)

    if len(data) < 10:
        return None

    model = load_model(filename)
    input_seq = np.array(data[-10:]).reshape(1, 10, 4)
    pred = model.predict(input_seq, verbose=0)[0].reshape(4, 10)

    top6 = []
    for i in range(4):
        top = list(np.argsort(-pred[i])[:6])
        top6.append(top)

    return top6

def anti_top6_lstm(df, lokasi=None):
    filename = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"
    if not os.path.exists(filename):
        return None

    data = df['angka'].dropna().apply(lambda x: [int(d) for d in f"{int(x):04d}"]).tolist()
    data = np.array(data)

    if len(data) < 10:
        return None

    model = load_model(filename)
    input_seq = np.array(data[-10:]).reshape(1, 10, 4)
    pred = model.predict(input_seq, verbose=0)[0].reshape(4, 10)

    anti_top6 = []
    for i in range(4):
        sorted_idx = np.argsort(pred[i])  # ascending
        lowest6 = list(sorted_idx[:6])
        anti_top6.append(lowest6)

    return anti_top6

def low6_lstm(df, lokasi=None):
    filename = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"
    if not os.path.exists(filename):
        return None

    data = df['angka'].dropna().apply(lambda x: [int(d) for d in f"{int(x):04d}"]).tolist()
    data = np.array(data)

    if len(data) < 10:
        return None

    model = load_model(filename)
    input_seq = np.array(data[-10:]).reshape(1, 10, 4)
    pred = model.predict(input_seq, verbose=0)[0].reshape(4, 10)

    low6 = []
    for i in range(4):
        low = list(np.argsort(pred[i])[:6])
        low6.append(low)

    return low6
