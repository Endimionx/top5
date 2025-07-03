import numpy as np
from pathlib import Path
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(exist_ok=True)

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

    y_encoded = np.array([to_categorical(d, num_classes=10) for d in y])
    y_encoded = y_encoded.reshape(-1, 40)
    return X, y_encoded

def model_exists(lokasi):
    model_path = MODEL_DIR / f"model_lstm_{lokasi.lower().replace(' ', '_')}.h5"
    return model_path.exists()

def train_and_save_lstm(df, lokasi):
    X, y = prepare_lstm_data(df)
    if X is None or y is None:
        return None

    model = Sequential()
    model.add(LSTM(128, input_shape=(10, 4)))
    model.add(Dense(40, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(X, y, epochs=50, batch_size=16, verbose=0)

    model_path = MODEL_DIR / f"model_lstm_{lokasi.lower().replace(' ', '_')}.h5"
    model.save(model_path)
    return model

def top5_lstm(df, lokasi=None):
    data = df['angka'].dropna().apply(lambda x: [int(d) for d in f"{int(x):04d}"]).tolist()
    data = np.array(data)

    if len(data) < 11:
        return None

    X = []
    for i in range(len(data) - 10):
        X.append(data[i:i+10])
    X = np.array(X)

    if len(X) == 0:
        return None

    input_seq = np.array(data[-10:]).reshape(1, 10, 4)

    model = None
    if lokasi:
        model_path = MODEL_DIR / f"model_lstm_{lokasi.lower().replace(' ', '_')}.h5"
        if not model_path.exists():
            return None
        model = load_model(model_path)
    else:
        # Latih instan jika lokasi tidak diberikan
        _, y = prepare_lstm_data(df)
        if y is None:
            return None
        model = Sequential()
        model.add(LSTM(128, input_shape=(10, 4)))
        model.add(Dense(40, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        model.fit(X, y, epochs=50, batch_size=16, verbose=0)

    pred = model.predict(input_seq, verbose=0)[0].reshape(4, 10)
    top5 = [list(np.argsort(-pred[i])[:5]) for i in range(4)]
    return top5
