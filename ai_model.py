import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from pathlib import Path

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
    X = np.array(X)
    y_encoded = np.array([to_categorical(d, num_classes=10) for d in y])
    y_encoded = y_encoded.reshape(-1, 40)
    return X, y_encoded

def train_and_save_lstm(df, lokasi):
    X, y = prepare_lstm_data(df)
    if X is None: return

    model = Sequential()
    model.add(LSTM(128, return_sequences=False, input_shape=(10, 4)))
    model.add(Dense(40, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(X, y, epochs=50, batch_size=16, verbose=0)
    model.save(MODEL_DIR / f"model_lstm_{lokasi.lower().replace(' ', '_')}.h5")

def top5_lstm(df, lokasi, use_saved_model=False):
    data = df['angka'].dropna().apply(lambda x: [int(d) for d in f"{int(x):04d}"]).tolist()
    data = np.array(data)
    if len(data) < 11:
        return None

    input_seq = np.array(data[-10:]).reshape(1, 10, 4)

    model_path = MODEL_DIR / f"model_lstm_{lokasi.lower().replace(' ', '_')}.h5"

    if use_saved_model and model_path.exists():
        model = load_model(model_path)
    else:
        X, y = prepare_lstm_data(df)
        if X is None: return None
        model = Sequential()
        model.add(LSTM(128, return_sequences=False, input_shape=(10, 4)))
        model.add(Dense(40, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        model.fit(X, y, epochs=50, batch_size=16, verbose=0)

    pred = model.predict(input_seq, verbose=0)[0].reshape(4, 10)
    top5 = [list(np.argsort(-pred[i])[:5]) for i in range(4)]
    return top5
