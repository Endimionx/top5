import numpy as np
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ======================= PREPARE =======================

def prepare_lstm_data(df):
    data = df['angka'].dropna().apply(lambda x: [int(d) for d in f"{int(x):04d}"]).tolist()
    data = np.array(data)
    if len(data) < 11:
        return None, None
    X, y = [], []
    for i in range(len(data) - 10):
        X.append(data[i:i+10])
        y.append(data[i+10])
    return np.array(X), np.array(y)

# ======================= LATIH MODEL =======================

def train_lstm_model(X, y):
    y_encoded = np.array([to_categorical(d, num_classes=10) for d in y])
    y_encoded = y_encoded.reshape(-1, 40)

    model = Sequential()
    model.add(LSTM(128, input_shape=(10, 4)))
    model.add(Dense(40, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    model.fit(X, y_encoded, epochs=50, batch_size=16, verbose=0, validation_split=0.1)
    return model

# ======================= SIMPAN & LOAD =======================

def save_model(model, lokasi):
    model.save(os.path.join(MODEL_DIR, f"model_lstm_{lokasi}.h5"))

def load_saved_model(lokasi):
    path = os.path.join(MODEL_DIR, f"model_lstm_{lokasi}.h5")
    return load_model(path) if os.path.exists(path) else None

# ======================= PREDIKSI =======================

def top5_lstm(df, lokasi="default"):
    X, y = prepare_lstm_data(df)
    if X is None or y is None or len(X) == 0:
        return None

    model = load_saved_model(lokasi)
    if model is None:
        model = train_lstm_model(X, y)
        save_model(model, lokasi)

    input_seq = X[-1].reshape(1, 10, 4)
    pred = model.predict(input_seq, verbose=0)[0].reshape(4, 10)

    top5 = []
    for i in range(4):
        top = list(np.argsort(-pred[i])[:5])
        top5.append(top)

    return top5
