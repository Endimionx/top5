import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

def model_exists(lokasi):
    model_path = os.path.join(MODEL_DIR, f"lstm_{lokasi.lower().replace(' ', '_')}.h5")
    return os.path.exists(model_path)

def train_and_save_lstm(df, lokasi="default"):
    data = df['angka'].dropna().apply(lambda x: [int(d) for d in f"{int(x):04d}"]).tolist()
    data = np.array(data)
    if len(data) < 20: return None

    X, y = [], []
    for i in range(len(data) - 10):
        X.append(data[i:i+10])
        y.append(data[i+10])
    X, y = np.array(X), np.array(y)

    y_encoded = np.array([to_categorical(d, num_classes=10) for d in y])
    y_encoded = y_encoded.reshape(-1, 40)

    model = Sequential()
    model.add(LSTM(128, return_sequences=False, input_shape=(10, 4)))
    model.add(Dense(40, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(X, y_encoded, epochs=50, batch_size=16, verbose=0)

    model_path = os.path.join(MODEL_DIR, f"lstm_{lokasi.lower().replace(' ', '_')}.h5")
    model.save(model_path)

def top5_lstm(df, lokasi="default"):
    try:
        data = df['angka'].dropna().apply(lambda x: [int(d) for d in f"{int(x):04d}"]).tolist()
        data = np.array(data)
        if len(data) < 11: return None

        model_path = os.path.join(MODEL_DIR, f"lstm_{lokasi.lower().replace(' ', '_')}.h5")
        if not os.path.exists(model_path): return None

        model = load_model(model_path)
        input_seq = np.array(data[-10:]).reshape(1, 10, 4)
        pred = model.predict(input_seq, verbose=0).reshape(4, 10)

        top5 = []
        for i in range(4):
            top = list(np.argsort(-pred[i])[:5])
            top5.append(top)
        return top5
    except Exception as e:
        print(f"[ERROR top5_lstm]: {e}")
        return None
