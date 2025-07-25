# tab6_fungsi.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Reshape
from tensorflow.keras.utils import to_categorical

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

def train_lstm4d(model, X, y, epochs=50, batch_size=16):
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

def predict_lstm4d_top6_per_digit(model, df, window_size=10, top_k=6):
    sequences = df['angka'].astype(str).apply(lambda x: [int(d) for d in x]).tolist()
    if len(sequences) < window_size:
        return None, None
    latest_window = sequences[-window_size:]
    X_input = np.array([latest_window])  # shape: (1, window_size, 4)
    preds = model.predict(X_input, verbose=0)[0]  # shape: (4, 10)
    top_k_per_digit = [np.argsort(p)[::-1][:top_k].tolist() for p in preds]
    full_probs = preds.tolist()
    return top_k_per_digit, full_probs

def match_top_with_reference(top_digits, reference):
    """Cek apakah salah satu top digit masuk dalam referensi."""
    return any(d in reference for d in top_digits if isinstance(d, int))
