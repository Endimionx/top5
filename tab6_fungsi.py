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
    y = to_categorical(y, num_classes=10)  # y.shape = (n, 4, 10)
    return X, y

def train_lstm4d(model, X, y, epoch, batch):
    model.fit(X, y, epochs=epoch, batch_size=batch, verbose=0)
    return model

def predict_lstm4d_top6_per_digit(model, df, window_size):
    sequences = df['angka'].astype(str).apply(lambda x: [int(d) for d in x]).tolist()
    if len(sequences) < window_size:
        return None, None
    latest_window = sequences[-window_size:]
    X_input = np.array([latest_window])  # (1, window_size, 4)
    preds = model.predict(X_input, verbose=0)[0]  # (4, 10)
    top6_per_digit = [np.argsort(p)[::-1][:6].tolist() for p in preds]
    return top6_per_digit, preds.tolist()  # full probs
