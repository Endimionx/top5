import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Reshape
from tensorflow.keras.utils import to_categorical

# Label posisi digit tetap
DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

def build_lstm4d_model(window_size):
    """
    Bangun model LSTM Bidirectional untuk prediksi 4D (output per digit).
    Output shape = (4, 10) => probabilitas 0-9 per digit.
    """
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
    """
    Ubah data df['angka'] menjadi input dan label untuk training LSTM 4D.
    """
    sequences = df['angka'].astype(str).apply(lambda x: [int(d) for d in x]).tolist()
    X, y = [], []
    for i in range(len(sequences) - window_size):
        window = sequences[i:i + window_size]
        label = sequences[i + window_size]
        X.append(window)
        y.append(label)
    X = np.array(X)  # shape: (samples, window_size, 4)
    y = np.array(y)  # shape: (samples, 4)
    y = to_categorical(y, num_classes=10)  # shape: (samples, 4, 10)
    return X, y

def train_lstm4d(df, window_size=10, epochs=15, batch_size=16, verbose=0):
    """
    Melatih model LSTM untuk prediksi 4D dari data df.
    """
    X, y = prepare_lstm4d_data(df, window_size)
    model = build_lstm4d_model(window_size)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model

def predict_lstm4d_top6_per_digit(model, df, window_size):
    """
    Prediksi angka 4D berikutnya.
    Hasil:
        - top6_per_digit: list dari top-6 prediksi tiap posisi digit
        - full_probs: probabilitas penuh 0-9 tiap posisi (bisa untuk ensemble)
    """
    sequences = df['angka'].astype(str).apply(lambda x: [int(d) for d in x]).tolist()
    if len(sequences) < window_size:
        return None, None

    latest_window = sequences[-window_size:]
    X_input = np.array([latest_window])  # shape: (1, window_size, 4)
    preds = model.predict(X_input, verbose=0)[0]  # shape: (4, 10)

    top6_per_digit = [np.argsort(p)[::-1][:6].tolist() for p in preds]
    full_probs = preds.tolist()
    return top6_per_digit, full_probs
