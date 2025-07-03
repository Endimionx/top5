import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

def top5_lstm(df):
    data = df['angka'].dropna().apply(lambda x: [int(d) for d in f"{int(x):04d}"]).tolist()
    data = np.array(data)

    if len(data) < 11:
        return None

    X, y = [], []
    for i in range(len(data) - 10):
        X.append(data[i:i+10])
        y.append(data[i+10])
    X, y = np.array(X), np.array(y)

    if len(X) == 0 or len(y) == 0:
        return None

    y_encoded = np.array([to_categorical(d, num_classes=10) for d in y])
    y_encoded = y_encoded.reshape(-1, 40)

    model = Sequential()
    model.add(LSTM(128, return_sequences=False, input_shape=(10, 4)))
    model.add(Dense(40, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    val_split = 0.1 if len(X) >= 10 else 0
    model.fit(X, y_encoded, epochs=50, batch_size=16, verbose=0, validation_split=val_split)

    input_seq = np.array(data[-10:]).reshape(1, 10, 4)
    pred = model.predict(input_seq, verbose=0)[0].reshape(4, 10)

    top5 = []
    for i in range(4):
        top = list(np.argsort(-pred[i])[:5])
        top5.append(top)

    return top5
