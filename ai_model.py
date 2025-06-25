
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def top5_lstm(df):
    data = df['angka'].dropna().apply(lambda x: [int(d) for d in f"{int(x):04d}"]).tolist()
    data = np.array(data)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(data_scaled) - 10):
        X.append(data_scaled[i:i+10])
        y.append(data_scaled[i+10])
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(4, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=30, verbose=0)

    input_seq = data_scaled[-10:].reshape(1, 10, 4)
    pred_scaled = model.predict(input_seq)[0]
    pred_unscaled = scaler.inverse_transform([pred_scaled])[0]

    top5 = []
    for val in pred_unscaled:
        dists = [(i, abs(val - i)) for i in range(10)]
        dists.sort(key=lambda x: x[1])
        top5.append([i[0] for i in dists[:5]])
    return top5
