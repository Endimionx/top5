# ai_model.py

import numpy as np
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

@st.cache_resource
def train_lstm_model(X, y):
    model = Sequential()
    model.add(LSTM(128, return_sequences=False, input_shape=(10, 4)))
    model.add(Dense(40, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(X, y, epochs=50, batch_size=16, verbose=0, validation_split=0.1)
    return model

def prepare_lstm_data(df):
    data = df['angka'].dropna().apply(lambda x: [int(d) for d in f"{int(x):04d}"]).tolist()
    data = np.array(data)

    if len(data) < 11:
        return None, None, None

    X, y = [], []
    for i in range(len(data) - 10):
        X.append(data[i:i+10])
        y.append(data[i+10])
    X, y = np.array(X), np.array(y)

    y_encoded = np.array([to_categorical(d, num_classes=10) for d in y])
    y_encoded = y_encoded.reshape(-1, 40)

    input_seq = np.array(data[-10:]).reshape(1, 10, 4)

    return X, y_encoded, input_seq

def top5_lstm(df):
    X, y, input_seq = prepare_lstm_data(df)
    if X is None or y is None or input_seq is None:
        return [[0], [0], [0], [0]]

    model = train_lstm_model(X, y)
    pred = model.predict(input_seq, verbose=0)[0].reshape(4, 10)

    top5 = []
    for i in range(4):
        top = list(np.argsort(-pred[i])[:5])
        top5.append(top)

    return top5
