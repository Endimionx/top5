import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Layer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.utils import to_categorical

# Positional Encoding dummy layer (bisa dikembangkan)
class PositionalEncoding(Layer):
    def call(self, inputs):
        return inputs

# Attention Layer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def call(self, inputs):
        query, value = inputs, inputs
        scores = tf.matmul(query, value, transpose_b=True)
        weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(weights, value)
        return tf.reduce_mean(context, axis=1)

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

    if len(X) == 0 or len(y) == 0:
        return None, None

    y_encoded = [to_categorical(y[:, i], num_classes=10) for i in range(4)]

    return X, y_encoded

def model_exists(lokasi):
    filename = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"
    return os.path.exists(filename)

def train_and_save_lstm(df, lokasi):
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("training_logs", exist_ok=True)
    filename = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"
    log_file = f"training_logs/history_{lokasi.lower().replace(' ', '_')}.csv"

    X, y = prepare_lstm_data(df)
    if X is None or y is None:
        return

    if os.path.exists(filename):
        print(f"🧠 Fine-tuning model untuk {lokasi}")
        model = load_model(filename, custom_objects={
            "AttentionLayer": AttentionLayer,
            "PositionalEncoding": PositionalEncoding
        })
    else:
        print(f"🧠 Membuat model baru untuk {lokasi}")
        inputs = Input(shape=(10, 4))
        x = PositionalEncoding()(inputs)
        x = LSTM(128, return_sequences=True)(x)
        x = Dropout(0.3)(x)
        x = AttentionLayer()(x)
        x = Dropout(0.2)(x)
        outputs = [Dense(10, activation='softmax', name=f'output_{i}')(x) for i in range(4)]
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=0),
        CSVLogger(log_file, append=False)
    ]

    model.fit(X, y, epochs=50, batch_size=16, verbose=0, callbacks=callbacks)
    model.save(filename)

def top6_lstm(df, lokasi=None):
    filename = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"
    if not os.path.exists(filename):
        return None

    data = df['angka'].dropna().apply(lambda x: [int(d) for d in f"{int(x):04d}"]).tolist()
    if len(data) < 10:
        return None

    input_seq = np.array(data[-10:]).reshape(1, 10, 4)
    model = load_model(filename, custom_objects={
        "AttentionLayer": AttentionLayer,
        "PositionalEncoding": PositionalEncoding
    })
    preds = model.predict(input_seq, verbose=0)

    top6 = []
    for i in range(4):
        top = list(np.argsort(-preds[i][0])[:6])
        top6.append(top)
    return top6

def anti_top6_lstm(df, lokasi=None):
    filename = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"
    if not os.path.exists(filename):
        return None

    data = df['angka'].dropna().apply(lambda x: [int(d) for d in f"{int(x):04d}"]).tolist()
    if len(data) < 10:
        return None

    input_seq = np.array(data[-10:]).reshape(1, 10, 4)
    model = load_model(filename, custom_objects={
        "AttentionLayer": AttentionLayer,
        "PositionalEncoding": PositionalEncoding
    })
    preds = model.predict(input_seq, verbose=0)

    anti6 = []
    for i in range(4):
        lowest = list(np.argsort(preds[i][0])[:6])
        anti6.append(lowest)
    return anti6

def low6_lstm(df, lokasi=None):
    return anti_top6_lstm(df, lokasi)

def combo_4d_lstm(df, lokasi=None, n=10):
    pred = top6_lstm(df, lokasi)
    if pred is None:
        return None
    from itertools import product
    combos = list(product(*[pred[i][:4] for i in range(4)]))
    combos = [''.join(map(str, c)) for c in combos]
    return combos[:n]
