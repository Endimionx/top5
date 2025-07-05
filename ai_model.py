import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, Layer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
import pandas as pd

# Positional Encoding Layer
class PositionalEncoding(Layer):
    def __init__(self, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)

    def call(self, x):
        pos = tf.range(tf.shape(x)[1], dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(tf.shape(x)[2], dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(tf.shape(x)[2], tf.float32))
        angle_rads = pos * angle_rates
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        return x + pos_encoding

# Attention Layer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def call(self, inputs):
        scores = tf.matmul(inputs, inputs, transpose_b=True)
        weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(weights, inputs)
        return tf.reduce_mean(context, axis=1)

# Data preprocessing
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
    y_encoded = [to_categorical(y[:, i], num_classes=10) for i in range(4)]
    return X, y_encoded

# Cek model
def model_exists(lokasi):
    filename = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"
    return os.path.exists(filename)

# Training model
def train_and_save_lstm(df, lokasi):
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("training_logs", exist_ok=True)
    filename = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"
    log_file = f"training_logs/history_{lokasi.lower().replace(' ', '_')}.csv"

    X, y = prepare_lstm_data(df)
    if X is None or y is None:
        return

    input_layer = Input(shape=(10, 4))
    x = PositionalEncoding()(input_layer)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    x = AttentionLayer()(x)
    x = Dense(64, activation='relu')(x)
    outputs = [Dense(10, activation='softmax', name=f"output_{i}")(x) for i in range(4)]
    model = Model(inputs=input_layer, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
        CSVLogger(log_file, append=False)
    ]

    model.fit(X, y, epochs=50, batch_size=16, verbose=0, callbacks=callbacks)
    model.save(filename)

# Prediksi top 6
def top6_lstm(df, lokasi=None):
    filename = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"
    if not os.path.exists(filename):
        return None
    data = df['angka'].dropna().apply(lambda x: [int(d) for d in f"{int(x):04d}"]).tolist()
    if len(data) < 10:
        return None
    input_seq = np.array(data[-10:]).reshape(1, 10, 4)
    model = load_model(filename, custom_objects={
        "PositionalEncoding": PositionalEncoding,
        "AttentionLayer": AttentionLayer
    })
    preds = model.predict(input_seq, verbose=0)
    top6 = []
    for i in range(4):
        top = list(np.argsort(-preds[i][0])[:6])
        top6.append(top)
    return top6

# Prediksi angka probabilitas rendah (anti-top6)
def anti_top6_lstm(df, lokasi=None):
    filename = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"
    if not os.path.exists(filename):
        return None
    data = df['angka'].dropna().apply(lambda x: [int(d) for d in f"{int(x):04d}"]).tolist()
    if len(data) < 10:
        return None
    input_seq = np.array(data[-10:]).reshape(1, 10, 4)
    model = load_model(filename, custom_objects={
        "PositionalEncoding": PositionalEncoding,
        "AttentionLayer": AttentionLayer
    })
    preds = model.predict(input_seq, verbose=0)
    anti_top6 = []
    for i in range(4):
        bottom = list(np.argsort(preds[i][0])[:6])
        anti_top6.append(bottom)
    return anti_top6

# Alias untuk kompatibilitas lama
def low6_lstm(df, lokasi=None):
    return anti_top6_lstm(df, lokasi)
