import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Layer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.initializers import RandomNormal

class PositionalEncoding(Layer):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        angles = tf.cast(positions[:, tf.newaxis] / tf.pow(10000., (2 * (tf.range(tf.shape(x)[2]) // 2)) / tf.cast(tf.shape(x)[2], tf.float32)), tf.float32)
        angle_rads = tf.where(tf.range(tf.shape(x)[2]) % 2 == 0, tf.math.sin(angles), tf.math.cos(angles))
        return x + angle_rads

class AttentionLayer(Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],), initializer="zeros", trainable=True)
        self.V = self.add_weight(name="att_var", shape=(input_shape[-1], 1), initializer="glorot_uniform", trainable=True)

    def call(self, inputs):
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.V, axes=1), axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

def prepare_lstm_data(df):
    data = df['angka'].dropna().apply(lambda x: [int(d) for d in f"{int(x):04d}"]).tolist()
    data = np.array(data)

    if len(data) < 11:
        return None, None

    X, y = [], []
    for i in range(len(data) - 10):
        X.append(data[i:i + 10])
        y.append(data[i + 10])
    X = np.array(X)
    y = np.array(y)

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
        model = load_model(filename, custom_objects={
            "AttentionLayer": AttentionLayer,
            "PositionalEncoding": PositionalEncoding
        })
    else:
        inputs = Input(shape=(10, 4))
        x = PositionalEncoding()(inputs)
        x = LSTM(128, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = AttentionLayer()(x)
        x = Dense(64, activation='relu')(x)
        outputs = [Dense(10, activation='softmax', name=f'output_{i}')(x) for i in range(4)]
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='loss', patience=3, factor=0.5, verbose=0),
        CSVLogger(log_file, append=True)
    ]

    model.fit(X, y, epochs=50, batch_size=16, verbose=0, callbacks=callbacks)
    model.save(filename)

def top6_lstm(df, lokasi=None, return_probs=False):
    filename = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"
    if not os.path.exists(filename):
        return None

    data = df['angka'].dropna().apply(lambda x: [int(d) for d in f"{int(x):04d}"]).tolist()
    data = np.array(data)
    if len(data) < 10:
        return None

    model = load_model(filename, custom_objects={
        "AttentionLayer": AttentionLayer,
        "PositionalEncoding": PositionalEncoding
    })
    input_seq = np.array(data[-10:]).reshape(1, 10, 4)
    preds = model.predict(input_seq, verbose=0)

    if return_probs:
        return [pred[0] for pred in preds]  # 4 x 10 list of softmax scores

    top6 = []
    for i in range(4):
        top = list(np.argsort(-preds[i][0])[:6])
        top6.append(top)
    return top6

def anti_top6_lstm(df, lokasi=None):
    probs = top6_lstm(df, lokasi, return_probs=True)
    if probs is None:
        return None
    anti = []
    for i in range(4):
        lowest = list(np.argsort(probs[i])[:6])
        anti.append(lowest)
    return anti

def low6_lstm(df, lokasi=None):
    return anti_top6_lstm(df, lokasi)

def prediksi_kombinasi_4d(df, lokasi=None, top_k=10):
    probs = top6_lstm(df, lokasi, return_probs=True)
    if probs is None:
        return None

    ribuan = [(d, probs[0][d]) for d in range(10)]
    ratusan = [(d, probs[1][d]) for d in range(10)]
    puluhan = [(d, probs[2][d]) for d in range(10)]
    satuan = [(d, probs[3][d]) for d in range(10)]

    kombinasi = []
    for a in ribuan:
        for b in ratusan:
            for c in puluhan:
                for d in satuan:
                    angka = f"{a[0]}{b[0]}{c[0]}{d[0]}"
                    conf = a[1] * b[1] * c[1] * d[1]
                    kombinasi.append((angka, conf))

    kombinasi = sorted(kombinasi, key=lambda x: -x[1])
    return kombinasi[:top_k]
