import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Layer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],), initializer="zeros", trainable=True)
        self.u = self.add_weight(name="context_vector", shape=(input_shape[-1],), initializer="glorot_uniform", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        v = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        vu = tf.tensordot(v, self.u, axes=1)
        alphas = tf.nn.softmax(vu, axis=1)
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), axis=1)
        return output

class PositionalEncoding(Layer):
    def __init__(self, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs  # Dummy, tidak diubah

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
        model = load_model(filename, custom_objects={"AttentionLayer": AttentionLayer, "PositionalEncoding": PositionalEncoding})
    else:
        print(f"🧠 Membuat model baru untuk {lokasi}")
        inputs = Input(shape=(10, 4))
        x = LSTM(128, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        x = AttentionLayer()(x)
        x = Dense(64, activation='relu')(x)
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
    data = np.array(data)
    if len(data) < 10:
        return None

    model = load_model(filename, custom_objects={"AttentionLayer": AttentionLayer, "PositionalEncoding": PositionalEncoding})
    input_seq = np.array(data[-10:]).reshape(1, 10, 4)
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
    data = np.array(data)
    if len(data) < 10:
        return None

    model = load_model(filename, custom_objects={"AttentionLayer": AttentionLayer, "PositionalEncoding": PositionalEncoding})
    input_seq = np.array(data[-10:]).reshape(1, 10, 4)
    preds = model.predict(input_seq, verbose=0)

    anti_top6 = []
    for i in range(4):
        bottom = list(np.argsort(preds[i][0])[:6])
        anti_top6.append(bottom)
    return anti_top6

def low6_lstm(df, lokasi=None):
    return anti_top6_lstm(df, lokasi)
