import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, Layer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger

# ===== Positional Encoding Layer =====
class PositionalEncoding(Layer):
    def __init__(self, maxlen, d_model):
        super().__init__()
        self.maxlen = maxlen
        self.d_model = d_model

    def call(self, x):
        pos = np.arange(self.maxlen)[:, np.newaxis]
        i = np.arange(self.d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(self.d_model))
        angle_rads = pos * angle_rates
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        return x + tf.cast(pos_encoding, tf.float32)

# ===== Attention Layer =====
class AttentionLayer(Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        query = tf.keras.layers.Dense(64)(inputs)
        key = tf.keras.layers.Dense(64)(inputs)
        value = tf.keras.layers.Dense(64)(inputs)
        scores = tf.matmul(query, key, transpose_b=True)
        scores /= tf.math.sqrt(tf.cast(tf.shape(key)[-1], tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(weights, value)
        return tf.reduce_mean(context, axis=1)

# ===== Persiapan Data =====
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

def model_exists(lokasi):
    filename = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"
    return os.path.exists(filename)

# ===== Training & Simpan =====
def train_and_save_lstm(df, lokasi):
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("training_logs", exist_ok=True)
    filename = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"
    log_file = f"training_logs/history_{lokasi.lower().replace(' ', '_')}.csv"

    X, y = prepare_lstm_data(df)
    if X is None or y is None:
        return

    inputs = Input(shape=(10, 4))
    x = PositionalEncoding(10, 4)(inputs)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    x = AttentionLayer()(x)
    outputs = [Dense(10, activation='softmax', name=f'output_{i}')(x) for i in range(4)]
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=1),
        CSVLogger(log_file, append=False)
    ]

    model.fit(X, y, epochs=50, batch_size=16, verbose=0, callbacks=callbacks)
    model.save(filename)

# ===== Prediksi Top-6 Digit =====
def top6_lstm(df, lokasi=None):
    filename = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"
    if not os.path.exists(filename):
        return None
    data = df['angka'].dropna().apply(lambda x: [int(d) for d in f"{int(x):04d}"]).tolist()
    if len(data) < 10:
        return None
    input_seq = np.array(data[-10:]).reshape(1, 10, 4)
    model = load_model(filename, custom_objects={"PositionalEncoding": PositionalEncoding, "AttentionLayer": AttentionLayer})
    preds = model.predict(input_seq, verbose=0)
    top6 = [list(np.argsort(-p[0])[:6]) for p in preds]
    return top6

# ===== Anti Top-6 Digit =====
def anti_top6_lstm(df, lokasi=None):
    filename = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"
    if not os.path.exists(filename):
        return None
    data = df['angka'].dropna().apply(lambda x: [int(d) for d in f"{int(x):04d}"]).tolist()
    if len(data) < 10:
        return None
    input_seq = np.array(data[-10:]).reshape(1, 10, 4)
    model = load_model(filename, custom_objects={"PositionalEncoding": PositionalEncoding, "AttentionLayer": AttentionLayer})
    preds = model.predict(input_seq, verbose=0)
    low6 = [list(np.argsort(p[0])[:6]) for p in preds]
    return low6

# ===== Alias Legacy =====
def low6_lstm(df, lokasi=None):
    return anti_top6_lstm(df, lokasi)
