import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, LayerNormalization, Bidirectional
from tensorflow.keras.layers import MultiHeadAttention, Add, Embedding
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

# --- Config ---
window_size = 5
num_digits = 10
embedding_dim = 16

def positional_encoding(length, depth):
    angle_rads = np.arange(length)[:, np.newaxis] / np.power(10000, (2 * (np.arange(depth)[np.newaxis, :]//2)) / depth)
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)

def build_model(input_len=window_size):
    inputs = Input(shape=(input_len,))
    x = Embedding(input_dim=10, output_dim=embedding_dim)(inputs)

    pos_encoding = positional_encoding(input_len, embedding_dim)
    x += pos_encoding

    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = LayerNormalization()(x)

    attn_output = MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    x = Add()([x, attn_output])
    x = LayerNormalization()(x)

    x = Bidirectional(LSTM(32))(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(num_digits, activation='softmax')(x)
    model = Model(inputs, x)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def preprocess_data(df, window_size=window_size):
    angka = [int(x) for x in df['angka'].values if len(x) == 4]
    sequences = []
    labels = [[] for _ in range(4)]
    for i in range(len(angka) - window_size):
        seq = [int(d) for num in angka[i:i+window_size] for d in f"{num:04d}"]
        target = f"{angka[i+window_size]:04d}"
        sequences.append(seq)
        for j in range(4):
            labels[j].append(int(target[j]))
    X = np.array(sequences).reshape(-1, window_size * 4)
    X = X.reshape((-1, window_size * 4)) % 10
    return X, [np.array(l) for l in labels]

def save_model(model, path):
    model.save(path)

def model_exists(path):
    return os.path.exists(path)

def load_model_if_exists(path):
    return load_model(path) if os.path.exists(path) else None

def top6_lstm(df, lokasi, return_probs=False, top_n=6):
    if len(df) < window_size + 1:
        return [[] for _ in range(4)]

    X, _ = preprocess_data(df)
    X_last = X[-1:]
    hasil = []
    confs = []

    for i in range(4):
        path = f"saved_models/{lokasi.lower().replace(' ', '_')}_digit{i}.h5"
        if not os.path.exists(path):
            hasil.append([])
            confs.append([])
            continue
        model = load_model(path)
        probas = model.predict(X_last, verbose=0)[0]
        top_idx = np.argsort(probas)[::-1][:top_n]
        hasil.append(top_idx.tolist())
        confs.append([probas[k] for k in top_idx])

    return (hasil, confs) if return_probs else hasil

def top6_ensemble(df, lokasi, weight_lstm=0.6, weight_markov=0.4, top_n=6):
    from markov_model import top6_markov
    result_lstm, conf_lstm = top6_lstm(df, lokasi, return_probs=True, top_n=10)
    result_markov, conf_markov = top6_markov(df)
    final = []

    for i in range(4):
        prob_dict = {}
        for idx, p in zip(result_lstm[i], conf_lstm[i]):
            prob_dict[idx] = prob_dict.get(idx, 0) + p * weight_lstm
        for idx in result_markov[i]:
            prob_dict[idx] = prob_dict.get(idx, 0) + (1.0 / len(result_markov[i])) * weight_markov
        sorted_prob = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        final.append([k for k, _ in sorted_prob])

    return final

def kombinasi_4d(df, lokasi, top_n=10, mode="average"):
    digit_pred, conf_pred = top6_lstm(df, lokasi, return_probs=True, top_n=6)
    kombinasi = []

    for a in range(len(digit_pred[0])):
        for b in range(len(digit_pred[1])):
            for c in range(len(digit_pred[2])):
                for d in range(len(digit_pred[3])):
                    angka = f"{digit_pred[0][a]}{digit_pred[1][b]}{digit_pred[2][c]}{digit_pred[3][d]}"
                    if mode == "average":
                        score = (conf_pred[0][a] + conf_pred[1][b] + conf_pred[2][c] + conf_pred[3][d]) / 4
                    else:
                        score = conf_pred[0][a] * conf_pred[1][b] * conf_pred[2][c] * conf_pred[3][d]
                    kombinasi.append((angka, score))

    kombinasi.sort(key=lambda x: x[1], reverse=True)
    return kombinasi[:top_n]
