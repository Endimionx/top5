import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM, Dropout, Dense,
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import os
import pandas as pd
from itertools import product
from markov_model import top6_markov

DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

class PositionalEncoding(tf.keras.layers.Layer):
    def call(self, x):
        seq_len = tf.shape(x)[1]
        d_model = tf.shape(x)[2]
        pos = tf.cast(tf.range(seq_len)[:, tf.newaxis], dtype=tf.float32)
        i = tf.cast(tf.range(d_model)[tf.newaxis, :], dtype=tf.float32)
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = pos * angle_rates
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = tf.expand_dims(pos_encoding, 0)
        return x + tf.cast(pos_encoding, tf.float32)

def preprocess_data(df, window_size_dict):
    angka = df["angka"].values
    total_data = len(angka)
    sequences_dict = {}
    targets_dict = {}

    for label in DIGIT_LABELS:
        window_size = window_size_dict.get(label, 7)
        if total_data < window_size + 1:
            sequences_dict[label] = np.array([])
            targets_dict[label] = np.array([])
            continue

        num_windows = (total_data - 1) // window_size
        start_index = total_data - (num_windows * window_size + 1)
        if start_index < 0:
            start_index = 0

        sequences, targets = [], []
        for i in range(start_index, total_data - window_size):
            window = angka[i:i + window_size + 1]
            if any(len(str(x)) != 4 or not str(x).isdigit() for x in window):
                continue
            seq = [int(d) for num in window[:-1] for d in f"{int(num):04d}"]
            sequences.append(seq)
            target_digits = [int(d) for d in f"{int(window[-1]):04d}"]
            targets.append(to_categorical(target_digits[DIGIT_LABELS.index(label)], num_classes=10))

        sequences_dict[label] = np.array(sequences)
        targets_dict[label] = np.array(targets)
    return sequences_dict, targets_dict

def build_lstm_model(input_len, embed_dim=32, lstm_units=128, attention_heads=4, temperature=0.5):
    inputs = Input(shape=(input_len,))
    x = Embedding(input_dim=10, output_dim=embed_dim)(inputs)
    x = PositionalEncoding()(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = MultiHeadAttention(num_heads=attention_heads, key_dim=embed_dim)(x, x)
    x = Dropout(0.2)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    logits = Dense(10)(x)
    outputs = tf.keras.layers.Activation('softmax')(logits / temperature)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_and_save_model(df, lokasi, window_size_dict, model_type="lstm"):
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("training_logs", exist_ok=True)
    X_dict, y_dict = preprocess_data(df, window_size_dict)
    for label in DIGIT_LABELS:
        X = X_dict[label]
        y = y_dict[label]
        if X.shape[0] == 0:
            continue
        suffix = model_type
        loc_id = lokasi.lower().strip().replace(" ", "_")
        model_path = f"saved_models/{loc_id}_{label}_{suffix}.h5"
        log_path = f"training_logs/history_{loc_id}_{label}_{suffix}.csv"
        callbacks = [
            CSVLogger(log_path),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0)
        ]
        if os.path.exists(model_path):
            model = load_model(model_path, compile=True, custom_objects={"PositionalEncoding": PositionalEncoding})
        else:
            model = build_lstm_model(X.shape[1])
        model.fit(X, y, epochs=50, batch_size=32, verbose=0, validation_split=0.2, callbacks=callbacks)
        model.save(model_path)

def model_exists(lokasi, model_type="lstm"):
    loc_id = lokasi.lower().strip().replace(" ", "_")
    return all(os.path.exists(f"saved_models/{loc_id}_{label}_{model_type}.h5") for label in DIGIT_LABELS)

def top6_model(df, lokasi=None, model_type="lstm", return_probs=False, temperature=0.5, window_size_dict=None, mode_prediksi="hybrid", threshold=0.001):
    X_dict, _ = preprocess_data(df, window_size_dict)
    if not X_dict:
        return None
    results, probs = [], []
    loc_id = lokasi.lower().replace(" ", "_")
    for label in DIGIT_LABELS:
        X = X_dict[label]
        if X.shape[0] == 0:
            return None
        path = f"saved_models/{loc_id}_{label}_{model_type}.h5"
        if not os.path.exists(path):
            return None
        model = load_model(path, compile=False, custom_objects={"PositionalEncoding": PositionalEncoding})
        if model.input_shape[1] != X.shape[1]:
            return None
        pred = model.predict(X, verbose=0)
        avg = np.mean(pred, axis=0)
        avg /= np.sum(avg)
        if mode_prediksi == "confidence":
            top6 = avg.argsort()[-6:][::-1]
        elif mode_prediksi == "ranked":
            score_dict = {i: (1.0 / (1 + rank)) for rank, i in enumerate(avg.argsort()[::-1])}
            top6 = sorted(score_dict.items(), key=lambda x: -x[1])[:6]
            top6 = [d for d, _ in top6]
        else:
            score_dict = {i: avg[i] * (1.0 / (1 + rank)) for rank, i in enumerate(avg.argsort()[::-1])}
            sorted_scores = sorted(score_dict.items(), key=lambda x: -x[1])
            top6 = [d for d, score in sorted_scores if avg[d] >= threshold][:6]
        results.append(top6)
        probs.append([avg[d] for d in top6])
    return (results, probs) if return_probs else results

def kombinasi_4d(df, lokasi, model_type="lstm", top_n=10, min_conf=0.0001, power=1.5, mode='product', window_size_dict=None, mode_prediksi="hybrid"):
    result, probs = top6_model(df, lokasi=lokasi, model_type=model_type, return_probs=True, window_size_dict=window_size_dict, mode_prediksi=mode_prediksi)
    if result is None or probs is None:
        return []
    combinations = list(product(*result))
    scores = []
    for combo in combinations:
        digit_scores = []
        valid = True
        for i in range(4):
            try:
                idx = result[i].index(combo[i])
                digit_scores.append(probs[i][idx] ** power)
            except:
                valid = False
                break
        if not valid:
            continue
        score = np.prod(digit_scores) if mode == 'product' else np.mean(digit_scores)
        if score >= min_conf:
            scores.append(("".join(map(str, combo)), score))
    return sorted(scores, key=lambda x: -x[1])[:top_n]

def top6_ensemble(df, lokasi, model_type="lstm", lstm_weight=0.6, markov_weight=0.4, window_size_dict=None):
    lstm_result = top6_model(df, lokasi=lokasi, model_type=model_type, window_size_dict=window_size_dict)
    markov_result, _ = top6_markov(df)
    if lstm_result is None or markov_result is None:
        return None
    ensemble = []
    for i in range(4):
        all_digits = lstm_result[i] + markov_result[i]
        scores = {}
        for digit in all_digits:
            scores[digit] = scores.get(digit, 0)
            if digit in lstm_result[i]:
                scores[digit] += lstm_weight * (1.0 / (1 + lstm_result[i].index(digit)))
            if digit in markov_result[i]:
                scores[digit] += markov_weight * (1.0 / (1 + markov_result[i].index(digit)))
        top6 = sorted(scores.items(), key=lambda x: -x[1])[:6]
        ensemble.append([x[0] for x in top6])
    return ensemble

def evaluate_top6_accuracy(model, X, y_true):
    pred = model.predict(X, verbose=0)
    top6 = np.argsort(pred, axis=1)[:, -6:]
    true_labels = np.argmax(y_true, axis=1)
    correct = sum([true_labels[i] in top6[i] for i in range(len(true_labels))])
    return correct / len(true_labels)

def evaluate_lstm_accuracy_all_digits(df, lokasi, model_type="lstm", window_size_dict=None):
    X_dict, y_dict = preprocess_data(df, window_size_dict)
    if not X_dict:
        return None, None, None
    acc_top1_list, acc_top6_list, label_accuracy_list = [], [], []
    loc_id = lokasi.lower().strip().replace(" ", "_")
    for label in DIGIT_LABELS:
        X = X_dict[label]
        y_true = y_dict[label]
        if X.shape[0] == 0:
            continue
        path = f"saved_models/{loc_id}_{label}_{model_type}.h5"
        if not os.path.exists(path):
            return None, None, None
        model = load_model(path, compile=True, custom_objects={"PositionalEncoding": PositionalEncoding})
        if model.input_shape[1] != X.shape[1]:
            return None, None, None
        acc_top1 = model.evaluate(X, y_true, verbose=0)[1]
        acc_top6 = evaluate_top6_accuracy(model, X, y_true)
        acc_top1_list.append(acc_top1)
        acc_top6_list.append(acc_top6)
        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(model.predict(X, verbose=0), axis=1)
        label_acc = {}
        for d in range(10):
            idx = np.where(y_true_labels == d)[0]
            label_acc[d] = np.mean(y_pred_labels[idx] == d) if len(idx) > 0 else None
        label_accuracy_list.append(label_acc)
    return acc_top1_list, acc_top6_list, label_accuracy_list

def cari_window_terbaik(df, lokasi, model_type="lstm", min_ws=3, max_ws=20):
    best_ws = {}
    for label in DIGIT_LABELS:
        acc_scores = []
        for ws in range(min_ws, max_ws + 1):
            window_size_dict = {label: ws}
            X_dict, y_dict = preprocess_data(df, window_size_dict)
            X, y = X_dict[label], y_dict[label]
            if X.shape[0] < 10:
                continue
            model = build_lstm_model(X.shape[1])
            model.fit(X, y, epochs=10, batch_size=32, verbose=0, validation_split=0.2)
            acc = model.evaluate(X, y, verbose=0)[1]
            acc_scores.append((ws, acc))
        if acc_scores:
            best_ws[label] = max(acc_scores, key=lambda x: x[1])[0]
        else:
            best_ws[label] = 7
    return best_ws
