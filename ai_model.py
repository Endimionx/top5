import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM, GRU, Dropout, Dense,
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

def preprocess_data(df, window_size=7):
    if len(df) < window_size + 1:
        return np.array([]), {label: np.array([]) for label in DIGIT_LABELS}
    sequences = []
    targets = {label: [] for label in DIGIT_LABELS}
    angka = df["angka"].values
    for i in range(len(angka) - window_size):
        window = angka[i:i+window_size]
        if any(len(x) != 4 or not x.isdigit() for x in window):
            continue
        seq = [int(d) for num in window[:-1] for d in f"{int(num):04d}"]
        sequences.append(seq)
        target_digits = [int(d) for d in f"{int(window[-1]):04d}"]
        for j, label in enumerate(DIGIT_LABELS):
            targets[label].append(to_categorical(target_digits[j], num_classes=10))
    X = np.array(sequences)
    y_dict = {label: np.array(targets[label]) for label in DIGIT_LABELS}
    return X, y_dict

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
    skip = x
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = tf.keras.layers.Add()([x, skip])
    logits = Dense(10)(x)
    outputs = tf.keras.layers.Activation('softmax')(logits / temperature)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=["accuracy"])
    return model

def build_transformer_model(input_len, embed_dim=32, heads=4, temperature=0.5):
    inputs = Input(shape=(input_len,))
    x = Embedding(input_dim=10, output_dim=embed_dim)(inputs)
    x = PositionalEncoding()(x)
    for _ in range(2):
        attn = MultiHeadAttention(num_heads=heads, key_dim=embed_dim)(x, x)
        x = LayerNormalization()(x + attn)
        ff = Dense(embed_dim, activation='relu')(x)
        x = LayerNormalization()(x + ff)
    x = GlobalAveragePooling1D()(x)
    skip = x
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = tf.keras.layers.Add()([x, skip])
    logits = Dense(10)(x)
    outputs = tf.keras.layers.Activation('softmax')(logits / temperature)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=["accuracy"])
    return model

def build_gru_model(input_len, embed_dim=32, gru_units=128, temperature=0.5):
    inputs = Input(shape=(input_len,))
    x = Embedding(input_dim=10, output_dim=embed_dim)(inputs)
    x = PositionalEncoding()(x)
    x = Bidirectional(GRU(gru_units, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(GRU(gru_units, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    skip = x
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = tf.keras.layers.Add()([x, skip])
    logits = Dense(10)(x)
    outputs = tf.keras.layers.Activation('softmax')(logits / temperature)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=["accuracy"])
    return model

def train_and_save_model(df, lokasi, model_type="lstm", window_size=7):
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("training_logs", exist_ok=True)
    X, y_dict = preprocess_data(df, window_size=window_size)
    if X.shape[0] == 0:
        print("[ERROR] Data tidak cukup.")
        return

    loc_id = lokasi.lower().replace(" ", "_")
    candidates = [
        ("LSTM", lambda input_len: build_lstm_model(input_len)),
        ("Transformer", lambda input_len: build_transformer_model(input_len)),
        ("GRU", lambda input_len: build_gru_model(input_len)),
    ]

    for label in DIGIT_LABELS:
        best_model, best_score, best_name = None, 0.0, ""
        for name, fn in candidates:
            try:
                model = fn(X.shape[1])
                history = model.fit(X, y_dict[label], epochs=30, batch_size=32, validation_split=0.2, verbose=0)
                val_acc = max(history.history['val_accuracy'])
                if val_acc > best_score:
                    best_model, best_score, best_name = model, val_acc, name
            except Exception as e:
                print(f"[ERROR] {label} {name}: {e}")

        if best_model:
            model_path = f"saved_models/{loc_id}_{label}_{model_type}.h5"
            log_path = f"training_logs/history_{loc_id}_{label}_{model_type}.csv"
            type_path = f"training_logs/best_model_type_{loc_id}_{label}.txt"
            best_model.fit(X, y_dict[label], epochs=50, batch_size=32, validation_split=0.2,
                           callbacks=[
                               CSVLogger(log_path),
                               EarlyStopping(patience=5, monitor="val_loss", restore_best_weights=True),
                               ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_loss")
                           ], verbose=0)
            best_model.save(model_path)
            with open(type_path, "w") as f:
                f.write(f"{best_name}\t{best_score:.4f}")
            print(f"[✅] {label.upper()} ({best_name}) disimpan.")

def model_exists(lokasi, model_type="lstm"):
    loc_id = lokasi.lower().strip().replace(" ", "_")
    return all(os.path.exists(f"saved_models/{loc_id}_{label}_{model_type}.h5") for label in DIGIT_LABELS)


def top6_model(df, lokasi, model_type="lstm", return_probs=False, temperature=0.5, window_size=7, mode_prediksi="hybrid", threshold=0.001):
    X, _ = preprocess_data(df, window_size=window_size)
    if X.shape[0] == 0:
        return None
    results, probs = [], []
    loc_id = lokasi.lower().replace(" ", "_")
    for label in DIGIT_LABELS:
        type_path = f"training_logs/best_model_type_{loc_id}_{label}.txt"
        if os.path.exists(type_path):
            model_type = open(type_path).read().split("\t")[0].lower()
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
            top6 = [i for i in avg.argsort()[::-1][:6]]
        else:
            top6 = [i for i in avg.argsort()[::-1] if avg[i] >= threshold][:6]
        results.append(top6)
        probs.append([avg[i] for i in top6])
    return (results, probs) if return_probs else results

def kombinasi_4d(df, lokasi, model_type="lstm", top_n=10, min_conf=0.0001, power=1.5, mode='product', window_size=7, mode_prediksi="hybrid"):
    result, probs = top6_model(df, lokasi=lokasi, model_type=model_type, return_probs=True, window_size=window_size, mode_prediksi=mode_prediksi)
    if result is None or probs is None:
        return []
    combinations = list(product(*result))
    scores = []
    for combo in combinations:
        try:
            digit_scores = [probs[i][result[i].index(combo[i])] ** power for i in range(4)]
        except:
            continue
        score = np.prod(digit_scores) if mode == 'product' else np.mean(digit_scores)
        if score >= min_conf:
            scores.append(("".join(map(str, combo)), score))
    return sorted(scores, key=lambda x: -x[1])[:top_n]

def top6_ensemble(df, lokasi, model_type="lstm", lstm_weight=0.6, markov_weight=0.4, window_size=7, mode_prediksi="hybrid"):
    lstm_result, lstm_probs = top6_model(df, lokasi=lokasi, model_type=model_type, return_probs=True, window_size=window_size, mode_prediksi=mode_prediksi)
    markov_result, _ = top6_markov(df)
    if lstm_result is None or markov_result is None:
        return None
    ensemble = []
    for i in range(4):
        all_digits = list(set(lstm_result[i] + markov_result[i]))
        scores = {}
        for d in all_digits:
            lstm_score = lstm_weight * (lstm_probs[i][lstm_result[i].index(d)] if d in lstm_result[i] else 0)
            markov_score = markov_weight * (1.0 / (1 + markov_result[i].index(d))) if d in markov_result[i] else 0
            scores[d] = lstm_score + markov_score
        top6 = sorted(scores.items(), key=lambda x: -x[1])[:6]
        ensemble.append([x[0] for x in top6])
    return ensemble

def evaluate_top6_accuracy(model, X, y_true):
    pred = model.predict(X, verbose=0)
    top6 = np.argsort(pred, axis=1)[:, -6:]
    true = np.argmax(y_true, axis=1)
    return np.mean([true[i] in top6[i] for i in range(len(true))])

def evaluate_lstm_accuracy_all_digits(df, lokasi, model_type="lstm", window_size=7):
    X, y_dict = preprocess_data(df, window_size)
    if X.shape[0] == 0:
        return None, None, None
    loc_id = lokasi.lower().replace(" ", "_")
    acc_top1_list, acc_top6_list, label_accuracy_list = [], [], []
    for label in DIGIT_LABELS:
        type_path = f"training_logs/best_model_type_{loc_id}_{label}.txt"
        if os.path.exists(type_path):
            model_type = open(type_path).read().split("\t")[0].lower()
        path = f"saved_models/{loc_id}_{label}_{model_type}.h5"
        if not os.path.exists(path):
            return None, None, None
        model = load_model(path, compile=True, custom_objects={"PositionalEncoding": PositionalEncoding})
        if model.input_shape[1] != X.shape[1]:
            return None, None, None
        y_true = y_dict[label]
        acc1 = model.evaluate(X, y_true, verbose=0)[1]
        acc6 = evaluate_top6_accuracy(model, X, y_true)
        acc_top1_list.append(acc1)
        acc_top6_list.append(acc6)
        y_pred = np.argmax(model.predict(X, verbose=0), axis=1)
        y_true_label = np.argmax(y_true, axis=1)
        label_acc = {d: np.mean(y_pred[y_true_label == d] == d) if np.any(y_true_label == d) else None for d in range(10)}
        label_accuracy_list.append(label_acc)
    return acc_top1_list, acc_top6_list, label_accuracy_list
