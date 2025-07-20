import numpy as np
import streamlit as st
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
import time
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
    
    angka = df["angka"].values
    total_data = len(angka)
    num_windows = (total_data - 1) // window_size
    start_index = total_data - (num_windows * window_size + 1)
    if start_index < 0:
        start_index = 0

    sequences = []
    targets = {label: [] for label in DIGIT_LABELS}

    for i in range(start_index, total_data - window_size):
        window = angka[i:i+window_size+1]
        if any(len(str(x)) != 4 or not str(x).isdigit() for x in window):
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
    inputs = Input(shape=(input_len,), name="input_layer")
    x = Embedding(input_dim=10, output_dim=embed_dim, name="embedding")(inputs)
    x = PositionalEncoding()(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True), name="bilstm_1")(x)
    x = LayerNormalization(name="layernorm_1")(x)
    x = Dropout(0.3, name="dropout_1")(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True), name="bilstm_2")(x)
    x = LayerNormalization(name="layernorm_2")(x)
    x = MultiHeadAttention(num_heads=attention_heads, key_dim=embed_dim, name="multihead_attn")(x, x)
    x = Dropout(0.2, name="dropout_2")(x)
    x = GlobalAveragePooling1D(name="gap")(x)
    x = Dense(512, activation='relu', name="dense_1")(x)
    x = Dropout(0.3, name="dropout_3")(x)
    x = Dense(128, activation='relu', name="dense_2")(x)
    logits = Dense(10, name="logits")(x)
    outputs = tf.keras.layers.Activation('softmax', name="softmax")(logits / temperature)
    model = Model(inputs, outputs, name="lstm_digit_model")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
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
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    logits = Dense(10)(x)
    outputs = tf.keras.layers.Activation('softmax')(logits / temperature)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_and_save_model(df, lokasi, window_dict, model_type="lstm"):
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("training_logs", exist_ok=True)
    for label in DIGIT_LABELS:
        window_size = window_dict.get(label, 7)
        if len(df) < window_size + 5:
            continue
        
        # Penting: panggil ulang preprocess_data di setiap label agar window_size sesuai
        X, y_dict = preprocess_data(df, window_size=window_size)
        y = y_dict[label]

        if X.shape[0] == 0 or y.shape[0] == 0:
            continue

        suffix = model_type
        loc_id = lokasi.lower().strip().replace(" ", "_")
        log_path = f"training_logs/history_{loc_id}_{label}_{suffix}.csv"
        model_path = f"saved_models/{loc_id}_{label}_{suffix}.h5"
        callbacks = [
            CSVLogger(log_path),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
        ]

        if os.path.exists(model_path):
            model = load_model(model_path, compile=True, custom_objects={"PositionalEncoding": PositionalEncoding})
        else:
            model = build_transformer_model(X.shape[1]) if model_type == "transformer" else build_lstm_model(X.shape[1])

        model.fit(X, y, epochs=50, batch_size=32, verbose=0, validation_split=0.2, callbacks=callbacks)
        model.save(model_path)

def top6_model(df, lokasi=None, model_type="lstm", return_probs=False, temperature=0.5, window_dict=None, mode_prediksi="hybrid", threshold=0.001):
    results, probs = [], []
    loc_id = lokasi.lower().replace(" ", "_")
    for label in DIGIT_LABELS:
        window_size = window_dict.get(label, 7)
        X, _ = preprocess_data(df, window_size=window_size)
        if X.shape[0] == 0:
            return None
        path = f"saved_models/{loc_id}_{label}_{model_type}.h5"
        if not os.path.exists(path):
            return None
        try:
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
        except Exception as e:
            print(f"[ERROR {label}] {e}")
            return None
    return (results, probs) if return_probs else results

def kombinasi_4d(df, lokasi, model_type="lstm", top_n=10, min_conf=0.0001, power=1.5, mode='product', window_dict=None, mode_prediksi="hybrid"):
    result, probs = top6_model(df, lokasi=lokasi, model_type=model_type, return_probs=True,
                               window_dict=window_dict, mode_prediksi=mode_prediksi)
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

def top6_ensemble(df, lokasi, model_type="lstm", lstm_weight=0.6, markov_weight=0.4, window_dict=None, temperature=0.5, mode_prediksi="hybrid"):
    # LSTM prediction
    lstm_result = top6_model(
        df,
        lokasi=lokasi,
        model_type=model_type,
        return_probs=False,
        window_dict=window_dict,
        temperature=temperature,
        mode_prediksi=mode_prediksi
    )

    # Markov prediction
    markov_result, _ = top6_markov(df)
    
    if lstm_result is None or markov_result is None:
        return None

    ensemble = []
    for i in range(4):
        all_digits = lstm_result[i] + markov_result[i]
        scores = {}
        for digit in all_digits:
            scores[digit] = 0
            if digit in lstm_result[i]:
                scores[digit] += lstm_weight * (1.0 / (1 + lstm_result[i].index(digit)))
            if digit in markov_result[i]:
                scores[digit] += markov_weight * (1.0 / (1 + markov_result[i].index(digit)))
        top6 = sorted(scores.items(), key=lambda x: -x[1])[:6]
        ensemble.append([x[0] for x in top6])
    return ensemble
    
def model_exists(lokasi, model_type="lstm"):
    loc_id = lokasi.lower().strip().replace(" ", "_")
    for label in ["ribuan", "ratusan", "puluhan", "satuan"]:
        model_path = f"saved_models/{loc_id}_{label}_{model_type}.h5"
        if not os.path.exists(model_path):
            return False
    return True
    
def evaluate_lstm_accuracy_all_digits(df, lokasi, model_type="lstm", window_size=7):
    if isinstance(window_size, int):
        # Jika window_size adalah angka tunggal, ubah jadi dict semua digit
        window_dict = {label: window_size for label in ["ribuan", "ratusan", "puluhan", "satuan"]}
    else:
        window_dict = window_size

    acc_top1_list, acc_top6_list, label_accuracy_list = [], [], []
    loc_id = lokasi.lower().strip().replace(" ", "_")

    for label in ["ribuan", "ratusan", "puluhan", "satuan"]:
        ws = window_dict.get(label, 7)
        X, y_dict = preprocess_data(df, window_size=ws)
        if X.shape[0] == 0:
            acc_top1_list.append(0)
            acc_top6_list.append(0)
            label_accuracy_list.append({})
            continue

        y_true = y_dict[label]
        path = f"saved_models/{loc_id}_{label}_{model_type}.h5"
        if not os.path.exists(path):
            acc_top1_list.append(0)
            acc_top6_list.append(0)
            label_accuracy_list.append({})
            continue

        try:
            model = load_model(path, compile=True, custom_objects={"PositionalEncoding": PositionalEncoding})
            if model.input_shape[1] != X.shape[1]:
                acc_top1_list.append(0)
                acc_top6_list.append(0)
                label_accuracy_list.append({})
                continue

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
        except Exception as e:
            print(f"[ERROR EVALUATE {label}] {e}")
            acc_top1_list.append(0)
            acc_top6_list.append(0)
            label_accuracy_list.append({})

    return acc_top1_list, acc_top6_list, label_accuracy_list
def evaluate_top6_accuracy(model, X, y_true):
    """
    Menghitung akurasi top-6: apakah label benar termasuk dalam 6 prediksi teratas.
    """
    try:
        y_pred = model.predict(X, verbose=0)
        y_true_labels = np.argmax(y_true, axis=1)
        top6_preds = np.argsort(y_pred, axis=1)[:, -6:]
        correct = np.array([
            true_label in top6 for true_label, top6 in zip(y_true_labels, top6_preds)
        ])
        return np.mean(correct)
    except Exception as e:
        print(f"[ERROR evaluate_top6_accuracy] {e}")
        return 0.0
def find_best_window_size_with_model(df, label, lokasi, model_type="lstm", min_ws=3, max_ws=30):
    best_ws = min_ws
    best_acc = 0
    loc_id = lokasi.lower().strip().replace(" ", "_")
    for ws in range(min_ws, max_ws + 1):
        X, y_dict = preprocess_data(df, window_size=ws)
        y = y_dict[label]
        if X.shape[0] == 0 or y.shape[0] == 0:
            continue
        try:
            if model_type == "transformer":
                model = build_transformer_model(X.shape[1])
            else:
                model = build_lstm_model(X.shape[1])
            model.fit(X, y, epochs=5, batch_size=32, verbose=0)
            acc = model.evaluate(X, y, verbose=0)[1]
            print(f"[INFO {label} WS={ws}] Acc={acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_ws = ws
        except Exception as e:
            print(f"[ERROR {label} WS={ws}] {e}")
            continue
    return best_ws

def find_best_window_size_with_model_fast(df, label, lokasi, model_type="lstm", min_ws=3, max_ws=20):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Embedding, LSTM, Dense, Bidirectional,
        GlobalAveragePooling1D, Dropout
    )
    from tensorflow.keras.callbacks import EarlyStopping
    import numpy as np
    import streamlit as st

    def quick_model(input_len):
        inp = Input(shape=(input_len,))
        x = Embedding(input_dim=10, output_dim=32)(inp)
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        out = Dense(10, activation='softmax')(x)
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    best_acc = 0
    best_ws = min_ws

    for ws in range(min_ws, max_ws + 1):  # coba semua ws, ganjil & genap
        try:
            subset_len = int(0.75 * len(df))  # 75% terakhir dari data
            X, y_dict = preprocess_data(df.iloc[-subset_len:], window_size=ws)
            y = y_dict[label]
            if X.shape[0] == 0 or y.shape[0] == 0:
                continue

            model = quick_model(X.shape[1])
            history = model.fit(
                X, y,
                epochs=10,
                batch_size=32,
                verbose=0,
                validation_split=0.2,
                callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)]
            )
            val_acc = np.max(history.history['val_accuracy'])

            if val_acc > best_acc:
                best_acc = val_acc
                best_ws = ws

        except Exception as e:
            print(f"[ERROR {label} WS={ws}] {e}")
            continue

    # Tampilkan info hanya untuk hasil terbaik
    st.info(f"✅ {label.upper()} | Window Size Terbaik: {best_ws} | Akurasi: {best_acc:.2%}")
    return best_ws
    
def evaluate_top6_accuracy(model, X, y_true):
    import numpy as np
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_probs = model.predict(X, verbose=0)
    y_pred_top6 = np.argsort(y_pred_probs, axis=1)[:, -6:]
    match = [y_true_labels[i] in y_pred_top6[i] for i in range(len(y_true_labels))]
    return np.mean(match)

def find_best_window_size_with_model_true(df, label, lokasi, model_type="lstm", min_ws=4, max_ws=20, temperature=1.0, use_cv=False, cv_folds=5):
    import numpy as np
    import pandas as pd
    import streamlit as st
    import seaborn as sns
    import matplotlib.pyplot as plt
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.model_selection import KFold

    best_ws = None
    best_acc = 0
    best_score = 0
    table_data = []
    all_scores = []
    digit_counter = {i: 0 for i in range(10)}

    st.markdown(f"### 🔍 Pencarian Window Size - {label.upper()}")

    ws_range = list(range(min_ws, max_ws + 1))

    for ws in ws_range:
        try:
            X, y_dict = preprocess_data(df, window_size=ws)
            y = y_dict[label]
            if X.shape[0] == 0 or y.shape[0] == 0:
                continue

            acc_scores = []
            conf_scores = []
            top6_all = []

            if use_cv:
                kf = KFold(n_splits=cv_folds)
                for train_index, val_index in kf.split(X):
                    X_train, X_val = X[train_index], X[val_index]
                    y_train, y_val = y[train_index], y[val_index]

                    model = build_transformer_model(X.shape[1]) if model_type == "transformer" else build_lstm_model(X.shape[1])
                    model.fit(
                        X_train, y_train,
                        epochs=10,
                        batch_size=32,
                        verbose=0,
                        validation_data=(X_val, y_val),
                        callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]
                    )

                    val_preds = model.predict(X_val, verbose=0)
                    val_labels = np.argmax(val_preds, axis=1)
                    acc = np.mean(val_labels == y_val)
                    acc_scores.append(acc)

                    last_pred = model.predict(X[-1:], verbose=0)[0]
                    if temperature != 1.0:
                        last_pred = np.exp(np.log(last_pred + 1e-8) / temperature)
                        last_pred /= np.sum(last_pred)
                    preds = last_pred
                    conf_scores.append(np.mean(np.sort(last_pred)[::-1][:6]))
                    top6 = np.argsort(last_pred)[::-1][:6]
                    top6_all.extend(top6)

            else:
                model = build_transformer_model(X.shape[1]) if model_type == "transformer" else build_lstm_model(X.shape[1])
                history = model.fit(
                    X, y,
                    epochs=10,
                    batch_size=32,
                    verbose=0,
                    validation_split=0.2,
                    callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
                )
                val_acc = max(history.history.get("val_accuracy", [0]))
                preds = model.predict(X[-1:], verbose=0)[0]
                if temperature != 1.0:
                    preds = np.exp(np.log(preds + 1e-8) / temperature)
                    preds /= np.sum(preds)
                avg_conf = np.mean(np.sort(preds)[::-1][:6])
                top6 = np.argsort(preds)[::-1][:6]
                top6_all = top6.tolist()
                acc_scores = [val_acc]
                conf_scores = [avg_conf]

            val_acc = np.mean(acc_scores)
            avg_conf = np.mean(conf_scores)
            score = val_acc * avg_conf

            # Rata-rata Top6 dari semua fold
            top6_freq = sorted({d: top6_all.count(d) for d in set(top6_all)}.items(), key=lambda x: -x[1])[:6]
            top6_digits = [d for d, _ in top6_freq]

            table_data.append((ws, round(val_acc * 100, 2), round(avg_conf * 100, 2), top6_digits))
            all_scores.append((ws, val_acc, avg_conf, top6_digits, score))

            if score > best_score:
                best_score = score
                best_acc = val_acc
                best_ws = ws

        except Exception as e:
            print(f"[GAGAL {label.upper()} WS={ws}]: {e}")
            continue

    # Ambil top-5 berdasarkan skor
    top5 = sorted(all_scores, key=lambda x: -x[4])[:5]
    top5_top6 = []
    for _, _, _, top6, _ in top5:
        for d in top6:
            digit_counter[d] += 1
        top5_top6.extend(top6)

    avg_top6_digits = [x[0] for x in sorted(
        {d: top5_top6.count(d) for d in set(top5_top6)}.items(),
        key=lambda x: -x[1]
    )[:6]]

    # Tampilkan tabel hasil WS
    if table_data:
        df_table = pd.DataFrame(table_data, columns=["Window Size", "Val Accuracy (%)", "Avg Confidence (%)", "Top-6 Digit"])
        df_table = df_table.sort_values("Window Size")
        st.dataframe(df_table)

    st.markdown("#### 🔥 Heatmap Jumlah Kemunculan Top-6 Digit (Top-5 WS)")
    heat_df = pd.DataFrame([digit_counter]).T
    heat_df.columns = ["Count"]
    heat_df.index.name = "Digit"
    fig, ax = plt.subplots(figsize=(8, 1.5))
    sns.heatmap(heat_df.T, annot=True, cmap="YlGnBu", cbar=False, ax=ax)
    st.pyplot(fig)

    st.markdown(f"**🔁 Top-6 Rata-rata dari 5 WS terbaik:** `{', '.join(map(str, avg_top6_digits))}`")
    st.success(f"✅ {label.upper()} - WS terbaik: {best_ws} (Val Acc: {best_acc:.2%})")
    return best_ws, avg_top6_digits
        
def find_best_window_smart(df, label, lokasi, model_type="lstm", min_ws=4, max_ws=20, temperature=1.0, repeats=3):
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import streamlit as st
    import tensorflow as tf
    import random
    from tensorflow.keras.callbacks import EarlyStopping

    # Set random seed
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    st.markdown(f"### 🔍 Pencarian Window Size Cerdas - {label.upper()}")

    best_ws = None
    best_score = 0
    best_val_acc = 0
    table_data = []
    digit_counter = {i: 0 for i in range(10)}
    top6_pool = []

    for ws in range(min_ws, max_ws + 1):
        try:
            X, y_dict = preprocess_data(df, window_size=ws)
            y = y_dict[label]
            if X.shape[0] == 0 or y.shape[0] == 0:
                continue

            val_acc_list = []
            conf_list = []
            all_top6 = []

            for _ in range(repeats):
                model = build_transformer_model(X.shape[1]) if model_type == "transformer" else build_lstm_model(X.shape[1])
                history = model.fit(
                    X, y,
                    epochs=10,
                    batch_size=32,
                    verbose=0,
                    validation_split=0.2,
                    callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
                )
                val_acc = max(history.history.get("val_accuracy", [0]))
                preds = model.predict(X[-1:], verbose=0)[0]
                if temperature != 1.0:
                    preds = np.exp(np.log(preds + 1e-8) / temperature)
                    preds /= np.sum(preds)

                avg_conf = np.mean(np.sort(preds)[::-1][:6])
                top6 = np.argsort(preds)[::-1][:6].tolist()

                val_acc_list.append(val_acc)
                conf_list.append(avg_conf)
                all_top6.extend(top6)

            avg_val_acc = np.mean(val_acc_list)
            avg_conf = np.mean(conf_list)
            score = avg_val_acc * avg_conf

            # Tambahkan ke heatmap pool
            for d in all_top6:
                digit_counter[d] += 1
            top6_pool.extend(all_top6)

            table_data.append((ws, round(avg_val_acc * 100, 2), round(avg_conf * 100, 2), list(np.unique(all_top6)[:6])))

            if score > best_score:
                best_score = score
                best_val_acc = avg_val_acc
                best_ws = ws

        except Exception as e:
            print(f"[GAGAL SMART {label.upper()} WS={ws}]: {e}")
            continue

    # Tabel hasil
    if table_data:
        df_table = pd.DataFrame(table_data, columns=["Window Size", "Val Accuracy (%)", "Avg Confidence (%)", "Top-6 Digit"])
        df_table = df_table.sort_values("Window Size")
        st.dataframe(df_table)

    # Heatmap
    st.markdown("#### 🔥 Heatmap Jumlah Kemunculan Top-6 Digit")
    heat_df = pd.DataFrame([digit_counter]).T
    heat_df.columns = ["Count"]
    heat_df.index.name = "Digit"
    fig, ax = plt.subplots(figsize=(8, 1.5))
    sns.heatmap(heat_df.T, annot=True, cmap="YlGnBu", cbar=False, ax=ax)
    st.pyplot(fig)

    # Rangkuman
    avg_top6_digits = [x[0] for x in sorted(
        {d: top6_pool.count(d) for d in set(top6_pool)}.items(),
        key=lambda x: -x[1]
    )[:6]]

    st.markdown(f"**🔁 Top-6 Rata-rata dari semua WS:** `{', '.join(map(str, avg_top6_digits))}`")
    st.success(f"✅ {label.upper()} - WS terbaik: {best_ws} (Val Acc: {best_val_acc:.2%})")

    return best_ws, avg_top6_digits

def find_best_window_smart_fast(df, label, lokasi, model_type="lstm", min_ws=4, max_ws=20, temperature=1.0, repeats=3):
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import streamlit as st
    import tensorflow as tf
    import random
    from tensorflow.keras.callbacks import EarlyStopping

    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    st.markdown(f"### ⚡ Pencarian WS Cepat - {label.upper()}")

    best_ws = None
    best_score = 0
    best_val_acc = 0
    table_data = []
    digit_counter = {i: 0 for i in range(10)}
    top6_pool = []

    for ws in range(min_ws, max_ws + 1):
        try:
            X, y_dict = preprocess_data(df, window_size=ws)
            y = y_dict[label]
            if X.shape[0] == 0 or y.shape[0] == 0:
                continue

            val_acc_list = []
            conf_list = []
            all_top6 = []

            for _ in range(repeats):
                model = build_lstm_model(X.shape[1])
                history = model.fit(
                    X, y,
                    epochs=5,
                    batch_size=32,
                    verbose=0,
                    validation_split=0.2,
                    callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]
                )
                val_acc = max(history.history.get("val_accuracy", [0]))
                preds = model.predict(X[-1:], verbose=0)[0]
                if temperature != 1.0:
                    preds = np.exp(np.log(preds + 1e-8) / temperature)
                    preds /= np.sum(preds)

                avg_conf = np.mean(np.sort(preds)[::-1][:6])
                top6 = np.argsort(preds)[::-1][:6].tolist()

                val_acc_list.append(val_acc)
                conf_list.append(avg_conf)
                all_top6.extend(top6)

            avg_val_acc = np.mean(val_acc_list)
            avg_conf = np.mean(conf_list)
            score = avg_val_acc * avg_conf

            for d in all_top6:
                digit_counter[d] += 1
            top6_pool.extend(all_top6)

            table_data.append((ws, round(avg_val_acc * 100, 2), round(avg_conf * 100, 2), list(np.unique(all_top6)[:6])))

            if score > best_score:
                best_score = score
                best_val_acc = avg_val_acc
                best_ws = ws

        except Exception as e:
            print(f"[GAGAL SMART FAST {label.upper()} WS={ws}]: {e}")
            continue

    # Tampilkan tabel
    if table_data:
        df_table = pd.DataFrame(table_data, columns=["Window Size", "Val Accuracy (%)", "Avg Confidence (%)", "Top-6 Digit"])
        df_table = df_table.sort_values("Window Size")
        st.dataframe(df_table)

    # Heatmap
    st.markdown("#### 🔥 Heatmap Top-6 Digit (Semua WS)")
    heat_df = pd.DataFrame([digit_counter]).T
    heat_df.columns = ["Count"]
    heat_df.index.name = "Digit"
    fig, ax = plt.subplots(figsize=(8, 1.5))
    sns.heatmap(heat_df.T, annot=True, cmap="YlGnBu", cbar=False, ax=ax)
    st.pyplot(fig)

    avg_top6_digits = [x[0] for x in sorted(
        {d: top6_pool.count(d) for d in set(top6_pool)}.items(),
        key=lambda x: -x[1]
    )[:6]]

    st.markdown(f"**🔁 Rata-rata Top-6 Digit:** `{', '.join(map(str, avg_top6_digits))}`")
    st.success(f"✅ {label.upper()} - WS terbaik: {best_ws} (Val Acc: {best_val_acc:.2%})")

    return best_ws, avg_top6_digits
