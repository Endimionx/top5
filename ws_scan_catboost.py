import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_val_score
from catboost import CatBoostClassifier
from tensorflow.keras.utils import to_categorical
from ai_model import (
    preprocess_data
)

DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]


def preprocess_datacat(df, window_size=7):
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


def get_top6_catboost(X, y, seed=42):
    """
    Latih model CatBoost dan ambil top-6 digit dengan confidence tertinggi.
    """
    model = CatBoostClassifier(verbose=0, random_seed=seed)
    model.fit(X, y)
    proba = model.predict_proba(X)
    mean_proba = np.mean(proba, axis=0)
    top6 = np.argsort(mean_proba)[-6:][::-1]
    return top6.tolist()


def scan_ws_catboost(df, label, min_ws=5, max_ws=15, cv_folds=3, seed=42):
    """
    Mencari window size terbaik menggunakan CatBoostClassifier.
    Menampilkan info, progress bar, dan menyimpan hasil ke session_state.
    """
    np.random.seed(seed)
    results = []

    total = max_ws - min_ws + 1
    progress = st.progress(0.0, text=f"⏳ Mulai proses SCAN CatBoost {label.upper()}...")
    
    for idx, ws in enumerate(range(min_ws, max_ws + 1), 1):
        progress.progress(idx / total, text=f"🔄 Evaluasi WS={ws} untuk {label.upper()}")
        #st.info(f"🔍 Sedang proses WS={ws} ({idx}/{total})", icon="🔁")

        X_all, y_dict = preprocess_data(df, window_size=ws)

        if len(X_all) == 0 or label not in y_dict:
            continue

        y_onehot = y_dict[label]
        if len(y_onehot) == 0:
            continue

        y = np.argmax(y_onehot, axis=1)

        if len(y) < cv_folds:
            continue

        model = CatBoostClassifier(verbose=0, random_seed=seed)
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        scores = cross_val_score(model, X_all, y, cv=kf, scoring="accuracy")

        # Ambil top6
        try:
            top6_digits = get_top6_catboost(X_all, y, seed=seed)
            top6_str = ",".join(map(str, top6_digits))
        except:
            top6_str = "-"

        results.append({
            "WS": ws,
            "Accuracy Mean": np.mean(scores),
            "Accuracy Std": np.std(scores),
            "Jumlah Sample": len(y),
            "Top6": top6_str
        })

    df_result = pd.DataFrame(results)
    st.session_state[f"catboost_ws_results_{label}"] = df_result

    progress.progress(1.0, text=f"✅ Selesai scan CatBoost {label.upper()}")
    st.success(f"✅ Scan CatBoost {label.upper()} selesai. Ditemukan {len(df_result)} WS.")
    return df_result


def get_best_ws_from_catboost(label):
    """
    Ambil WS terbaik berdasarkan Accuracy Mean tertinggi dari hasil scan CatBoost.
    """
    df_result = st.session_state.get(f"catboost_ws_results_{label}")
    if df_result is not None and not df_result.empty:
        best_row = df_result.loc[df_result["Accuracy Mean"].idxmax()]
        return int(best_row["WS"]), best_row["Top6"]
    return None, []


def show_catboost_heatmaps(df_result, label):
    if df_result is None or df_result.empty:
        return

    df_result_sorted = df_result.sort_values("WS")

    # Heatmap Akurasi
    st.markdown(f"#### 🎯 Heatmap Akurasi - {label.upper()}")
    fig1, ax1 = plt.subplots(figsize=(8, 1.5))
    acc_df = pd.DataFrame([df_result_sorted["Accuracy Mean"].values], columns=df_result_sorted["WS"].astype(int))
    sns.heatmap(acc_df, annot=True, cmap="YlGn", cbar=False, ax=ax1, fmt=".3f")
    ax1.set_yticklabels(["Acc"])
    st.pyplot(fig1)

    # Heatmap Confidence berdasarkan Top6
    st.markdown(f"#### 🔢 Heatmap Confidence - {label.upper()}")
    top6_all = df_result_sorted["Top6"].apply(lambda x: [int(i) for i in x.split(",") if i.strip().isdigit()])
    digit_counts = {i: 0 for i in range(10)}
    for top6 in top6_all:
        for d in top6:
            digit_counts[d] += 1
    conf_df = pd.DataFrame([digit_counts]).T
    conf_df.columns = ["Top-6 Count"]
    conf_df.index.name = "Digit"
    fig2, ax2 = plt.subplots(figsize=(8, 1.5))
    sns.heatmap(conf_df.T, annot=True, cmap="Oranges", cbar=False, ax=ax2)
    st.pyplot(fig2)

def get_top6_catboost(X, y, seed=42):
    from catboost import CatBoostClassifier
    from collections import Counter

    model = CatBoostClassifier(verbose=0, random_seed=seed)
    model.fit(X, y)
    y_pred = model.predict(X)
    top6 = [int(d) for d, _ in Counter(y_pred).most_common(6)]
    return top6



        
