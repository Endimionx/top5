import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import KFold, cross_val_score
from catboost import CatBoostClassifier

def create_features_targets(df, window_size, target_digit):
    """
    Membuat fitur dan target dari data angka 4D untuk digit tertentu.
    """
    data = [list(map(int, list(angka))) for angka in df["angka"] if angka.isdigit() and len(angka) == 4]
    X, y = [], []

    for i in range(window_size, len(data)):
        window = data[i - window_size:i]
        flat = np.array(window).flatten()
        X.append(flat)
        digit = data[i][get_digit_index(target_digit)]
        y.append(digit)

    return np.array(X), np.array(y)

def get_digit_index(label):
    """
    Mengubah label digit ke index dalam angka 4D.
    """
    return {"ribuan": 0, "ratusan": 1, "puluhan": 2, "satuan": 3}.get(label, 0)

def scan_ws_catboost(df, label, min_ws=5, max_ws=15, cv_folds=3, seed=42):
    """
    Mencari window size terbaik menggunakan CatBoostClassifier.
    Menampilkan info, progress bar, dan menyimpan hasil ke session_state.
    """
    from catboost import CatBoostClassifier
    from sklearn.model_selection import KFold, cross_val_score

    np.random.seed(seed)
    results = []

    total = max_ws - min_ws + 1
    progress = st.progress(0.0, text=f"⏳ Mulai proses SCAN CatBoost {label.upper()}...")
    
    for idx, ws in enumerate(range(min_ws, max_ws + 1), 1):
        progress.progress(idx / total, text=f"🔄 Evaluasi WS={ws} untuk {label.upper()}")
        st.info(f"🔍 Sedang proses WS={ws} ({idx}/{total})", icon="🔁")

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

        results.append({
            "WS": ws,
            "Accuracy Mean": np.mean(scores),
            "Accuracy Std": np.std(scores),
            "Jumlah Sample": len(y)
        })

    df_result = pd.DataFrame(results)

    # Simpan ke session_state
    st.session_state[f"catboost_ws_results_{label}"] = df_result

    progress.progress(1.0, text=f"✅ Selesai scan CatBoost {label.upper()}")
    st.success(f"✅ Scan CatBoost {label.upper()} selesai. Ditemukan {len(df_result)} WS.")
    return df_result
