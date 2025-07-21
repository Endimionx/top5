import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score, KFold

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
    Mencari window size terbaik menggunakan model CatBoostClassifier.
    Mengembalikan DataFrame hasil akurasi per window size.
    """
    np.random.seed(seed)
    results = []

    for ws in range(min_ws, max_ws + 1):
        X, y = create_features_targets(df, ws, label)
        if len(X) == 0:
            continue
        model = CatBoostClassifier(verbose=0, random_seed=seed)
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")
        results.append({
            "WS": ws,
            "Accuracy Mean": np.mean(scores),
            "Accuracy Std": np.std(scores),
            "Jumlah Sample": len(y)
        })

    return pd.DataFrame(results)
