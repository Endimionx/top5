import numpy as np
from collections import defaultdict

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def ensemble_probabilistic(probs_list, catboost_accuracies=None):
    """
    Menggabungkan prediksi probabilistik dari beberapa model.
    """
    score = defaultdict(float)

    for i, probs in enumerate(probs_list):
        if probs is None or len(probs) != 10:
            continue

        # Normalisasi jika belum softmax
        if not np.isclose(np.sum(probs), 1.0):
            probs = softmax(probs)

        weight = catboost_accuracies[i] if catboost_accuracies and i < len(catboost_accuracies) else 1.0

        for d, p in enumerate(probs):
            score[d] += weight * p

    if not score:
        return []

    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)
    return [digit for digit, _ in ranked[:6]]
