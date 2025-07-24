import numpy as np
from collections import defaultdict

def ensemble_probabilistic(probs_list, catboost_accuracies=None):
    from collections import defaultdict
    score = defaultdict(float)

    for i, probs in enumerate(probs_list):
        # Validasi panjang
        if probs is None or len(probs) != 10:
            continue

        # Normalisasi jika belum
        if not np.isclose(np.sum(probs), 1.0):
            probs = softmax(probs)

        # Ambil bobot
        weight = catboost_accuracies[i] if catboost_accuracies and i < len(catboost_accuracies) else 1.0

        for d, p in enumerate(probs):
            score[d] += weight * p

    if not score:
        return []

    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)
    return [digit for digit, _ in ranked[:6]]
