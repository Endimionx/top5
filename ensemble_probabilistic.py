from collections import defaultdict

def ensemble_probabilistic(
    top6_lstm_conf: dict,
    catboost_accuracies: list,
    heatmap_weights: dict = None,
    weight_lstm: float = 1.2,
    weight_cb: float = 1.0,
    weight_heatmap: float = 0.8
):
    """
    Menggabungkan prediksi LSTM, CatBoost, dan Heatmap Top-6 ke dalam skor total,
    lalu mengembalikan Top-6 digit terbaik.

    Args:
        top6_lstm_conf (dict): Mapping digit -> confidence (float) dari model LSTM.
        catboost_accuracies (list of tuples): List (digit, accuracy_score) dari hasil CatBoost.
        heatmap_weights (dict, optional): Mapping digit -> count/score dari heatmap Top6.
        weight_lstm (float): Bobot skor dari LSTM.
        weight_cb (float): Bobot skor dari CatBoost.
        weight_heatmap (float): Bobot skor dari heatmap.

    Returns:
        top6_final (list): Daftar Top-6 digit dengan skor tertinggi.
        score_dict (dict): Dictionary digit -> total score.
    """

    score_dict = defaultdict(float)

    # 🔹 LSTM Confidence
    for digit, conf in top6_lstm_conf.items():
        score_dict[int(digit)] += conf * weight_lstm

    # 🔹 CatBoost Accuracy
    for digit, acc in catboost_accuracies:
        score_dict[int(digit)] += acc * weight_cb

    # 🔹 Heatmap (jika ada)
    if heatmap_weights:
        for digit, count in heatmap_weights.items():
            score_dict[int(digit)] += count * weight_heatmap

    # 🔹 Ranking final
    top6_final = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:6]
    top6_digits = [digit for digit, _ in top6_final]

    return top6_digits, dict(score_dict)

