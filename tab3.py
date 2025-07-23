import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from collections import Counter, defaultdict
from ensemble_probabilistic import ensemble_probabilistic
from ws_scan_catboost import (
    scan_ws_catboost,
    train_temp_lstm_model,
    get_top6_lstm_temp,
    DIGIT_LABELS,
)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def ensemble_confidence_voting(
    lstm_dict, catboost_top6, heatmap_counts,
    weights=[1.2, 1.0, 0.6],
    min_lstm_conf=0.3
):
    score = defaultdict(float)
    for ws, (digits, confs) in lstm_dict.items():
        if not digits or confs is None or len(confs) == 0:
            continue
        if max(confs) < min_lstm_conf:
            continue
        norm_confs = softmax(confs)
        for d, c in zip(digits, norm_confs):
            score[d] += weights[0] * c
    for d in catboost_top6:
        score[d] += weights[1]
    for d, count in heatmap_counts.items():
        score[d] += weights[2] * count
    if not score:
        return []
    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:6]]

def hybrid_voting(conf, prob, alpha=0.5):
    if not conf and not prob:
        return []
    counter = defaultdict(float)
    for i, d in enumerate(conf):
        counter[d] += alpha * (6 - i)
    for i, d in enumerate(prob):
        counter[d] += (1 - alpha) * (6 - i)
    ranked = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:6]]

def stacked_hybrid(hybrid, pred_direct):
    counter = defaultdict(float)
    for i, d in enumerate(hybrid):
        counter[d] += (6 - i)
    for i, d in enumerate(pred_direct):
        counter[d] += (6 - i)
    ranked = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:6]]
    
def dynamic_alpha(acc_conf, acc_prob):
    if acc_conf + acc_prob == 0:
        return 0.5
    return acc_conf / (acc_conf + acc_prob)

def log_prediction(label, conf, prob, hybrid, alpha, stacked=None):
    log_path = "log_tab3.txt"
    with open(log_path, "a") as f:
        f.write(f"[{label.upper()}]\n")
        f.write(f"Confidence Voting: {conf}\n")
        f.write(f"Probabilistic Voting: {prob}\n")
        f.write(f"Hybrid Voting (α={alpha:.2f}): {hybrid}\n")
        if stacked:
            f.write(f"Stacked Hybrid: {stacked}\n")
        f.write("-" * 40 + "\n")

def meta_learning(votes):
    # Menggunakan voting terbaik berdasarkan skor akurasi
    max_vote = max(votes, key=lambda x: x[1])
    return max_vote[0]

def calibration(confidences, probs):
    # Kalibrasi dengan min-max scaling
    min_conf, max_conf = min(confidences), max(confidences)
    conf_calibrated = [(c - min_conf) / (max_conf - min_conf) for c in confidences]
    prob_calibrated = [(p - min(probs)) / (max(probs) - min(probs)) for p in probs]
    return conf_calibrated, prob_calibrated

def auto_ml_ws_selection(ws_results):
    # Memilih WS terbaik berdasarkan stabilitas
    best_ws = max(ws_results, key=lambda x: x['accuracy'] - x['std'])
    return best_ws

def tab3(df):
    min_ws_cb3 = st.number_input("🔁 Min WS", 3, 20, 5, key="tab3_min_ws")
    max_ws_cb3 = st.number_input("🔁 Max WS", min_ws_cb3 + 1, 30, min_ws_cb3 + 6, key="tab3_max_ws")
    folds_cb3 = st.slider("📂 Jumlah Fold", 2, 10, 3, key="tab3_cv")
    temp_seed = st.number_input("🎲 Seed", 0, 9999, 42, key="tab3_seed")

    st.markdown("### ⚖️ Bobot Voting")
    lstm_weight = st.slider("LSTM Weight", 0.5, 2.0, 1.2, 0.1, key="tab3_lstm_weight")
    cb_weight = st.slider("CatBoost Weight", 0.5, 2.0, 1.0, 0.1, key="tab3_cb_weight")
    hm_weight = st.slider("Heatmap Weight", 0.0, 1.0, 0.6, 0.1, key="tab3_hm_weight")
    lstm_min_conf = st.slider("Min Confidence LSTM", 0.0, 1.0, 0.3, 0.05, key="tab3_min_conf")

    st.markdown("### 🔧 Hybrid Voting Option")
    hybrid_mode = st.selectbox("Mode Hybrid Voting", ["Dynamic Alpha", "Manual Alpha"], key="tab3_hybrid_mode")
    if hybrid_mode == "Manual Alpha":
        alpha_manual = st.slider("Alpha Manual", 0.0, 1.0, 0.5, 0.05, key="tab3_alpha")

    for key in ["tab3_full_results", "tab3_top6_acc", "tab3_top6_conf", "tab3_ensemble", "tab3_ensemble_prob", "tab3_hybrid", "tab3_stacked"]:
        if key not in st.session_state:
            st.session_state[key] = {}

    selected_digit_tab3 = st.selectbox("Pilih digit yang ingin discan", ["(Semua)"] + DIGIT_LABELS, key="tab3_selected_digit")

    if st.button("🔎 Scan Per Digit", use_container_width=True):
        for key in ["tab3_full_results", "tab3_top6_acc", "tab3_top6_conf", "tab3_ensemble", "tab3_ensemble_prob", "tab3_hybrid", "tab3_stacked"]:
            st.session_state[key] = {}

        target_digits = DIGIT_LABELS if selected_digit_tab3 == "(Semua)" else [selected_digit_tab3]

        for label in target_digits:
            st.markdown(f"## 🔍 {label.upper()}")
            try:
                result_df = scan_ws_catboost(df, label, min_ws_cb3, max_ws_cb3, folds_cb3, temp_seed)

                top3_acc = result_df.sort_values("Accuracy Mean", ascending=False).head(3)
                st.session_state.tab3_top6_acc[label] = {}
                for _, row in top3_acc.iterrows():
                    ws = int(row["WS"])
                    try:
                        model = train_temp_lstm_model(df, label, ws, temp_seed)
                        top6, probs = get_top6_lstm_temp(model, df, ws)
                        st.session_state.tab3_top6_acc[label][ws] = (top6, probs)
                    except:
                        st.session_state.tab3_top6_acc[label][ws] = ([], [])

                top3_conf = result_df.sort_values("Top6 Conf", ascending=False).head(3)
                st.session_state.tab3_top6_conf[label] = {}
                for _, row in top3_conf.iterrows():
                    ws = int(row["WS"])
                    try:
                        model = train_temp_lstm_model(df, label, ws, temp_seed)
                        top6, probs = get_top6_lstm_temp(model, df, ws)
                        st.session_state.tab3_top6_conf[label][ws] = (top6, probs)
                    except:
                        st.session_state.tab3_top6_conf[label][ws] = ([], [])

                best_row = result_df.loc[result_df["Accuracy Mean"].idxmax()]
                best_ws = int(best_row["WS"])
                st.session_state.tab3_full_results[label] = {
                    "ws": best_ws,
                    "acc": best_row["Accuracy Mean"],
                    "result_df": result_df,
                }

                lstm_dict = st.session_state.tab3_top6_acc.get(label, {})
                if not isinstance(lstm_dict, dict):
                    lstm_dict = {}

                catboost_top6_all = []
                for ws_data in st.session_state.tab3_top6_conf[label].values():
                    catboost_top6_all.extend(ws_data[0])

                top6_all = result_df["Top6"].apply(lambda x: [int(i) for i in str(x).split(",") if i.strip().isdigit()])
                heatmap_counts = Counter()
                for t in top6_all:
                    heatmap_counts.update(t)

                final_ens_conf = ensemble_confidence_voting(
                    lstm_dict,
                    catboost_top6_all,
                    heatmap_counts,
                    weights=[lstm_weight, cb_weight, hm_weight],
                    min_lstm_conf=lstm_min_conf
                )
                st.session_state.tab3_ensemble[label] = final_ens_conf

                all_probs = []
                for _, (digits, probs) in lstm_dict.items():
                    if probs is not None and len(probs) > 0:
                        all_probs.append(probs)

                if all_probs:
                    catboost_accuracies = list(result_df.sort_values("Accuracy Mean", ascending=False)["Accuracy Mean"].head(3))
                    final_ens_prob = ensemble_probabilistic(all_probs, catboost_accuracies)
                    st.session_state.tab3_ensemble_prob[label] = final_ens_prob
                else:
                    final_ens_prob = []

                acc_conf = best_row["Accuracy Mean"]
                acc_prob = np.mean(catboost_accuracies) if all_probs else 0.0
                alpha_used = alpha_manual if hybrid_mode == "Manual Alpha" else dynamic_alpha(acc_conf, acc_prob)
                hybrid = hybrid_voting(final_ens_conf, final_ens_prob, alpha=alpha_used)
                st.session_state.tab3_hybrid[label] = hybrid

                # Langsung prediksi pakai WS terbaik
                try:
                    model = train_temp_lstm_model(df, label, best_ws, temp_seed)
                    top6_direct, _ = get_top6_lstm_temp(model, df, best_ws)
                except:
                    top6_direct = []

                final_stacked = stacked_hybrid(hybrid, top6_direct)
                st.session_state.tab3_stacked[label] = final_stacked

                log_prediction(label, final_ens_conf, final_ens_prob, hybrid, alpha_used, final_stacked)

                st.markdown(f"### 🧠 Final Ensemble Top6 - {label.upper()}")
                st.write(f"Confidence Voting: `{final_ens_conf}`")
                st.write(f"Probabilistic Voting: `{final_ens_prob}`")
                st.write(f"Hybrid Voting (α={alpha_used:.2f}): `{hybrid}`")
                st.success(f"📌 Stacked Hybrid (Hybrid + Prediksi Langsung): `{final_stacked}`")

            except Exception as e:
                st.error(f"❌ Gagal proses {label.upper()}: {e}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 Lihat Log Prediksi", use_container_width=True):
            log_path = "log_tab3.txt"
            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    st.code(f.read(), language="text")
            else:
                st.info("Belum ada log tersimpan.")
    with col2:
        if st.button("🧹 Hapus Log", use_container_width=True):
            log_path = "log_tab3.txt"
            if os.path.exists(log_path):
                os.remove(log_path)
                st.success("Log berhasil dihapus.")
            else:
                st.info("Tidak ada log yang bisa dihapus.")
    
