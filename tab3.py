import streamlit as st
import pandas as pd
import numpy as np
import os
from collections import defaultdict, Counter
from ensemble_probabilistic import ensemble_probabilistic
from ws_scan_catboost import (
    scan_ws_catboost,
    train_temp_lstm_model,
    get_top6_lstm_temp,
    DIGIT_LABELS,
)
from markov_model import top6_markov_hybrid

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def ensemble_confidence_voting(lstm_dict, catboost_top6, heatmap_counts, weights=[1.2, 1.0, 0.6], min_lstm_conf=0.3):
    score = defaultdict(float)
    for ws, (digits, confs) in lstm_dict.items():
        if digits and confs is not None and len(confs) > 0:
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

def stacked_hybrid_auto(hybrid, pred_direct, acc_hybrid=0.6, acc_direct=0.4):
    weight_hybrid, weight_direct = get_stacked_weights(acc_hybrid, acc_direct)
    counter = defaultdict(float)
    for i, d in enumerate(hybrid):
        counter[d] += weight_hybrid * np.exp(-(i / 2))
    for i, d in enumerate(pred_direct):
        counter[d] += weight_direct * np.exp(-(i / 2))
    ranked = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:6]]

def get_stacked_weights(acc_hybrid, acc_direct):
    total = acc_hybrid + acc_direct
    if total == 0:
        return 0.5, 0.5
    return acc_hybrid / total, acc_direct / total

def final_ensemble_with_markov(stacked, markov):
    counter = defaultdict(float)
    for i, d in enumerate(stacked):
        counter[d] += (6 - i)
    for i, d in enumerate(markov):
        counter[d] += (6 - i)
    ranked = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:6]]

def dynamic_alpha(acc_conf, acc_prob):
    total = acc_conf + acc_prob
    return acc_conf / total if total > 0 else 0.5

def log_prediction(label, conf, prob, hybrid, alpha, stacked=None, final=None, lokasi=None):
    log_path = "log_tab3.txt"
    with open(log_path, "a") as f:
        if lokasi:
            f.write(f"[{label.upper()}] | Lokasi: {lokasi}\n")
        else:
            f.write(f"[{label.upper()}]\n")
        f.write(f"Confidence Voting: {conf}\n")
        f.write(f"Probabilistic Voting: {prob}\n")
        f.write(f"Hybrid Voting (α={alpha:.2f}): {hybrid}\n")
        if stacked:
            f.write(f"Stacked Hybrid: {stacked}\n")
        if final:
            f.write(f"Final Hybrid + Markov: {final}\n")
        f.write("-" * 40 + "\n")

def simulate_live_accuracy(real_digit, pred_top6):
    return 1 if real_digit in pred_top6 else 0

def tab3(df, lokasi):
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

    selected_digit_tab3 = st.selectbox("Pilih digit yang ingin discan", ["(Semua)"] + DIGIT_LABELS, key="tab3_selected_digit")

    if st.button("🔎 Scan Per Digit", use_container_width=True):
        target_digits = DIGIT_LABELS if selected_digit_tab3 == "(Semua)" else [selected_digit_tab3]
        markov_all = top6_markov_hybrid(df)

        for label in target_digits:
            st.markdown(f"## 🔍 {label.upper()}")
            try:
                result_df = scan_ws_catboost(df, label, min_ws_cb3, max_ws_cb3, folds_cb3, temp_seed)
                top3_acc = result_df.sort_values("Accuracy Mean", ascending=False).head(3)

                lstm_dict = {}
                for _, row in top3_acc.iterrows():
                    ws = int(row["WS"])
                    try:
                        model = train_temp_lstm_model(df, label, ws, temp_seed)
                        top6, probs = get_top6_lstm_temp(model, df, ws)
                        lstm_dict[ws] = (top6, probs)
                    except:
                        lstm_dict[ws] = ([], [])

                catboost_top6 = []
                for _, row in result_df.iterrows():
                    if isinstance(row["Top6"], str):
                        catboost_top6.extend([int(i) for i in row["Top6"].split(",") if i.isdigit()])

                heatmap_counts = Counter(catboost_top6)
                final_ens_conf = ensemble_confidence_voting(lstm_dict, catboost_top6, heatmap_counts,
                                                            weights=[lstm_weight, cb_weight, hm_weight],
                                                            min_lstm_conf=lstm_min_conf)

                all_probs = [probs for _, probs in lstm_dict.values() if probs is not None and len(probs) > 0]
                if all_probs:
                    accs = list(result_df["Accuracy Mean"].head(len(all_probs)))
                    final_ens_prob = ensemble_probabilistic(all_probs, accs)
                else:
                    final_ens_prob = []

                acc_conf = result_df["Accuracy Mean"].max()
                acc_prob = np.mean(accs) if all_probs else 0.0

                alpha_used = alpha_manual if hybrid_mode == "Manual Alpha" else dynamic_alpha(acc_conf, acc_prob)
                hybrid = hybrid_voting(final_ens_conf, final_ens_prob, alpha=alpha_used)

                # pred direct
                best_ws = int(result_df.loc[result_df["Accuracy Mean"].idxmax()]["WS"])
                try:
                    model = train_temp_lstm_model(df, label, best_ws, temp_seed)
                    pred_direct, probs = get_top6_lstm_temp(model, df, best_ws)
                    if probs is not None and np.max(probs) < lstm_min_conf:
                        pred_direct = []
                except:
                    pred_direct = []

                stacked = stacked_hybrid_auto(hybrid, pred_direct, acc_hybrid=acc_conf, acc_direct=acc_conf)
                markov_top6 = markov_all[DIGIT_LABELS.index(label)]
                final_hybrid = final_ensemble_with_markov(stacked, markov_top6)

                # Simulasi Live Accuracy
                real_digit = int(str(df["angka"].iloc[-1]).zfill(4)[DIGIT_LABELS.index(label)])
                acc_sim = simulate_live_accuracy(real_digit, final_hybrid)
                st.info(f"🎯 Simulasi Live Accuracy: `{acc_sim}` (Real: {real_digit})")

                log_prediction(label, final_ens_conf, final_ens_prob, hybrid, alpha_used, stacked, final_hybrid, lokasi)

                st.write(f"Confidence Voting: `{final_ens_conf}`")
                st.write(f"Probabilistic Voting: `{final_ens_prob}`")
                st.write(f"Hybrid Voting (α={alpha_used:.2f}): `{hybrid}`")
                st.write(f"Stacked Hybrid: `{stacked}`")
                st.success(f"📌 Final Hybrid + Markov: `{final_hybrid}`")

            except Exception as e:
                st.error(f"❌ Gagal proses {label.upper()}: {e}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 Lihat Log Prediksi", use_container_width=True):
            if os.path.exists("log_tab3.txt"):
                with open("log_tab3.txt", "r") as f:
                    st.code(f.read(), language="text")
            else:
                st.info("Belum ada log tersimpan.")
    with col2:
        if st.button("🧹 Hapus Log", use_container_width=True):
            if os.path.exists("log_tab3.txt"):
                os.remove("log_tab3.txt")
                st.success("Log berhasil dihapus.")
            else:
                st.info("Tidak ada log yang bisa dihapus.")
