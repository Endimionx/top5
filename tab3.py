import streamlit as st
import pandas as pd
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
from markov_model import top6_markov_hybrid

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def ensemble_confidence_voting(lstm_dict, catboost_top6, heatmap_counts,
                                weights=[1.2, 1.0, 0.6], min_lstm_conf=0.3):
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
    counter = defaultdict(float)
    for i, d in enumerate(conf or []):
        counter[d] += alpha * (6 - i)
    for i, d in enumerate(prob or []):
        counter[d] += (1 - alpha) * (6 - i)
    ranked = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:6]]

def stacked_hybrid_auto(hybrid, pred_direct, acc_hybrid=0.6, acc_direct=0.4):
    w_hybrid, w_direct = get_stacked_weights(acc_hybrid, acc_direct)
    counter = defaultdict(float)
    for i, d in enumerate(hybrid or []):
        counter[d] += w_hybrid * np.exp(-(i / 2))
    for i, d in enumerate(pred_direct or []):
        counter[d] += w_direct * np.exp(-(i / 2))
    ranked = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:6]]

def final_ensemble_with_markov(stacked, markov):
    counter = defaultdict(float)
    for i, d in enumerate(stacked or []):
        counter[d] += (6 - i)
    for i, d in enumerate(markov or []):
        counter[d] += (6 - i)
    ranked = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:6]]

def get_stacked_weights(acc_hybrid, acc_direct):
    total = acc_hybrid + acc_direct
    return (acc_hybrid / total, acc_direct / total) if total else (0.5, 0.5)

def dynamic_alpha(acc_conf, acc_prob):
    return acc_conf / (acc_conf + acc_prob) if (acc_conf + acc_prob) else 0.5

def log_prediction(label, conf, prob, hybrid, alpha, stacked=None, final=None, lokasi=None):
    with open("log_tab3.txt", "a") as f:
        f.write(f"[{label.upper()}] | Lokasi: {lokasi}\n" if lokasi else f"[{label.upper()}]\n")
        f.write(f"Confidence Voting: {conf}\n")
        f.write(f"Probabilistic Voting: {prob}\n")
        f.write(f"Hybrid Voting (α={alpha:.2f}): {hybrid}\n")
        if stacked: f.write(f"Stacked Hybrid: {stacked}\n")
        if final: f.write(f"Final Hybrid: {final}\n")
        f.write("-" * 40 + "\n")

def show_live_accuracy(df, pred_dict):
    st.markdown("### 📊 Simulasi Live Accuracy")
    col1, col2 = st.columns(2)
    jumlah = col1.number_input("Jumlah Data Terakhir", 10, 200, 50, key="live_acc_n")
    kunci = col2.selectbox("Jenis Prediksi", list(pred_dict.keys()), key="live_acc_kunci")

    if kunci not in pred_dict: return st.warning("Belum ada hasil.")
    real = df["angka"].astype(str).apply(lambda x: int(x[DIGIT_LABELS.index(kunci)]))[-jumlah:]
    pred = pred_dict[kunci]
    benar = sum([r in pred for r in real])
    st.success(f"Akurasi Real ({kunci.upper()}): `{benar / len(real):.2%}` dari {jumlah} data.")

def show_auto_ensemble_adaptive(pred_dict):
    st.markdown("### 🧠 Auto Ensemble Adaptive")
    for label, pred in pred_dict.items():
        if pred:
            st.write(f"{label.upper()}: `{pred}`")

def tab3(df, lokasi):
    st.markdown("## 🎯 Prediksi Per Digit")
    min_ws = st.number_input("Min WS", 3, 20, 5)
    max_ws = st.number_input("Max WS", min_ws+1, 30, min_ws+6)
    folds = st.slider("Jumlah Fold", 2, 10, 3)
    seed = st.number_input("Seed", 0, 9999, 42)

    st.markdown("### ⚖️ Bobot Voting")
    lstm_w = st.slider("LSTM Weight", 0.5, 2.0, 1.2, 0.1)
    cb_w = st.slider("CatBoost Weight", 0.5, 2.0, 1.0, 0.1)
    hm_w = st.slider("Heatmap Weight", 0.0, 1.0, 0.6, 0.1)
    min_conf = st.slider("Min Confidence LSTM", 0.0, 1.0, 0.3, 0.05)

    hybrid_mode = st.selectbox("Mode Hybrid Voting", ["Dynamic Alpha", "Manual Alpha"])
    if hybrid_mode == "Manual Alpha":
        alpha_manual = st.slider("Alpha Manual", 0.0, 1.0, 0.5, 0.05)

    digit = st.selectbox("Digit yang ingin discan", ["(Semua)"] + DIGIT_LABELS)

    for key in ["tab3_stacked", "tab3_final", "simulasi_prediksi", "simulasi_target_real"]:
        if key not in st.session_state:
            st.session_state[key] = {}

    if st.button("🔎 Scan Per Digit", use_container_width=True):
        st.session_state.tab3_final = {}
        target_digits = DIGIT_LABELS if digit == "(Semua)" else [digit]
        simulasi_pred = {}
        simulasi_real = {}

        for label in target_digits:
            st.markdown(f"### 🔍 {label.upper()}")
            try:
                result_df = scan_ws_catboost(df, label, min_ws, max_ws, folds, seed)
                best_row = result_df.loc[result_df["Accuracy Mean"].idxmax()]
                best_ws = int(best_row["WS"])
                acc_conf = best_row["Accuracy Mean"]

                lstm_dict = {}
                for _, row in result_df.sort_values("Accuracy Mean", ascending=False).head(3).iterrows():
                    ws = int(row["WS"])
                    try:
                        model = train_temp_lstm_model(df, label, ws, seed)
                        top6, probs = get_top6_lstm_temp(model, df, ws)
                        lstm_dict[ws] = (top6, probs)
                    except:
                        lstm_dict[ws] = ([], [])

                catboost_top6_all = [d for ws in lstm_dict.values() for d in ws[0]]
                heatmap_counts = Counter()
                for t in result_df["Top6"].apply(lambda x: [int(i) for i in str(x).split(",") if i.strip().isdigit()]): heatmap_counts.update(t)

                conf = ensemble_confidence_voting(lstm_dict, catboost_top6_all, heatmap_counts,
                                                  weights=[lstm_w, cb_w, hm_w], min_lstm_conf=min_conf)

                all_probs = [probs for _, probs in lstm_dict.values() if probs is not None]
                acc_prob = np.mean(result_df.sort_values("Accuracy Mean", ascending=False)["Accuracy Mean"].head(3))
                prob = ensemble_probabilistic(all_probs, [acc_conf]*len(all_probs)) if all_probs else []

                alpha_used = alpha_manual if hybrid_mode == "Manual Alpha" else dynamic_alpha(acc_conf, acc_prob)
                hybrid = hybrid_voting(conf, prob, alpha_used)

                try:
                    model = train_temp_lstm_model(df, label, best_ws, seed)
                    top6_direct, probs = get_top6_lstm_temp(model, df, best_ws)
                    if probs is not None and np.max(probs) < 0.3: top6_direct = []
                except:
                    top6_direct = []

                stacked = stacked_hybrid_auto(hybrid, top6_direct, acc_conf, acc_conf)
                markov_top6 = top6_markov_hybrid(df)[DIGIT_LABELS.index(label)]
                final = final_ensemble_with_markov(stacked, markov_top6)

                st.session_state.tab3_stacked[label] = stacked
                st.session_state.tab3_final[label] = final

                log_prediction(label, conf, prob, hybrid, alpha_used, stacked, final, lokasi)

                st.write(f"Confidence: `{conf}`")
                st.write(f"Probabilistic: `{prob}`")
                st.write(f"Hybrid α={alpha_used:.2f}: `{hybrid}`")
                st.write(f"Stacked Hybrid: `{stacked}`")
                st.success(f"📌 Final + Markov: `{final}`")

                real_digit = int(str(df.iloc[-1]["angka"])[DIGIT_LABELS.index(label)])
                simulasi_pred[label] = final
                simulasi_real[label] = real_digit

            except Exception as e:
                st.error(f"Gagal {label.upper()}: {e}")
                st.session_state.tab3_stacked[label] = []
                st.session_state.tab3_final[label] = []

        st.session_state.simulasi_prediksi = simulasi_pred
        st.session_state.simulasi_target_real = simulasi_real

    show_live_accuracy(df, st.session_state.tab3_final)
    show_auto_ensemble_adaptive(st.session_state.tab3_final)

    st.markdown("### 🎯 Hasil Prediksi Simulasi (Top-6 per Posisi)")
    simulasi_tabel = []
    for pos in DIGIT_LABELS:
        top6 = st.session_state.simulasi_prediksi.get(pos, [])
        real = st.session_state.simulasi_target_real.get(pos, None)
        if top6:
            simulasi_tabel.append({
                "Posisi": pos,
                "Top-6": ", ".join(str(d) for d in top6),
                "Target Real": real,
                "Match": "✅" if real in top6 else "❌"
            })
    if simulasi_tabel:
        st.table(pd.DataFrame(simulasi_tabel))

    st.markdown("---")
    col1, col2 = st.columns(2)
    if col1.button("📄 Lihat Log Prediksi", use_container_width=True):
        if os.path.exists("log_tab3.txt"):
            with open("log_tab3.txt", "r") as f: st.code(f.read())
        else: st.info("Belum ada log.")
    if col2.button("🧹 Hapus Log", use_container_width=True):
        if os.path.exists("log_tab3.txt"):
            os.remove("log_tab3.txt")
            st.success("Log dihapus.")
        else: st.info("Tidak ada log.")
