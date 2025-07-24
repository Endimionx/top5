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

def ensemble_confidence_voting(lstm_dict, catboost_top6, heatmap_counts, weights=[1.2, 1.0, 0.6], min_lstm_conf=0.3):
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
    weight_hybrid, weight_direct = get_stacked_weights(acc_hybrid, acc_direct)
    counter = defaultdict(float)
    for i, d in enumerate(hybrid or []):
        counter[d] += weight_hybrid * np.exp(-(i / 2))
    for i, d in enumerate(pred_direct or []):
        counter[d] += weight_direct * np.exp(-(i / 2))
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
    if total == 0:
        return 0.5, 0.5
    w_hybrid = acc_hybrid / total
    w_direct = acc_direct / total
    return w_hybrid, w_direct

def dynamic_alpha(acc_conf, acc_prob):
    if acc_conf + acc_prob == 0:
        return 0.5
    return acc_conf / (acc_conf + acc_prob)

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
            f.write(f"Final Hybrid: {final}\n")
        f.write("-" * 40 + "\n")

def show_live_accuracy(df, predictions_dict):
    st.markdown("### 📊 Simulasi Live Accuracy")
    col1, col2 = st.columns(2)
    with col1:
        jumlah = st.number_input("Jumlah Data Terakhir", 10, 200, 50, key="live_acc_n")
    with col2:
        kunci = st.selectbox("Jenis Prediksi", list(predictions_dict.keys()), key="live_acc_kunci")

    if kunci not in predictions_dict:
        st.warning("Belum ada hasil prediksi.")
        return

    real = df["angka"].astype(str).apply(lambda x: int(x[DIGIT_LABELS.index(kunci)]))[-jumlah:]
    pred = predictions_dict[kunci]
    benar = sum([r in pred for r in real])
    acc = benar / len(real) if len(real) > 0 else 0
    st.success(f"Akurasi Real ({kunci.upper()}): `{acc:.2%}` dari {jumlah} data terakhir.")

def show_auto_ensemble_adaptive(predictions_dict):
    st.markdown("### 🧠 Auto Ensemble Adaptive")
    for label, pred in predictions_dict.items():
        if pred:
            st.write(f"{label.upper()}: `{pred}`")

def safe_in(item, collection):
    if isinstance(collection, (np.ndarray, list)):
        return item in collection
    return False

def tab3(df, lokasi):
    simulasi_prediksi = {}
    simulasi_target_real = {}

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

    for key in ["tab3_full_results", "tab3_top6_acc", "tab3_top6_conf", "tab3_ensemble", "tab3_ensemble_prob", "tab3_stacked", "tab3_final"]:
        if key not in st.session_state:
            st.session_state[key] = {}

    if st.button("🔎 Scan Per Digit", use_container_width=True):
        st.session_state.tab3_final = {}

        target_digits = DIGIT_LABELS if selected_digit_tab3 == "(Semua)" else [selected_digit_tab3]

        for label in target_digits:
            st.markdown(f"## 🔍 {label.upper()}")
            try:
                result_df = scan_ws_catboost(df, label, min_ws_cb3, max_ws_cb3, folds_cb3, temp_seed)

                top3_acc = result_df.sort_values("Accuracy Mean", ascending=False).head(3)
                top6_acc_dict = {}
                for _, row in top3_acc.iterrows():
                    ws = int(row["WS"])
                    try:
                        model = train_temp_lstm_model(df, label, ws, temp_seed)
                        top6, probs = get_top6_lstm_temp(model, df, ws)
                        top6_acc_dict[ws] = (top6, probs)
                    except:
                        top6_acc_dict[ws] = ([], [])
                st.session_state.tab3_top6_acc[label] = top6_acc_dict

                top3_conf = result_df.sort_values("Top6 Conf", ascending=False).head(3)
                top6_conf_dict = {}
                for _, row in top3_conf.iterrows():
                    ws = int(row["WS"])
                    try:
                        model = train_temp_lstm_model(df, label, ws, temp_seed)
                        top6, probs = get_top6_lstm_temp(model, df, ws)
                        top6_conf_dict[ws] = (top6, probs)
                    except:
                        top6_conf_dict[ws] = ([], [])
                st.session_state.tab3_top6_conf[label] = top6_conf_dict

                best_row = result_df.loc[result_df["Accuracy Mean"].idxmax()]
                best_ws = int(best_row["WS"])
                acc_conf = best_row["Accuracy Mean"]
                st.session_state.tab3_full_results[label] = {"ws": best_ws, "acc": acc_conf}

                lstm_dict = st.session_state.tab3_top6_acc.get(label, {})
                catboost_top6_all = [d for ws in st.session_state.tab3_top6_conf[label].values() for d in ws[0]]

                top6_all = result_df["Top6"].apply(lambda x: [int(i) for i in str(x).split(",") if i.strip().isdigit()])
                heatmap_counts = Counter()
                for t in top6_all:
                    heatmap_counts.update(t)

                final_ens_conf = ensemble_confidence_voting(
                    lstm_dict, catboost_top6_all, heatmap_counts,
                    weights=[lstm_weight, cb_weight, hm_weight],
                    min_lstm_conf=lstm_min_conf
                )
                st.session_state.tab3_ensemble[label] = final_ens_conf

                all_probs = [probs for _, probs in lstm_dict.values() if probs is not None and len(probs) > 0]
                acc_prob = np.mean(result_df.sort_values("Accuracy Mean", ascending=False)["Accuracy Mean"].head(3)) if all_probs else 0.0
                final_ens_prob = ensemble_probabilistic(all_probs, [acc_conf]*len(all_probs)) if all_probs else []

                alpha_used = alpha_manual if hybrid_mode == "Manual Alpha" else dynamic_alpha(acc_conf, acc_prob)
                hybrid = hybrid_voting(final_ens_conf, final_ens_prob, alpha_used)

                try:
                    model = train_temp_lstm_model(df, label, best_ws, temp_seed)
                    top6_direct, probs = get_top6_lstm_temp(model, df, best_ws)
                    if probs is not None and np.max(probs) < 0.3:
                        top6_direct = []
                except:
                    top6_direct = []

                stacked = stacked_hybrid_auto(hybrid, top6_direct, acc_conf, acc_conf)
                markov_all = top6_markov_hybrid(df)
                markov_top6 = markov_all[DIGIT_LABELS.index(label)]

                final_hybrid = final_ensemble_with_markov(stacked, markov_top6)
                st.session_state.tab3_stacked[label] = final_hybrid
                st.session_state.tab3_final[label] = final_hybrid

                # Simulasi
                simulasi_prediksi[label] = stacked
                real_digit = int(df["angka"].astype(str).iloc[-1][DIGIT_LABELS.index(label)])
                simulasi_target_real[label] = real_digit

                log_prediction(label, final_ens_conf, final_ens_prob, hybrid, alpha_used, stacked, final_hybrid, lokasi)

                st.markdown(f"### 🧠 Final Ensemble Top6 - {label.upper()}")
                st.write(f"Confidence Voting: `{final_ens_conf}`")
                st.write(f"Probabilistic Voting: `{final_ens_prob}`")
                st.write(f"Hybrid Voting (α={alpha_used:.2f}): `{hybrid}`")
                st.write(f"Stacked Hybrid: `{stacked}`")
                st.success(f"📌 Final Hybrid + Markov: `{final_hybrid}`")

            except Exception as e:
                st.error(f"❌ Gagal proses {label.upper()}: {e}")
                st.session_state.tab3_stacked[label] = []
                st.session_state.tab3_final[label] = []

    show_live_accuracy(df, st.session_state.tab3_final)
    show_auto_ensemble_adaptive(st.session_state.tab3_final)

    # Tampilkan hasil simulasi prediksi top-6 untuk semua posisi
    st.markdown("### 🎯 Hasil Prediksi Simulasi (Top-6 per Posisi)")
    simulasi_tabel = []
    digit_order = ['RIBUAN', 'RATUSAN', 'PULUHAN', 'SATUAN']
    for posisi in digit_order:
        top6 = simulasi_prediksi.get(posisi, [])
        real = simulasi_target_real.get(posisi, None)
        if top6 and real is not None:
            simulasi_tabel.append({
                "Posisi": posisi,
                "Top-6": ", ".join(str(d) for d in top6),
                "Target Real": real,
                "Match (in Top-6)": "✅" if safe_in(real, top6) else "❌"
            })
    if simulasi_tabel:
        st.table(pd.DataFrame(simulasi_tabel))

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
