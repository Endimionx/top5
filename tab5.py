# tab5_full.py
import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from ws_scan_catboost import scan_ws_catboost, train_temp_lstm_model, get_top6_lstm_temp, DIGIT_LABELS
from ensemble_probabilistic import ensemble_probabilistic
from markov_model import top6_markov_hybrid

def softmax(x): e_x = np.exp(x - np.max(x)); return e_x / e_x.sum()

def detect_anomaly(df, window=10):
    if len(df) < window + 1: return False
    digits = df["angka"].astype(str).apply(lambda x: [int(d) for d in x])[-(window+1):]
    std = np.std(np.array(digits), axis=0)
    return np.any(std > 2.0)

def ensemble_confidence(lstm_dict, catboost_top6, heatmap_counts, weights=[1.2, 1.0, 0.6], min_conf=0.3):
    score = defaultdict(float)
    for ws, (digits, confs) in lstm_dict.items():
        if not digits or confs is None or len(confs) == 0 or max(confs) < min_conf: continue
        norm_confs = softmax(confs)
        for d, c in zip(digits, norm_confs):
            score[d] += weights[0] * c
    for d in catboost_top6: score[d] += weights[1]
    for d, c in heatmap_counts.items(): score[d] += weights[2] * c
    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:6]]

def hybrid_vote(conf, prob, alpha=0.5):
    vote = defaultdict(float)
    for i, d in enumerate(conf): vote[d] += alpha * (6 - i)
    for i, d in enumerate(prob): vote[d] += (1 - alpha) * (6 - i)
    ranked = sorted(vote.items(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:6]]

def stacked_vote(hybrid, direct, a1=0.6, a2=0.4):
    total = a1 + a2 if a1 + a2 > 0 else 1
    w1, w2 = a1 / total, a2 / total
    vote = defaultdict(float)
    for i, d in enumerate(hybrid): vote[d] += w1 * (6 - i)
    for i, d in enumerate(direct): vote[d] += w2 * (6 - i)
    ranked = sorted(vote.items(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:6]]

def final_ensemble(stacked, markov, weight_markov=0.3):
    score = defaultdict(float)
    for i, d in enumerate(stacked): score[d] += 6 - i
    for i, d in enumerate(markov): score[d] += weight_markov * (6 - i)
    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:6]]

def dynamic_alpha(a_conf, a_prob): return a_conf / (a_conf + a_prob) if (a_conf + a_prob) else 0.5

def tab5(df, lokasi):
    st.header("🧠 Tab 5 - Maximal Adaptive Prediction")

    folds = st.slider("CV Fold", 2, 10, 3)
    seed = st.number_input("Seed", 0, 9999, 42)
    min_ws, max_ws = 4, 25
    min_conf = 0.3
    weight_markov = 0.3

    if "tab5_result" not in st.session_state:
        st.session_state.tab5_result = {}

    if st.button("🔍 Jalankan Prediksi Maksimal", use_container_width=True):
        st.session_state.tab5_result = {}
        anomaly = detect_anomaly(df)
        if anomaly: st.warning("⚠️ Anomali terdeteksi. Strategi adaptif diaktifkan.")

        for label in DIGIT_LABELS:
            try:
                st.markdown(f"#### 🔢 Posisi {label.upper()}")
                result_df = scan_ws_catboost(df, label, min_ws, max_ws, folds, seed)
                result_df["Stabilitas"] = result_df["Accuracy Mean"] - result_df["Accuracy Std"]
                top3 = result_df.sort_values("Stabilitas", ascending=False).head(3)
                best_ws = int(top3.iloc[0]["WS"])
                acc_conf = top3.iloc[0]["Accuracy Mean"]
                acc_prob = top3["Accuracy Mean"].mean()

                # LSTM + Probs
                lstm_dict = {}
                all_probs = []
                for _, row in top3.iterrows():
                    ws = int(row["WS"])
                    try:
                        model = train_temp_lstm_model(df, label, ws, seed)
                        top6, probs = get_top6_lstm_temp(model, df, ws)
                        lstm_dict[ws] = (top6, probs)
                        all_probs.append(probs)
                    except:
                        lstm_dict[ws] = ([], [])

                catboost_top6 = [d for pair in lstm_dict.values() for d in pair[0]]
                heatmap_counts = Counter()
                for top in result_df["Top6"]:
                    for d in str(top).split(","):
                        if d.strip().isdigit():
                            heatmap_counts[int(d.strip())] += 1

                conf = ensemble_confidence(lstm_dict, catboost_top6, heatmap_counts, min_conf=min_conf)
                prob = ensemble_probabilistic(all_probs, [acc_conf]*len(all_probs)) if all_probs else []

                alpha = dynamic_alpha(acc_conf, acc_prob)
                if anomaly: alpha *= 0.6
                hybrid = hybrid_vote(conf, prob, alpha)

                try:
                    model = train_temp_lstm_model(df, label, best_ws, seed)
                    direct, probs_direct = get_top6_lstm_temp(model, df, best_ws)
                    if probs_direct is not None and np.max(probs_direct) < 0.3: direct = []
                except:
                    direct = []

                stacked = stacked_vote(hybrid, direct, acc_conf, acc_conf)
                markov = top6_markov_hybrid(df)[DIGIT_LABELS.index(label)]
                final = final_ensemble(stacked, markov, weight_markov=0.15 if anomaly else 0.3)

                real_digit = int(str(df.iloc[-1]["angka"])[DIGIT_LABELS.index(label)])
                st.session_state.tab5_result[label] = {"final": final, "real": real_digit}

                st.success(f"Final (Top-6): {final}")
                st.write(f"Target Real: `{real_digit}` | {'✅' if real_digit in final else '❌'}")

            except Exception as e:
                st.error(f"Gagal memproses {label.upper()}: {e}")

    if st.session_state.tab5_result:
        st.markdown("### 📊 Hasil Ringkasan")
        summary = []
        for label in DIGIT_LABELS:
            hasil = st.session_state.tab5_result.get(label)
            if hasil:
                summary.append({
                    "Posisi": label,
                    "Top-6": ", ".join(map(str, hasil["final"])),
                    "Target": hasil["real"],
                    "Match": "✅" if hasil["real"] in hasil["final"] else "❌"
                })
        st.table(pd.DataFrame(summary))
