import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter, defaultdict
from itertools import product
from ensemble_probabilistic import ensemble_probabilistic

from ws_scan_catboost import (
    scan_ws_catboost,
    train_temp_lstm_model,
    get_top6_lstm_temp,
    show_catboost_heatmaps,
    DIGIT_LABELS,
)

def ensemble_confidence_voting(
    lstm_dict, catboost_top6, heatmap_counts,
    weights=[1.2, 1.0, 0.6],
    min_lstm_conf=0.3
):
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

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

def hybrid_voting(conf_votes, prob_votes, alpha=0.6):
    score = defaultdict(float)
    for rank, digit in enumerate(conf_votes):
        score[digit] += alpha * (6 - rank)
    for rank, digit in enumerate(prob_votes):
        score[digit] += (1 - alpha) * (6 - rank)
    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:6]]

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
    hybrid_alpha = st.slider("Hybrid Alpha (Conf vs Prob)", 0.0, 1.0, 0.6, 0.05, key="tab3_hybrid_alpha")

    for key in ["tab3_full_results", "tab3_top6_acc", "tab3_top6_conf", "tab3_ensemble", "tab3_ensemble_prob", "tab3_ensemble_hybrid"]:
        if key not in st.session_state:
            st.session_state[key] = {}

    st.markdown("Opsi Scan Per Digit")
    selected_digit_tab3 = st.selectbox("Pilih digit yang ingin discan", ["(Semua)"] + DIGIT_LABELS, key="tab3_selected_digit")

    if st.button("🔎 Scan Per Digit", use_container_width=True):
        for key in ["tab3_full_results", "tab3_top6_acc", "tab3_top6_conf", "tab3_ensemble", "tab3_ensemble_prob", "tab3_ensemble_hybrid"]:
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
                catboost_top6_all = []
                for ws_data in st.session_state.tab3_top6_conf[label].values():
                    catboost_top6_all.extend(ws_data[0])

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

                hybrid_result = hybrid_voting(final_ens_conf, final_ens_prob, alpha=hybrid_alpha)
                st.session_state.tab3_ensemble_hybrid[label] = hybrid_result

                st.markdown(f"### 🧠 Final Ensemble Top6 - {label.upper()}")
                st.write(f"Confidence Voting: `{final_ens_conf}`")
                st.write(f"Probabilistic Voting: `{final_ens_prob}`")
                st.success(f"Hybrid Voting: `{hybrid_result}`")

                fig_acc = plt.figure(figsize=(6, 2))
                plt.bar(result_df["WS"], result_df["Accuracy Mean"], color="skyblue")
                plt.title(f"Akurasi vs WS - {label.upper()}")
                plt.xlabel("WS")
                plt.ylabel("Akurasi")
                st.pyplot(fig_acc)

                show_catboost_heatmaps(result_df, label)

                try:
                    model = train_temp_lstm_model(df, label, best_ws, temp_seed)
                    top6, probs = get_top6_lstm_temp(model, df, best_ws)
                    st.markdown("**🎯 Prediksi Langsung (Top-6):**")
                    st.info(f"Top-6: {top6}")
                    df_conf = pd.DataFrame({"Digit": [str(d) for d in top6], "Confidence": probs})
                    fig_bar, ax = plt.subplots(figsize=(6, 2))
                    sns.barplot(x="Digit", y="Confidence", data=df_conf, palette="viridis", ax=ax)
                    ax.set_title(f"Confidence Bar - {label.upper()}")
                    st.pyplot(fig_bar)
                except Exception as e:
                    st.warning(f"⚠️ Gagal prediksi langsung: {e}")

            except Exception as e:
                st.error(f"❌ Gagal proses {label.upper()}: {e}")
