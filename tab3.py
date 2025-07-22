# tab3.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

def ensemble_confidence_voting(lstm_dict, catboost_top6, heatmap_counts, weights=[1.2, 1.0, 0.6]):
    score = defaultdict(float)

    for ws, (digits, confs) in lstm_dict.items():
        if not digits or not confs.any(): continue
        max_conf = max(confs)
        norm_confs = [c / max_conf if max_conf > 0 else 0 for c in confs]
        for d, c in zip(digits, norm_confs):
            score[d] += weights[0] * c

    for d in catboost_top6:
        score[d] += weights[1]

    for d, count in heatmap_counts.items():
        score[d] += weights[2] * count

    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:6]]

def tab3(df):
    min_ws_cb3 = st.number_input("🔁 Min WS", 3, 20, 5, key="tab3_min_ws")
    max_ws_cb3 = st.number_input("🔁 Max WS", min_ws_cb3 + 1, 30, min_ws_cb3 + 6, key="tab3_max_ws")
    folds_cb3 = st.slider("📂 Jumlah Fold", 2, 10, 3, key="tab3_cv")
    temp_seed = st.number_input("🎲 Seed", 0, 9999, 42, key="tab3_seed")

    if "tab3_full_results" not in st.session_state:
        st.session_state.tab3_full_results = {}
    if "tab3_top6_acc" not in st.session_state:
        st.session_state.tab3_top6_acc = {}
    if "tab3_top6_conf" not in st.session_state:
        st.session_state.tab3_top6_conf = {}
    if "tab3_ensemble" not in st.session_state:
        st.session_state.tab3_ensemble = {}

    st.markdown("### 📌 Opsi Scan Per Digit (Opsional)")
    selected_digit_tab3 = st.selectbox("Pilih digit yang ingin discan", ["(Semua)"] + DIGIT_LABELS, key="tab3_selected_digit")

    if st.button("🔎 Scan Per Digit", use_container_width=True):
        st.session_state.tab3_full_results = {}
        st.session_state.tab3_top6_acc = {}
        st.session_state.tab3_top6_conf = {}
        st.session_state.tab3_ensemble = {}

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

                fig_acc = plt.figure(figsize=(6, 2))
                plt.bar(result_df["WS"], result_df["Accuracy Mean"], color="skyblue")
                plt.title(f"Akurasi vs WS - {label.upper()}")
                plt.xlabel("WS")
                plt.ylabel("Akurasi")
                st.pyplot(fig_acc)

                show_catboost_heatmaps(result_df, label)

                lstm_dict = st.session_state.tab3_top6_acc[label]
                catboost_top6_all = []
                for ws_data in st.session_state.tab3_top6_conf[label].values():
                    catboost_top6_all.extend(ws_data[0])
                top6_all = result_df["Top6"].apply(lambda x: [int(i) for i in str(x).split(",") if i.strip().isdigit()])
                heatmap_counts = Counter()
                for t in top6_all:
                    heatmap_counts.update(t)

                final_ens = ensemble_confidence_voting(lstm_dict, catboost_top6_all, heatmap_counts)
                st.session_state.tab3_ensemble[label] = final_ens
                st.markdown(f"### 🧠 Final Ensemble Top6 - {label.upper()}")
                st.write(final_ens)

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

    if st.button("🚀 Jalankan Prediksi Otomatis", use_container_width=True):
        st.session_state.tab3_full_results = {}
        st.session_state.tab3_top6_acc = {}
        st.session_state.tab3_top6_conf = {}
        st.session_state.tab3_ensemble = {}

        for label in DIGIT_LABELS:
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

                result_df["ConfCount"] = result_df["Top6"].apply(
                    lambda x: sum([int(d) in range(10) for d in str(x).split(",") if d.strip().isdigit()])
                )
                top3_conf = result_df.sort_values("ConfCount", ascending=False).head(3)
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

                fig_acc = plt.figure(figsize=(6, 2))
                plt.bar(result_df["WS"], result_df["Accuracy Mean"], color="skyblue")
                plt.title(f"Akurasi vs WS - {label.upper()}")
                plt.xlabel("WS")
                plt.ylabel("Akurasi")
                st.pyplot(fig_acc)

                show_catboost_heatmaps(result_df, label)

                lstm_dict = st.session_state.tab3_top6_acc[label]
                catboost_top6_all = []
                for ws_data in st.session_state.tab3_top6_conf[label].values():
                    catboost_top6_all.extend(ws_data[0])
                top6_all = result_df["Top6"].apply(lambda x: [int(i) for i in str(x).split(",") if i.strip().isdigit()])
                heatmap_counts = Counter()
                for t in top6_all:
                    heatmap_counts.update(t)

                final_ens = ensemble_confidence_voting(lstm_dict, catboost_top6_all, heatmap_counts)
                st.session_state.tab3_ensemble[label] = final_ens
                st.markdown(f"### 🧠 Final Ensemble Top6 - {label.upper()}")
                st.write(final_ens)

            except Exception as e:
                st.error(f"❌ Gagal proses {label.upper()}: {e}")

    # === Rekap Final ===
    if st.session_state.tab3_top6_acc and st.session_state.tab3_top6_conf:
        st.markdown("---")
        st.subheader("📦 Rekap Top-6 dari Top-3 WS")

        all_ensemble = []

        for label in DIGIT_LABELS:
            st.markdown(f"### 🔢 {label.upper()}")

            top_acc = st.session_state.tab3_top6_acc.get(label, {})
            st.markdown("**🎯 Top-6 dari Top-3 WS by Accuracy**")
            df_acc = pd.DataFrame(dict([(f"WS {ws}", top6) for ws, (top6, _) in top_acc.items()]))
            st.dataframe(df_acc)

            top_conf = st.session_state.tab3_top6_conf.get(label, {})
            st.markdown("**🔥 Top-6 dari Top-3 WS by Confidence**")
            df_conf = pd.DataFrame(dict([(f"WS {ws}", top6) for ws, (top6, _) in top_conf.items()]))
            st.dataframe(df_conf)

            combined = []
            for lst, _ in list(top_acc.values()) + list(top_conf.values()):
                combined.extend(lst)

            counter = Counter(combined)
            top_ensemble = [digit for digit, _ in counter.most_common(6)]
            st.markdown(f"**⭐ Ensemble Top-6**: `{top_ensemble}`")

            all_ensemble.append(top_ensemble)
            st.session_state.tab3_ensemble[label] = top_ensemble

        if len(all_ensemble) == 4:
            st.subheader("🔢 Kombinasi 4D dari Ensemble Top-6")
            kombinasi = list(product(*all_ensemble))
            st.write(f"Total kombinasi: `{len(kombinasi)}`")
            for i, komb in enumerate(kombinasi[:20], 1):
                st.markdown(f"{i}. `{''.join(map(str, komb))}`")
