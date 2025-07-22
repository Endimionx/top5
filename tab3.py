```python
# tab3.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from itertools import product

from ws_scan_catboost import (
    scan_ws_catboost,
    train_temp_lstm_model,
    get_top6_lstm_temp,
    show_catboost_heatmaps,
    DIGIT_LABELS,
    ensemble_top6
)

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
                result_df = scan_ws_catboost(
                    df, label,
                    min_ws=min_ws_cb3,
                    max_ws=max_ws_cb3,
                    cv_folds=folds_cb3,
                    seed=temp_seed
                )

                result_df["ConfCount"] = result_df["Top6"].apply(
                    lambda x: sum([int(d) in range(10) for d in str(x).split(",") if d.strip().isdigit()])
                )
                result_df["ComboScore"] = 0.7 * result_df["Accuracy Mean"] + 0.3 * (result_df["ConfCount"] / 6)

                top3_combo = result_df.sort_values("ComboScore", ascending=False).head(3)
                st.session_state.tab3_top6_acc[label] = {}
                for _, row in top3_combo.iterrows():
                    ws = int(row["WS"])
                    try:
                        model = train_temp_lstm_model(df, label, window_size=ws, seed=temp_seed)
                        top6, _ = get_top6_lstm_temp(model, df, window_size=ws)
                        st.session_state.tab3_top6_acc[label][ws] = top6
                    except:
                        st.session_state.tab3_top6_acc[label][ws] = []

                st.session_state.tab3_top6_conf[label] = {}
                for _, row in top3_combo.iterrows():
                    ws = int(row["WS"])
                    st.session_state.tab3_top6_conf[label][ws] = row["Top6"]

                best_row = result_df.loc[result_df["ComboScore"].idxmax()]
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

                # Final ensemble
                top6_lstm = list(st.session_state.tab3_top6_acc[label].values())[0] if st.session_state.tab3_top6_acc[label] else []
                top6_catboost = [int(d) for d in str(best_row["Top6"]).split(",") if d.strip().isdigit()]
                top6_heatmap = []  # Placeholder jika ingin gunakan summary heatmap

                ensemble = ensemble_top6(
                    top6_lstm,
                    top6_catboost,
                    top6_heatmap,
                    weights=[1.2, 1.0, 0.8]
                )
                st.markdown(f"### 🧠 Final Ensemble Top6 - {label.upper()}")
                st.write(ensemble)
                st.session_state[f"final_ensemble_{label}"] = ensemble

                try:
                    model = train_temp_lstm_model(df, label, window_size=best_ws, seed=temp_seed)
                    top6, probs = get_top6_lstm_temp(model, df, window_size=best_ws)
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

    # === Rekap Final ===
    if st.session_state.tab3_top6_acc and st.session_state.tab3_top6_conf:
        st.markdown("---")
        st.subheader("📦 Rekap Top-6 dari Top-3 WS")

        all_ensemble = []

        for label in DIGIT_LABELS:
            st.markdown(f"### 🔢 {label.upper()}")

            top_acc = st.session_state.tab3_top6_acc.get(label, {})
            st.markdown("**🎯 Top-6 dari Top-3 WS by Combo Score**")
            df_acc = pd.DataFrame(dict([(f"WS {ws}", top6) for ws, top6 in top_acc.items()]))
            st.dataframe(df_acc)

            top_conf = st.session_state.tab3_top6_conf.get(label, {})
            st.markdown("**🔥 Top-6 dari CatBoost**")
            df_conf = pd.DataFrame(dict([(f"WS {ws}", str(top6).split(",")) for ws, top6 in top_conf.items()]))
            st.dataframe(df_conf)

            combined = []
            for lst in list(top_acc.values()) + [
                [int(d) for d in str(s).split(",") if d.strip().isdigit()] for s in top_conf.values()
            ]:
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
```
