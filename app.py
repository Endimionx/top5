import streamlit as st
import pandas as pd
import requests
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from markov_model import (
    top6_markov,
    top6_markov_order2,
    top6_markov_hybrid,
    kombinasi_4d_markov_hybrid
)
from ai_model import (
    top6_lstm,
    kombinasi_4d,
    top6_ensemble,
    load_training_history
)
from lokasi_list import lokasi_list
from user_manual import tampilkan_user_manual

st.set_page_config(page_title="Prediksi Togel AI", layout="wide")
tampilkan_user_manual()

# 🔧 Sidebar
hari_list = ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"]
metode_list = ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI", "Ensemble AI + Markov"]

with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    selected_hari = st.selectbox("📅 Pilih Hari", hari_list)
    jumlah_uji = st.number_input("📊 Data Uji Akurasi", min_value=3, max_value=100, value=7)
    metode = st.selectbox("🧠 Metode Prediksi", metode_list)
    cari_otomatis = st.toggle("🔍 Cari Putaran Otomatis", value=False)

    if cari_otomatis:
        max_putaran = st.number_input("🧮 Max Putaran Dicoba", min_value=50, max_value=1000, value=200)

    digit_weight_input = [1.0, 1.0, 1.0, 1.0]
    if metode == "Markov Gabungan":
        st.markdown("🎯 **Bobot Confidence Tiap Digit**")
        digit_weight_input = [
            st.slider("📌 Ribuan", 0.1, 3.0, 1.0, 0.1),
            st.slider("📌 Ratusan", 0.1, 3.0, 1.0, 0.1),
            st.slider("📌 Puluhan", 0.1, 3.0, 1.0, 0.1),
            st.slider("📌 Satuan", 0.1, 3.0, 1.0, 0.1),
        ]

# 🔄 Load Data
putaran = 100
df_all = pd.DataFrame()
if selected_lokasi and selected_hari:
    with st.spinner("📥 Mengambil data..."):
        try:
            url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran=1000&format=json&urut=asc"
            headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
            r = requests.get(url, headers=headers)
            angka_all = [x["result"] for x in r.json().get("data", []) if len(x["result"]) == 4]
            df_all = pd.DataFrame({"angka": angka_all})
        except Exception as e:
            st.error(f"❌ Error: {e}")

    if cari_otomatis and not df_all.empty:
        def cari_putaran_terbaik(df_all, lokasi, metode, jumlah_uji=10, max_putaran=200, digit_weights=None):
            best_score, best_n = 0, 0
            for n in range(30, min(len(df_all), max_putaran)):
                subset = df_all.tail(n).reset_index(drop=True)
                acc_total, acc_benar = 0, 0
                for i in range(min(jumlah_uji, len(subset) - 30)):
                    train_df = subset.iloc[:-(jumlah_uji - i)]
                    if len(train_df) < 30:
                        continue
                    try:
                        pred = (
                            top6_markov(train_df)[0] if metode == "Markov" else
                            top6_markov_order2(train_df) if metode == "Markov Order-2" else
                            top6_markov_hybrid(train_df, digit_weights=digit_weights) if metode == "Markov Gabungan" else
                            top6_lstm(train_df, lokasi=lokasi) if metode == "LSTM AI" else
                            top6_ensemble(train_df, lokasi=lokasi)
                        )
                        actual = f"{int(subset.iloc[-(jumlah_uji - i)]['angka']):04d}"
                        acc = sum(int(actual[j]) in pred[j] for j in range(4))
                        acc_benar += acc
                        acc_total += 4
                    except:
                        continue
                akurasi = acc_benar / acc_total * 100 if acc_total else 0
                if akurasi > best_score:
                    best_score = akurasi
                    best_n = n
            return best_n, best_score

        with st.spinner("🔍 Mencari putaran terbaik..."):
            best_n, best_score = cari_putaran_terbaik(df_all, selected_lokasi, metode, jumlah_uji, max_putaran,
                                                       digit_weights=digit_weight_input if metode == "Markov Gabungan" else None)
        if best_n:
            putaran = best_n
            st.success(f"✅ Putaran terbaik: {best_n} (Akurasi: {best_score:.2f}%)")

    elif not cari_otomatis:
        putaran = st.number_input("🔁 Jumlah Putaran", min_value=30, max_value=1000, value=100)

# 📦 Ambil Data Terbaru
df = df_all.tail(putaran).reset_index(drop=True) if not df_all.empty else pd.DataFrame()

# 🔮 Prediksi
if st.button("🔮 Prediksi"):
    if len(df) < 30:
        st.warning("❗ Minimal 30 data diperlukan")
    else:
        with st.spinner("🔮 Memproses prediksi..."):
            result = None
            if metode == "Markov":
                result, _ = top6_markov(df)
            elif metode == "Markov Order-2":
                result = top6_markov_order2(df)
            elif metode == "Markov Gabungan":
                result = top6_markov_hybrid(df, digit_weights=digit_weight_input)
            elif metode == "LSTM AI":
                result = top6_lstm(df, lokasi=selected_lokasi)
            elif metode == "Ensemble AI + Markov":
                result = top6_ensemble(df, lokasi=selected_lokasi)

        if result:
            with st.expander("🎯 Hasil Prediksi Top 6 Digit"):
                col1, col2 = st.columns(2)
                for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                    with (col1 if i % 2 == 0 else col2):
                        st.markdown(f"**{label}:** {', '.join(map(str, result[i]))}")

            # 💡 Kombinasi 4D
            with st.expander("💡 Simulasi Kombinasi 4D"):
                top_komb = kombinasi_4d(result, mode="average") if metode == "LSTM AI" else kombinasi_4d_markov_hybrid(
                    df,
                    top_n=10,
                    mode="average",
                    digit_weights={
                        "ribuan": digit_weight_input[0],
                        "ratusan": digit_weight_input[1],
                        "puluhan": digit_weight_input[2],
                        "satuan": digit_weight_input[3],
                    }
                )
                for komb, score in top_komb:
                    st.markdown(f"**{komb}** - ⚡ Confidence: `{score:.4f}`")

# 📈 Grafik Training
if metode == "LSTM AI":
    with st.expander("📈 Grafik Akurasi Training per Digit"):
        for i, digit in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
            log_path = f"training_logs/history_{selected_lokasi.lower().replace(' ', '_')}_digit{i}.csv"
            if os.path.exists(log_path):
                df_log = load_training_history(log_path)
                fig, ax = plt.subplots()
                sns.lineplot(data=df_log, x=df_log.index, y="accuracy", ax=ax, label="Akurasi")
                sns.lineplot(data=df_log, x=df_log.index, y="val_accuracy", ax=ax, label="Val Akurasi")
                ax.set_title(f"Akurasi Digit {digit}")
                st.pyplot(fig)

# 🔥 Heatmap Akurasi
if metode in ["LSTM AI", "Markov Gabungan"] and not df.empty:
    with st.expander("🌡️ Heatmap Akurasi per Digit"):
        digit_array = np.array([[int(ch) for ch in row] for row in df["angka"]])
        df_digit = pd.DataFrame(digit_array, columns=["R", "A", "B", "C"])
        fig, ax = plt.subplots()
        sns.heatmap(df_digit.apply(pd.Series.value_counts).fillna(0).astype(int), annot=True, fmt="d", cmap="YlGnBu", ax=ax)
        ax.set_title("Distribusi Digit per Posisi")
        st.pyplot(fig)
