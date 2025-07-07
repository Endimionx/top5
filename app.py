import streamlit as st
import pandas as pd
import requests
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from markov_model import (
    top6_markov,
    top6_markov_order2,
    top6_markov_hybrid,
    kombinasi_4d_markov_hybrid
)
from ai_model import (
    top6_lstm,
    train_and_save_lstm,
    kombinasi_4d,
    model_exists,
    top6_ensemble
)
from lokasi_list import lokasi_list
from user_manual import tampilkan_user_manual

def cari_putaran_terbaik(df_all, lokasi, metode, jumlah_uji=10, max_putaran=200):
    best_score, best_n, hasil_all = 0, 0, {}
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
                    top6_markov_hybrid(train_df) if metode == "Markov Gabungan" else
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
        hasil_all[n] = akurasi
        if akurasi > best_score:
            best_score = akurasi
            best_n = n
    return best_n, best_score, hasil_all

st.set_page_config(page_title="Prediksi Togel AI", layout="wide")
tampilkan_user_manual()

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
        max_putaran = st.number_input("🧮 Max Putaran untuk Dicoba", min_value=50, max_value=1000, value=200)

    putaran = 100
    best_score = None
    df_all = pd.DataFrame()

    if selected_lokasi and selected_hari:
        try:
            with st.spinner("📥 Mengambil semua data..."):
                url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran=1000&format=json&urut=asc"
                headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
                response = requests.get(url, headers=headers)
                data = response.json()
                angka_list_all = [item["result"] for item in data.get("data", []) if len(item["result"]) == 4 and item["result"].isdigit()]
                df_all = pd.DataFrame({"angka": angka_list_all})
        except Exception as e:
            st.error(f"❌ Gagal ambil data awal: {e}")

    if cari_otomatis and not df_all.empty:
        with st.spinner("🔍 Menganalisis putaran terbaik..."):
            best_n, best_score, _ = cari_putaran_terbaik(df_all, selected_lokasi, metode, jumlah_uji, max_putaran)
        if best_n > 0:
            putaran = best_n
            st.success(f"✅ Putaran terbaik: {best_n} (Akurasi: {best_score:.2f}%)")
        else:
            st.warning("⚠️ Gagal menemukan putaran terbaik.")
    elif not cari_otomatis:
        putaran = st.number_input("🔁 Jumlah Putaran", min_value=20, max_value=1000, value=100, step=1)

    min_conf = 0.0005
    power = 1.5
    digit_weight_input = [1.0, 1.0, 1.0, 1.0]

    if metode in ["LSTM AI", "Ensemble AI + Markov"]:
        min_conf = st.slider("🔎 Minimum Confidence", 0.0001, 0.001, 0.0005, step=0.0001, format="%.4f")
        power = st.slider("📈 Confidence Weight Power", 0.5, 3.0, 1.5, step=0.1)

if metode == "Markov Gabungan":
    st.markdown("🎯 **Bobot Confidence Tiap Digit (Markov Gabungan)**")
    digit_weight_input = [
        st.slider("📌 Ribuan", 0.1, 5.0, 1.0, 0.1),
        st.slider("📌 Ratusan", 0.1, 5.0, 1.0, 0.1),
        st.slider("📌 Puluhan", 0.1, 5.0, 1.0, 0.1),
        st.slider("📌 Satuan", 0.1, 5.0, 1.0, 0.1)
    ]

angka_list, riwayat_input = [], ""
df = pd.DataFrame()
if cari_otomatis and not df_all.empty:
    df = df_all.tail(putaran).reset_index(drop=True)
    angka_list = df["angka"].tolist()
    riwayat_input = "\n".join(angka_list)
    st.success(f"✅ Menggunakan {putaran} data dari hasil analisis otomatis.")
    with st.expander("📥 Lihat Data"):
        st.code(riwayat_input, language="text")
elif selected_lokasi and selected_hari:
    try:
        with st.spinner("📦 Mengambil data berdasarkan putaran..."):
            url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran={putaran}&format=json&urut=asc"
            headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
            response = requests.get(url, headers=headers)
            data = response.json()
            angka_list = [item["result"] for item in data.get("data", []) if len(item["result"]) == 4 and item["result"].isdigit()]
            df = pd.DataFrame({"angka": angka_list})
            riwayat_input = "\n".join(angka_list)
            st.success(f"✅ {len(angka_list)} angka berhasil diambil.")
            with st.expander("📥 Lihat Data"):
                st.code(riwayat_input, language="text")
    except Exception as e:
        st.error(f"❌ Gagal ambil data API: {e}")

if st.button("🔮 Prediksi"):
    if len(df) < 30:
        st.warning("❌ Minimal 30 data diperlukan.")
    else:
        with st.spinner("⏳ Melakukan prediksi..."):
            result = None
            if metode == "Markov":
                result, _ = top6_markov(df)
            elif metode == "Markov Order-2":
                result = top6_markov_order2(df)
            elif metode == "Markov Gabungan":
                result = top6_markov_hybrid(df)
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

            if metode == "Markov Gabungan":
                with st.spinner("🔢 Menghitung kombinasi 4D terbaik..."):
                    top_komb = kombinasi_4d_markov_hybrid(
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
                    if top_komb:
                        with st.expander("💡 Simulasi Kombinasi 4D (Markov Hybrid)"):
                            kode_output = "\n".join(
                                [f"{komb} - ⚡ Confidence: {score:.6f}" for komb, score in top_komb]
                            )
                            st.code(kode_output, language="text")

            # Heatmap Akurasi per Digit
            with st.expander("🔥 Heatmap Akurasi Persentase per Digit"):
                sim_count = min(100, len(df) - 30)
                acc_matrix = np.zeros((4, sim_count))
                for i in range(sim_count):
                    train_df = df.iloc[:-(sim_count - i)]
                    test = df.iloc[-(sim_count - i)]
                    try:
                        pred = (
                            top6_markov(train_df)[0] if metode == "Markov" else
                            top6_markov_order2(train_df) if metode == "Markov Order-2" else
                            top6_markov_hybrid(train_df) if metode == "Markov Gabungan" else
                            top6_lstm(train_df, lokasi=selected_lokasi) if metode == "LSTM AI" else
                            top6_ensemble(train_df, lokasi=selected_lokasi)
                        )
                        actual = f"{int(test['angka']):04d}"
                        for j in range(4):
                            acc_matrix[j][i] = 1 if int(actual[j]) in pred[j] else 0
                    except:
                        continue

                digit_accuracy = acc_matrix.sum(axis=1) / sim_count * 100
                df_heat = pd.DataFrame(digit_accuracy.reshape(-1, 1),
                                       index=["Ribuan", "Ratusan", "Puluhan", "Satuan"],
                                       columns=["Akurasi (%)"])
                fig, ax = plt.subplots(figsize=(4, 2))
                sns.heatmap(df_heat, annot=True, cmap="YlGnBu", fmt=".2f", cbar=False, ax=ax)
                ax.set_title("Akurasi Prediksi per Digit (%)")
                st.pyplot(fig)

            # Grafik Akurasi Persentase Keberhasilan
            with st.expander("📊 Grafik Akurasi % terhadap Jumlah Uji"):
                acc_total, acc_benar = 0, 0
                acc_list = []
                sim_range = min(jumlah_uji, len(df) - 30)
                for i in range(sim_range):
                    train_df = df.iloc[:-(sim_range - i)]
                    test = df.iloc[-(sim_range - i)]
                    try:
                        pred = (
                            top6_markov(train_df)[0] if metode == "Markov" else
                            top6_markov_order2(train_df) if metode == "Markov Order-2" else
                            top6_markov_hybrid(train_df) if metode == "Markov Gabungan" else
                            top6_lstm(train_df, lokasi=selected_lokasi) if metode == "LSTM AI" else
                            top6_ensemble(train_df, lokasi=selected_lokasi)
                        )
                        actual = f"{int(test['angka']):04d}"
                        acc = sum(int(actual[j]) in pred[j] for j in range(4))
                        akurasi_persen = acc / 4 * 100
                        acc_list.append(akurasi_persen)
                    except:
                        continue
                df_plot = pd.DataFrame({"Uji ke-": list(range(1, len(acc_list)+1)), "Akurasi (%)": acc_list})
                st.line_chart(df_plot.set_index("Uji ke-"))
