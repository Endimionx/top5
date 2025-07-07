import streamlit as st
import pandas as pd
import requests
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from markov_model import top6_markov, top6_markov_order2, top6_markov_hybrid
from ai_model import (
    top6_lstm,
    train_and_save_lstm,
    kombinasi_4d,
    top6_ensemble,
    model_exists
)
from lokasi_list import lokasi_list

st.set_page_config(page_title="Prediksi Togel AI", layout="wide")

def cari_putaran_terbaik(lokasi, max_putaran=200):
    best_score = 0
    best_putaran = 30
    start = time.time()
    for p in range(30, max_putaran + 1, 10):
        try:
            url = f"https://wysiwygscan.com/api?pasaran={lokasi.lower()}&hari=harian&putaran={p}&format=json&urut=asc"
            headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
            response = requests.get(url, headers=headers)
            data = response.json()
            angka_list = [item["result"] for item in data.get("data", []) if len(item["result"]) == 4 and item["result"].isdigit()]
            df = pd.DataFrame({"angka": angka_list})
            pred, _ = top6_markov(df)
            if pred:
                total, benar = 0, 0
                for angka in df["angka"][-10:]:
                    actual = f"{int(angka):04d}"
                    for i in range(4):
                        if int(actual[i]) in pred[i]:
                            benar += 1
                        total += 1
                acc = benar / total if total else 0
                if acc > best_score:
                    best_score = acc
                    best_putaran = p
        except:
            continue
        if time.time() - start > 5:
            break
    return best_putaran

# Sidebar
st.sidebar.title("⚙️ Pengaturan")
selected_lokasi = st.sidebar.selectbox("🌍 Pilih Pasaran", lokasi_list)
selected_hari = st.sidebar.selectbox("📅 Pilih Hari", ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"])
auto_cari = st.sidebar.checkbox("🔍 Cari Putaran Terbaik (rekomendasi)", value=False)
if auto_cari:
    putaran = None
else:
    putaran = st.sidebar.slider("🔁 Jumlah Putaran (Manual)", 30, 1000, 100)
jumlah_uji = st.sidebar.number_input("📊 Data Uji Akurasi", 5, 200, 10)
metode = st.sidebar.selectbox("🧠 Metode Prediksi", ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI", "Ensemble AI + Markov"])

min_conf = 0.0005
power = 1.5
if metode in ["LSTM AI", "Ensemble AI + Markov"]:
    min_conf = st.sidebar.slider("🔎 Minimum Confidence", 0.0001, 0.001, 0.0005, step=0.0001, format="%.4f")
    power = st.sidebar.slider("📈 Confidence Weight Power", 0.5, 3.0, 1.5, step=0.1)

angka_list = []
riwayat_input = ""
if selected_lokasi and selected_hari:
    try:
        with st.spinner("🔄 Mengambil data..."):
            if auto_cari:
                st.info("🔍 Mencari jumlah putaran terbaik...")
                putaran = cari_putaran_terbaik(selected_lokasi, max_putaran=200)
                st.success(f"✅ Ditemukan putaran terbaik: {putaran}")
            url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran={putaran}&format=json&urut=asc"
            headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
            response = requests.get(url, headers=headers)
            data = response.json()
            angka_list = [item["result"] for item in data.get("data", []) if len(item["result"]) == 4 and item["result"].isdigit()]
            riwayat_input = "\n".join(angka_list)
            st.success(f"✅ {len(angka_list)} angka berhasil diambil (Putaran: {putaran})")
            with st.expander("📥 Lihat Data"):
                st.code(riwayat_input, language="text")
    except Exception as e:
        st.error(f"❌ Gagal ambil data: {e}")

df = pd.DataFrame({"angka": angka_list})

# Manajemen Model
if metode == "LSTM AI":
    with st.expander("⚙️ Manajemen Model LSTM"):
        for i in range(4):
            model_path = f"saved_models/{selected_lokasi.lower().replace(' ', '_')}_digit{i}.h5"
            col1, col2 = st.columns([2, 1])
            with col1:
                if os.path.exists(model_path):
                    st.info(f"📂 Model Digit-{i} tersedia.")
                else:
                    st.warning(f"⚠️ Model Digit-{i} belum tersedia.")
            with col2:
                if os.path.exists(model_path):
                    if st.button(f"🗑 Hapus Digit-{i}", key=f"hapus_digit_{i}"):
                        os.remove(model_path)
                        st.warning(f"✅ Model Digit-{i} dihapus.")

        if st.button("📚 Latih & Simpan Semua Model"):
            with st.spinner("🔄 Melatih semua model per digit..."):
                train_and_save_lstm(df, selected_lokasi)
            st.success("✅ Semua model berhasil dilatih dan disimpan.")

# Tombol Prediksi
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

        if result is None:
            st.error("❌ Gagal melakukan prediksi.")
        else:
            with st.expander("🎯 Hasil Prediksi Top 6 Digit"):
                col1, col2 = st.columns(2)
                for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                    with (col1 if i % 2 == 0 else col2):
                        st.markdown(f"**{label}:** {', '.join(map(str, result[i]))}")

            if metode in ["LSTM AI", "Ensemble AI + Markov"]:
                with st.spinner("🔢 Menghitung kombinasi 4D terbaik..."):
                    top_komb = kombinasi_4d(df, lokasi=selected_lokasi, top_n=10, min_conf=min_conf, power=power)
                    if top_komb:
                        with st.expander("💡 Kombinasi 4D Confidence Tinggi"):
                            sim_col = st.columns(2)
                            for i, (komb, score) in enumerate(top_komb):
                                with sim_col[i % 2]:
                                    st.markdown(f"`{komb}` - ⚡️ Confidence: `{score:.4f}`")

        # Evaluasi Akurasi
        with st.spinner("📏 Menghitung akurasi..."):
            uji_df = df.tail(min(jumlah_uji, len(df)))
            total, benar = 0, 0
            akurasi_list = []
            digit_acc = {"Ribuan": [], "Ratusan": [], "Puluhan": [], "Satuan": []}

            for i in range(len(uji_df)):
                subset_df = df.iloc[:-(len(uji_df) - i)]
                if len(subset_df) < 30:
                    continue
                try:
                    pred = (
                        top6_markov(subset_df)[0] if metode == "Markov" else
                        top6_markov_order2(subset_df) if metode == "Markov Order-2" else
                        top6_markov_hybrid(subset_df) if metode == "Markov Gabungan" else
                        top6_lstm(subset_df, lokasi=selected_lokasi) if metode == "LSTM AI" else
                        top6_ensemble(subset_df, lokasi=selected_lokasi)
                    )
                    if pred is None:
                        continue

                    actual = f"{int(uji_df.iloc[i]['angka']):04d}"
                    skor = 0
                    for j, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                        if int(actual[j]) in pred[j]:
                            skor += 1
                            digit_acc[label].append(1)
                        else:
                            digit_acc[label].append(0)
                    total += 4
                    benar += skor
                    akurasi_list.append(skor / 4 * 100)
                except:
                    continue

            if total > 0:
                st.success(f"📈 Akurasi {metode}: {benar / total * 100:.2f}%")
                with st.expander("📊 Grafik Akurasi"):
                    st.line_chart(pd.DataFrame({"Akurasi (%)": akurasi_list}))
                with st.expander("🔥 Heatmap Akurasi per Digit"):
                    heat_df = pd.DataFrame({k: [sum(v)/len(v)*100 if v else 0] for k, v in digit_acc.items()})
                    fig, ax = plt.subplots()
                    sns.heatmap(heat_df, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
                    st.pyplot(fig)
            else:
                st.warning("⚠️ Tidak cukup data untuk evaluasi akurasi.")
