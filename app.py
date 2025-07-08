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
    top6_ensemble,
    kombinasi_4d,
    preprocess_data,
    build_model,
    model_exists
)
from cari_putaran_terbaik import cari_putaran_terbaik
from lokasi_list import lokasi_list
from user_manual import tampilkan_user_manual
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping

st.set_page_config(page_title="Prediksi Togel AI", layout="wide")
tampilkan_user_manual()

def load_training_history(path):
    return pd.read_csv(path)

hari_list = ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"]
metode_list = ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI", "Ensemble AI + Markov"]

with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    selected_hari = st.selectbox("📅 Pilih Hari", hari_list)
    jumlah_uji = st.number_input("📊 Data Uji Akurasi", 3, 100, value=7)
    metode = st.selectbox("🧠 Metode Prediksi", metode_list)
    cari_otomatis = st.toggle("🔍 Cari Putaran Otomatis", value=False)

    if cari_otomatis:
        max_putaran = st.number_input("🧮 Max Putaran Dicoba", 50, 1000, value=200)

    digit_weight_input = [1.0, 1.0, 1.0, 1.0]
    if metode == "Markov Gabungan":
        st.markdown("🎯 **Bobot Confidence Tiap Digit (Markov Gabungan)**")
        digit_weight_input = [
            st.slider("📌 Ribuan", 0.1, 3.0, 1.0, 0.1),
            st.slider("📌 Ratusan", 0.1, 3.0, 1.0, 0.1),
            st.slider("📌 Puluhan", 0.1, 3.0, 1.0, 0.1),
            st.slider("📌 Satuan", 0.1, 3.0, 1.0, 0.1)
        ]

putaran = 100
df_all = pd.DataFrame()
angka_list = []

if selected_lokasi and selected_hari:
    try:
        with st.spinner("📥 Mengambil semua data..."):
            url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran=1000&format=json&urut=asc"
            headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
            response = requests.get(url, headers=headers)
            angka_list_all = [d["result"] for d in response.json().get("data", []) if len(d["result"]) == 4 and d["result"].isdigit()]
            df_all = pd.DataFrame({"angka": angka_list_all})
    except Exception as e:
        st.error(f"❌ Gagal ambil data awal: {e}")

    if cari_otomatis and not df_all.empty:
        with st.spinner("🔍 Menganalisis putaran terbaik..."):
            best_n, best_score, _ = cari_putaran_terbaik(
                df_all,
                lokasi=selected_lokasi,
                metode=metode,
                jumlah_uji=jumlah_uji,
                max_putaran=max_putaran,
                digit_weights=digit_weight_input if metode == "Markov Gabungan" else None
            )
        if best_n > 0:
            putaran = best_n
            st.success(f"✅ Putaran terbaik: {best_n} (Akurasi: {best_score:.2f}%)")
        else:
            st.warning("⚠️ Gagal menemukan putaran terbaik.")
    else:
        putaran = st.number_input("🔁 Jumlah Putaran", 20, 1000, value=100)

df = pd.DataFrame()
try:
    if not df_all.empty:
        df = df_all.tail(putaran).reset_index(drop=True)
        angka_list = df["angka"].tolist()
    else:
        url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran={putaran}&format=json&urut=asc"
        headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
        response = requests.get(url, headers=headers)
        angka_list = [d["result"] for d in response.json().get("data", []) if len(d["result"]) == 4 and d["result"].isdigit()]
        df = pd.DataFrame({"angka": angka_list})
except Exception as e:
    st.error(f"❌ Gagal ambil data: {e}")

# ✅ Tampilkan data angka
if angka_list:
    with st.expander("📥 Lihat Data Angka dari API"):
        st.code("\n".join(angka_list), language="text")

# 🧠 Manajemen Model
if metode == "LSTM AI" and not df.empty:
    with st.expander("🧠 Manajemen Model LSTM per Digit"):
        for i, digit in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
            model_path = f"saved_models/{selected_lokasi.lower().replace(' ', '_')}_digit{i}.h5"
            st.markdown(f"### 🔢 Digit {digit}")
            if os.path.exists(model_path):
                st.success("✅ Model tersedia")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"🔁 Latih Ulang {digit}"):
                        X, y_all = preprocess_data(df)
                        y = y_all[i]
                        model = build_model(input_len=X.shape[1])
                        model.fit(X, y, epochs=50, batch_size=16, verbose=0, validation_split=0.2)
                        model.save(model_path)
                        st.success(f"✅ Model {digit} dilatih ulang.")
                with col2:
                    if st.button(f"🗑️ Hapus Model {digit}"):
                        os.remove(model_path)
                        st.warning(f"🧹 Model {digit} dihapus.")
            else:
                st.error("❌ Belum ada model")
                if st.button(f"📈 Latih Model {digit}"):
                    X, y_all = preprocess_data(df)
                    y = y_all[i]
                    model = build_model(input_len=X.shape[1])
                    model.fit(X, y, epochs=50, batch_size=16, verbose=0, validation_split=0.2)
                    model.save(model_path)
                    st.success(f"✅ Model {digit} berhasil dilatih.")

# 🔮 Prediksi
if st.button("🔮 Prediksi"):
    if len(df) < 30:
        st.warning("❌ Minimal 30 data diperlukan.")
    else:
        with st.spinner("🔮 Prediksi sedang diproses..."):
            result = (
                top6_markov(df)[0] if metode == "Markov" else
                top6_markov_order2(df) if metode == "Markov Order-2" else
                top6_markov_hybrid(df, digit_weights=digit_weight_input) if metode == "Markov Gabungan" else
                top6_lstm(df, lokasi=selected_lokasi) if metode == "LSTM AI" else
                top6_ensemble(df, lokasi=selected_lokasi)
            )

        if result:
            with st.expander("🎯 Hasil Prediksi"):
                for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                    st.markdown(f"**{label}:** {', '.join(map(str, result[i]))}")

            with st.expander("💡 Kombinasi 4D"):
                top_komb = (
                    kombinasi_4d(df, lokasi=selected_lokasi, top_n=10) if metode == "LSTM AI" else
                    kombinasi_4d_markov_hybrid(df, top_n=10, digit_weights={
                        "ribuan": digit_weight_input[0],
                        "ratusan": digit_weight_input[1],
                        "puluhan": digit_weight_input[2],
                        "satuan": digit_weight_input[3],
                    }) if metode == "Markov Gabungan" else None
                )
                if top_komb:
                    for komb, score in top_komb:
                        st.markdown(f"**{komb}** — ⚡ Confidence: `{score:.6f}`")
