import streamlit as st
import pandas as pd
import requests
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from markov_model import top6_markov, top6_markov_order2, top6_markov_hybrid
from ai_model import (
    top6_lstm,
    train_and_save_lstm,
    kombinasi_4d,
    top6_ensemble,
    model_exists
)
from lokasi_list import lokasi_list
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Prediksi 4D AI", layout="wide")

def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

lottie_predict = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_kkflmtur.json")
if lottie_predict:
    st_lottie(lottie_predict, speed=1, height=150, key="predict")

st.title("🔮 Prediksi 4D - AI & Markov")

hari_list = ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"]
metode_list = ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI", "Ensemble AI + Markov"]

with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    mode_otomatis = st.checkbox("🔁 Cari Putaran Terbaik Otomatis", value=False)
    putaran = 100 if mode_otomatis else st.slider("🔁 Jumlah Putaran", 30, 1000, 100, step=10)
    selected_hari = st.selectbox("📅 Pilih Hari", hari_list)
    jumlah_uji = st.number_input("📊 Data Uji Akurasi", min_value=5, max_value=200, value=10)
    metode = st.selectbox("🧠 Metode Prediksi", metode_list)
    min_conf = 0.0005
    power = 1.5
    if metode in ["LSTM AI", "Ensemble AI + Markov"]:
        min_conf = st.slider("🔎 Minimum Confidence", 0.0001, 0.001, 0.0005, step=0.0001, format="%.4f")
        power = st.slider("📈 Confidence Weight Power", 0.5, 3.0, 1.5, step=0.1)

angka_list = []
if selected_lokasi and selected_hari:
    try:
        with st.spinner("🔄 Mengambil data dari API..."):
            url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran={putaran}&format=json&urut=asc"
            headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
            response = requests.get(url, headers=headers)
            data = response.json()
            angka_list = [d["result"] for d in data.get("data", []) if len(d["result"]) == 4 and d["result"].isdigit()]
            st.success(f"✅ {len(angka_list)} angka berhasil diambil.")
            with st.expander("📥 Lihat Data"):
                st.code("\n".join(angka_list), language="text")
    except Exception as e:
        st.error(f"❌ Gagal ambil data API: {e}")

df = pd.DataFrame({"angka": angka_list})

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
            with st.spinner("🔄 Melatih semua model..."):
                train_and_save_lstm(df, selected_lokasi)
            st.success("✅ Semua model berhasil disimpan.")

if st.button("🔮 Prediksi"):
    if len(df) < 30:
        st.warning("❌ Minimal 30 data diperlukan.")
    else:
        result = None
        with st.spinner("⏳ Melakukan prediksi..."):
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
            with st.expander("🎯 Hasil Prediksi"):
                for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                    st.markdown(f"**{label}:** {', '.join(map(str, result[i]))}")

            if metode in ["LSTM AI", "Ensemble AI + Markov"]:
                with st.spinner("🧮 Menghitung kombinasi terbaik..."):
                    top_komb = kombinasi_4d(df, lokasi=selected_lokasi, top_n=10, min_conf=min_conf, power=power)
                    if top_komb:
                        with st.expander("💡 Kombinasi 4D"):
                            for komb, sc in top_komb:
                                st.markdown(f"`{komb}` - ⚡️ `{sc:.4f}`")

if mode_otomatis and metode in ["LSTM AI", "Ensemble AI + Markov"]:
    best_acc, best_df, best_p = 0, None, None
    if not model_exists(selected_lokasi):
        with st.spinner("⚙️ Model belum tersedia, sedang dilatih otomatis..."):
            train_and_save_lstm(df, selected_lokasi)
            st.success("✅ Model berhasil dilatih.")
    with st.spinner("🔍 Menganalisis putaran terbaik..."):
        for p in range(30, len(df) - jumlah_uji):
            subset = df[-(p+jumlah_uji):-jumlah_uji]
            test = df[-jumlah_uji:]
            if len(subset) < 30: continue
            pred = top6_lstm(subset, lokasi=selected_lokasi) if metode == "LSTM AI" else top6_ensemble(subset, lokasi=selected_lokasi)
            if pred is None: continue
            total, benar = 0, 0
            for i in range(len(test)):
                try:
                    actual = f"{int(test.iloc[i]['angka']):04d}"
                    skor = sum(int(actual[j]) in pred[j] for j in range(4))
                    total += 4
                    benar += skor
                except: continue
            acc = benar / total * 100 if total else 0
            if acc > best_acc:
                best_acc = acc
                best_df = subset
                best_p = p
        if best_df is not None:
            st.success(f"🏆 Putaran terbaik: {best_p}, Akurasi: {best_acc:.2f}%")
            st.write(f"📊 Data digunakan: {len(best_df)} angka terakhir")
        else:
            st.error("❌ Gagal menemukan putaran terbaik. Gunakan mode manual.")
