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

st.set_page_config(page_title="Prediksi Togel AI", layout="wide")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def cari_putaran_terbaik(df_all, lokasi, metode, jumlah_uji=10):
    best_score, best_n, hasil_all = 0, 0, {}
    for n in range(30, min(len(df_all), 200)):
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

lottie_predict = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_kkflmtur.json")
st_lottie(lottie_predict, speed=1, height=150, key="prediksi")

st.title("🔮 Prediksi 4D - AI & Markov")

# Sidebar
hari_list = ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"]
metode_list = ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI", "Ensemble AI + Markov"]

with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    selected_hari = st.selectbox("📅 Pilih Hari", hari_list)
    jumlah_uji = st.number_input("📊 Data Uji Akurasi", min_value=5, max_value=100, value=10)
    metode = st.selectbox("🧠 Metode Prediksi", metode_list)

    cari_otomatis = st.toggle("🔍 Cari Putaran Otomatis", value=False)
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
            best_n, best_score, _ = cari_putaran_terbaik(df_all, selected_lokasi, metode, jumlah_uji)
        if best_n > 0:
            putaran = best_n
            st.success(f"✅ Putaran terbaik: {best_n} (Akurasi: {best_score:.2f}%)")
        else:
            st.warning("⚠️ Gagal menemukan putaran terbaik.")
    elif not cari_otomatis:
        putaran = st.slider("🔁 Jumlah Putaran", 20, 1000, 100, step=10)

    min_conf = 0.0005
    power = 1.5
    if metode in ["LSTM AI", "Ensemble AI + Markov"]:
        min_conf = st.slider("🔎 Minimum Confidence", 0.0001, 0.001, 0.0005, step=0.0001, format="%.4f")
        power = st.slider("📈 Confidence Weight Power", 0.5, 3.0, 1.5, step=0.1)

# Ambil Data berdasarkan putaran terpilih
angka_list, riwayat_input = [], ""
df = pd.DataFrame()
if selected_lokasi and selected_hari:
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

# Lanjutkan fitur prediksi dan manajemen model di bawah sini...
