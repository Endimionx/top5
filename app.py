import streamlit as st
import pandas as pd
import requests
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from markov_model import top6_markov, top6_markov_order2, top6_markov_hybrid
from ai_model import top6_lstm, train_and_save_lstm, kombinasi_4d, top6_ensemble, model_exists
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
    st_lottie(lottie_predict, speed=1, height=150, key="prediksi")

st.title("🎯 Prediksi Togel 4D - AI & Markov")

# Sidebar
hari_list = ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"]
metode_list = ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI", "Ensemble AI + Markov"]
preset_mode = st.sidebar.radio("📋 Mode", ["Manual", "Auto Cari Putaran Terbaik"])
selected_lokasi = st.sidebar.selectbox("🌍 Pilih Pasaran", lokasi_list)
selected_hari = st.sidebar.selectbox("📅 Pilih Hari", hari_list)
metode = st.sidebar.selectbox("🧠 Metode", metode_list)
jumlah_uji = st.sidebar.number_input("📊 Uji Akurasi", min_value=1, max_value=100, value=10)

min_conf, power = 0.0005, 1.5
if metode in ["LSTM AI", "Ensemble AI + Markov"]:
    min_conf = st.sidebar.slider("🔎 Minimum Confidence", 0.0001, 0.001, 0.0005, step=0.0001, format="%.4f")
    power = st.sidebar.slider("📈 Confidence Weight Power", 0.5, 3.0, 1.5, step=0.1)

putaran = 100
angka_list, best_accuracy, best_putaran = [], 0, 0

# Ambil data
try:
    with st.spinner("🔄 Mengambil data dari API..."):
        url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran=500&format=json&urut=asc"
        headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
        response = requests.get(url, headers=headers)
        data = response.json()
        angka_list = [item["result"] for item in data.get("data", []) if len(item["result"]) == 4 and item["result"].isdigit()]
except Exception as e:
    st.error(f"❌ Gagal ambil data: {e}")

# Auto Putaran Terbaik
if preset_mode == "Auto Cari Putaran Terbaik" and metode in ["LSTM AI", "Ensemble AI + Markov"] and model_exists(selected_lokasi):
    for p in range(40, min(300, len(angka_list)), 10):
        df = pd.DataFrame({"angka": angka_list[-p:]})
        if len(df) < 30: continue
        uji_df = df.tail(jumlah_uji)
        total, benar = 0, 0
        for i in range(len(uji_df)):
            subset = df.iloc[:-(len(uji_df)-i)]
            if len(subset) < 30: continue
            pred = top6_lstm(subset, lokasi=selected_lokasi)
            if pred is None: continue
            actual = f"{int(uji_df.iloc[i]['angka']):04d}"
            for j in range(4):
                if int(actual[j]) in pred[j]:
                    benar += 1
                total += 1
        acc = benar / total if total else 0
        if acc > best_accuracy:
            best_accuracy = acc
            best_putaran = p
    putaran = best_putaran or 100
    st.sidebar.success(f"📌 Putaran terbaik: {putaran} (akurasi {best_accuracy*100:.1f}%)")
elif preset_mode == "Manual":
    putaran = st.sidebar.slider("🔁 Jumlah Putaran", 30, 500, 100)

# Buat DataFrame
df = pd.DataFrame({"angka": angka_list[-putaran:]})
st.success(f"✅ Diambil {len(df)} angka")
with st.expander("📥 Lihat Data"):
    st.code("\n".join(df["angka"].tolist()))

# Manajemen Model LSTM
if metode == "LSTM AI":
    with st.expander("🧠 Manajemen Model AI"):
        for i in range(4):
            model_path = f"saved_models/{selected_lokasi.lower().replace(' ', '_')}_digit{i}.h5"
            col1, col2 = st.columns([2, 1])
            with col1:
                if os.path.exists(model_path):
                    st.info(f"✅ Model Digit-{i} tersedia")
                else:
                    st.warning(f"❌ Model Digit-{i} belum ada")
            with col2:
                if os.path.exists(model_path):
                    if st.button(f"Hapus Digit-{i}", key=f"hapus{i}"):
                        os.remove(model_path)
                        st.warning(f"🗑 Model Digit-{i} dihapus")

        if st.button("📚 Latih & Simpan Semua Model"):
            with st.spinner("🔄 Melatih model..."):
                train_and_save_lstm(df, selected_lokasi)
            st.success("✅ Semua model berhasil disimpan")

# Prediksi
if st.button("🔮 Prediksi"):
    if len(df) < 30:
        st.warning("⚠️ Minimal 30 data dibutuhkan")
    else:
        with st.spinner("🧠 Menghitung prediksi..."):
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
            st.error("❌ Gagal memuat prediksi")
        else:
            with st.expander("🎯 Top 6 Digit"):
                col1, col2 = st.columns(2)
                for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                    with (col1 if i % 2 == 0 else col2):
                        st.write(f"**{label}:** {', '.join(map(str, result[i]))}")

            if metode in ["LSTM AI", "Ensemble AI + Markov"]:
                top_komb = kombinasi_4d(df, lokasi=selected_lokasi, top_n=10, min_conf=min_conf, power=power)
                if top_komb:
                    with st.expander("💡 Kombinasi 4D Confidence Tinggi"):
                        cols = st.columns(2)
                        for i, (komb, score) in enumerate(top_komb):
                            with cols[i % 2]:
                                st.write(f"`{komb}` ⚡ `{score:.4f}`")

        with st.spinner("📏 Evaluasi Akurasi..."):
            uji_df = df.tail(jumlah_uji)
            total, benar, akurasi_list = 0, 0, []
            digit_acc = {"Ribuan": [], "Ratusan": [], "Puluhan": [], "Satuan": []}
            for i in range(len(uji_df)):
                subset = df.iloc[:-(len(uji_df)-i)]
                if len(subset) < 30: continue
                pred = (
                    top6_markov(subset)[0] if metode == "Markov" else
                    top6_markov_order2(subset) if metode == "Markov Order-2" else
                    top6_markov_hybrid(subset) if metode == "Markov Gabungan" else
                    top6_lstm(subset, lokasi=selected_lokasi) if metode == "LSTM AI" else
                    top6_ensemble(subset, lokasi=selected_lokasi)
                )
                if pred is None: continue
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

            if total > 0:
                st.success(f"🎯 Akurasi: {benar / total * 100:.2f}%")
                with st.expander("📊 Grafik Akurasi"):
                    st.line_chart(pd.DataFrame({"Akurasi (%)": akurasi_list}))
                with st.expander("🔥 Heatmap Digit"):
                    heat_df = pd.DataFrame({k: [sum(v)/len(v)*100 if v else 0] for k, v in digit_acc.items()})
                    fig, ax = plt.subplots()
                    sns.heatmap(heat_df, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
                    st.pyplot(fig)
            else:
                st.warning("⚠️ Data tidak cukup untuk evaluasi.")
