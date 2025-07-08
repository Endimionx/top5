import streamlit as st
import pandas as pd
import requests
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from markov_model import top6_markov, top6_markov_order2, top6_markov_hybrid
from ai_model import (
    top6_model,
    train_and_save_model,
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

def fetch_data(lokasi, hari, putaran):
    url = f"https://wysiwygscan.com/api?pasaran={lokasi.lower()}&hari={hari}&putaran={putaran}&format=json&urut=asc"
    headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
    response = requests.get(url, headers=headers)
    data = response.json()
    return [item["result"] for item in data.get("data", []) if len(item["result"]) == 4 and item["result"].isdigit()]

lottie_predict = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_kkflmtur.json")
st_lottie(lottie_predict, speed=1, height=150, key="prediksi")

st.title("🔮 Prediksi 4D - AI & Markov")

# Sidebar
hari_list = ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"]
metode_list = ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI", "Ensemble AI + Markov"]
model_type = "lstm"

with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    selected_hari = st.selectbox("📅 Pilih Hari", hari_list)
    use_auto = st.checkbox("🔍 Cari Putaran Terbaik Otomatis")
    max_auto_putaran = st.number_input("🔢 Maks Putaran Otomatis", min_value=50, max_value=1000, value=300, step=50) if use_auto else None
    putaran = st.slider("🔁 Jumlah Putaran", 1, 1000, 100) if not use_auto else None
    jumlah_uji = st.number_input("📊 Data Uji Akurasi", min_value=1, max_value=200, value=10)
    metode = st.selectbox("🧠 Metode Prediksi", metode_list)

    min_conf = 0.005
    power = 1.5
    temperature = 0.5
    voting_mode = "product"
    model_type = "lstm"

    if metode in ["LSTM AI", "Ensemble AI + Markov"]:
        min_conf = st.slider("🔎 Minimum Confidence", 0.0001, 0.01, 0.0005, step=0.0001, format="%.4f")
        power = st.slider("📈 Confidence Power", 0.5, 3.0, 1.5, step=0.1)
        temperature = st.slider("🌡️ Temperature Scaling", 0.1, 2.0, 0.5, step=0.1)
        voting_mode = st.selectbox("⚖️ Kombinasi Mode", ["product", "average"])
        use_transformer = st.checkbox("🧠 Gunakan Transformer")
        model_type = "transformer" if use_transformer else "lstm"

# Cari putaran terbaik otomatis
angka_list = []
riwayat_input = ""
best_putaran = putaran

if selected_lokasi and selected_hari:
    try:
        if use_auto:
            st.info("🚀 Mencari putaran terbaik otomatis...")
            best_acc = -1
            for p in range(50, max_auto_putaran + 1, 50):
                data_try = fetch_data(selected_lokasi, selected_hari, p)
                df_try = pd.DataFrame({"angka": data_try})
                if len(df_try) < 11: continue
                pred = top6_model(df_try, lokasi=selected_lokasi, model_type=model_type)
                if not pred: continue
                uji_df = df_try.tail(10)
                total, benar = 0, 0
                for i in range(len(uji_df)):
                    actual = f"{int(uji_df.iloc[i]['angka']):04d}"
                    for j in range(4):
                        if int(actual[j]) in pred[j]:
                            benar += 1
                        total += 1
                acc = benar / total * 100 if total else 0
                if acc > best_acc:
                    best_acc = acc
                    best_putaran = p
            st.success(f"✅ Putaran terbaik: {best_putaran} (akurasi {best_acc:.2f}%)")

        with st.spinner(f"🔄 Mengambil data dari API (putaran {best_putaran})..."):
            angka_list = fetch_data(selected_lokasi, selected_hari, best_putaran)
            riwayat_input = "\n".join(angka_list)
            st.success(f"✅ {len(angka_list)} angka berhasil diambil.")
            with st.expander("📥 Lihat Data"):
                st.code(riwayat_input, language="text")
    except Exception as e:
        st.error(f"❌ Gagal ambil data API: {e}")

df = pd.DataFrame({"angka": angka_list})

# Manajemen Model
if metode == "LSTM AI":
    with st.expander("⚙️ Manajemen Model"):
        for i in range(4):
            model_path = f"saved_models/{selected_lokasi.lower().replace(' ', '_')}_digit{i}_{model_type}.h5"
            col1, col2 = st.columns([2, 1])
            with col1:
                if os.path.exists(model_path):
                    st.info(f"📂 Model Digit-{i} tersedia ({model_type}).")
                else:
                    st.warning(f"⚠️ Model Digit-{i} belum tersedia.")
            with col2:
                if os.path.exists(model_path):
                    if st.button(f"🗑 Hapus Digit-{i}", key=f"hapus_digit_{i}"):
                        os.remove(model_path)
                        st.warning(f"✅ Model Digit-{i} dihapus.")

        if st.button("📚 Latih & Simpan Semua Model"):
            with st.spinner(f"🔄 Melatih semua model per digit ({model_type})..."):
                train_and_save_model(df, selected_lokasi, model_type=model_type)
            st.success("✅ Semua model berhasil dilatih dan disimpan.")

# Tombol Prediksi
if st.button("🔮 Prediksi"):
    if len(df) < 11:
        st.warning("❌ Minimal 11 data diperlukan.")
    else:
        with st.spinner("⏳ Melakukan prediksi..."):
            result, probs = None, None
            if metode == "Markov":
                result, _ = top6_markov(df)
            elif metode == "Markov Order-2":
                result = top6_markov_order2(df)
            elif metode == "Markov Gabungan":
                result = top6_markov_hybrid(df)
            elif metode == "LSTM AI":
                pred = top6_model(df, lokasi=selected_lokasi, model_type=model_type, return_probs=True, temperature=temperature)
                if pred:
                    result, probs = pred
            elif metode == "Ensemble AI + Markov":
                pred = top6_model(df, lokasi=selected_lokasi, model_type=model_type, return_probs=True, temperature=temperature)
                if pred:
                    result, probs = pred
                    markov_result, _ = top6_markov(df)
                    if markov_result:
                        ensemble = []
                        for i in range(4):
                            combined = result[i] + markov_result[i]
                            freq = {x: combined.count(x) for x in set(combined)}
                            top6 = sorted(freq.items(), key=lambda x: -x[1])[:6]
                            ensemble.append([x[0] for x in top6])
                        result = ensemble

        if result is None:
            st.error("❌ Gagal melakukan prediksi.")
        else:
            with st.expander("🎯 Hasil Prediksi Top 6 Digit"):
                col1, col2 = st.columns(2)
                for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                    with (col1 if i % 2 == 0 else col2):
                        st.markdown(f"**{label}:** {', '.join(map(str, result[i]))}")

            if metode in ["LSTM AI", "Ensemble AI + Markov"] and probs:
                with st.expander("📊 Confidence Bar per Digit"):
                    for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                        st.markdown(f"**🔢 {label}**")
                        digit_data = pd.DataFrame({
                            "Digit": [str(d) for d in result[i]],
                            "Confidence": probs[i]
                        }).sort_values(by="Confidence", ascending=True)
                        st.bar_chart(digit_data.set_index("Digit"))

            if metode in ["LSTM AI", "Ensemble AI + Markov"]:
                with st.spinner("🔢 Menghitung kombinasi 4D terbaik..."):
                    top_komb = kombinasi_4d(df, lokasi=selected_lokasi, model_type=model_type,
                                            top_n=10, min_conf=min_conf, power=power, mode=voting_mode)
                    if top_komb:
                        with st.expander("💡 Simulasi Kombinasi 4D Terbaik"):
                            sim_col = st.columns(2)
                            for i, (komb, score) in enumerate(top_komb):
                                with sim_col[i % 2]:
                                    st.markdown(f"`{komb}` - ⚡️ Confidence: `{score:.4f}`")
