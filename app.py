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

def cari_putaran_terbaik(lokasi, max_putaran=300):
    scores = []
    for p in range(30, max_putaran + 1, 10):
        try:
            url = f"https://wysiwygscan.com/api?pasaran={lokasi.lower()}&hari=harian&putaran={p}&format=json&urut=asc"
            headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
            data = requests.get(url, headers=headers).json()
            angka_list = [item["result"] for item in data.get("data", []) if len(item["result"]) == 4 and item["result"].isdigit()]
            df = pd.DataFrame({"angka": angka_list})
            akurasi = evaluasi_akurasi(df, "Markov", max_uji=15, tampil=False)
            scores.append((p, akurasi))
        except:
            continue
    best = max(scores, key=lambda x: x[1], default=(100, 0))
    return best[0]

def evaluasi_akurasi(df, metode, max_uji=10, tampil=True, lokasi=None):
    uji_df = df.tail(min(max_uji, len(df)))
    total, benar = 0, 0
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
                top6_lstm(subset_df, lokasi=lokasi) if metode == "LSTM AI" else
                top6_ensemble(subset_df, lokasi=lokasi)
            )
            if pred is None:
                continue
            actual = f"{int(uji_df.iloc[i]['angka']):04d}"
            for j, label in enumerate(digit_acc.keys()):
                digit_acc[label].append(1 if int(actual[j]) in pred[j] else 0)
            total += 4
            benar += sum(digit_acc[label][-1] for label in digit_acc)
        except:
            continue
    acc = (benar / total * 100) if total else 0
    if tampil:
        st.success(f"📈 Akurasi {metode}: {acc:.2f}%")
        heat_df = pd.DataFrame({k: [sum(v)/len(v)*100 if v else 0] for k, v in digit_acc.items()})
        fig, ax = plt.subplots()
        sns.heatmap(heat_df, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
        st.pyplot(fig)
    return acc

lottie_predict = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_kkflmtur.json")
st_lottie(lottie_predict, speed=1, height=150, key="prediksi")

st.title("🔮 Prediksi 4D - AI & Markov")

hari_list = ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"]
metode_list = ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI", "Ensemble AI + Markov"]

with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    auto_cari = st.checkbox("🔍 Cari Putaran Terbaik", value=True)
    selected_hari = st.selectbox("📅 Pilih Hari", hari_list)
    putaran = 100
    if not auto_cari:
        putaran = st.slider("🔁 Jumlah Putaran", 30, 1000, 100)
    jumlah_uji = st.number_input("📊 Data Uji Akurasi", 5, 100, 10)
    metode = st.selectbox("🧠 Metode Prediksi", metode_list)
    min_conf = 0.0005
    power = 1.5
    if metode in ["LSTM AI", "Ensemble AI + Markov"]:
        min_conf = st.slider("🔎 Minimum Confidence", 0.0001, 0.001, 0.0005, step=0.0001, format="%.4f")
        power = st.slider("📈 Confidence Weight Power", 0.5, 3.0, 1.5, step=0.1)

angka_list, riwayat_input = [], ""
if selected_lokasi and selected_hari:
    try:
        with st.spinner("🔄 Mengambil data..."):
            if auto_cari:
                putaran = cari_putaran_terbaik(selected_lokasi)
            url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran={putaran}&format=json&urut=asc"
            headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
            data = requests.get(url, headers=headers).json()
            angka_list = [d["result"] for d in data.get("data", []) if len(d["result"]) == 4 and d["result"].isdigit()]
            riwayat_input = "\n".join(angka_list)
            st.success(f"✅ {len(angka_list)} data berhasil diambil (Putaran: {putaran})")
            with st.expander("📥 Data"):
                st.code(riwayat_input)
    except Exception as e:
        st.error(f"Gagal ambil data: {e}")

df = pd.DataFrame({"angka": angka_list})

if metode == "LSTM AI":
    with st.expander("⚙️ Manajemen Model"):
        for i in range(4):
            path = f"saved_models/{selected_lokasi.lower().replace(' ', '_')}_digit{i}.h5"
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"Model Digit-{i}: {'✅' if os.path.exists(path) else '❌'}")
            with col2:
                if os.path.exists(path):
                    if st.button(f"Hapus-{i}", key=f"hapus_digit_{i}"):
                        os.remove(path)
                        st.warning(f"Model Digit-{i} dihapus")
        if st.button("📚 Latih Model Semua Digit"):
            with st.spinner("Melatih model..."):
                train_and_save_lstm(df, selected_lokasi)
            st.success("✅ Pelatihan selesai")

if st.button("🔮 Prediksi"):
    if len(df) < 30:
        st.warning("❌ Minimal 30 data diperlukan.")
    else:
        with st.spinner("⏳ Memproses..."):
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
            with st.expander("🎯 Prediksi Top-6 per Digit"):
                col1, col2 = st.columns(2)
                for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                    with (col1 if i % 2 == 0 else col2):
                        st.markdown(f"**{label}:** {', '.join(map(str, result[i]))}")
            if metode in ["LSTM AI", "Ensemble AI + Markov"]:
                top_komb = kombinasi_4d(df, selected_lokasi, top_n=10, min_conf=min_conf, power=power)
                if top_komb:
                    with st.expander("🎰 Kombinasi 4D Confidence"):
                        sim_col = st.columns(2)
                        for i, (komb, score) in enumerate(top_komb):
                            with sim_col[i % 2]:
                                st.markdown(f"`{komb}` - ⚡️ {score:.4f}")
        else:
            st.error("❌ Gagal memprediksi.")

if st.button("📏 Evaluasi Akurasi"):
    if len(df) < 40:
        st.warning("Data terlalu sedikit untuk evaluasi.")
    else:
        evaluasi_akurasi(df, metode, max_uji=jumlah_uji, lokasi=selected_lokasi)
