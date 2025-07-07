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

st.set_page_config(page_title="Prediksi Togel AI", layout="wide")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200: return None
    return r.json()

def ambil_data(lokasi, putaran):
    try:
        url = f"https://wysiwygscan.com/api?pasaran={lokasi.lower()}&hari=harian&putaran={putaran}&format=json&urut=asc"
        headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
        response = requests.get(url, headers=headers)
        data = response.json()
        return [d['result'] for d in data.get('data', []) if len(d['result']) == 4 and d['result'].isdigit()]
    except:
        return []

lottie_predict = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_kkflmtur.json")
st_lottie(lottie_predict, speed=1, height=150, key="prediksi")

st.title("🔮 Prediksi 4D - AI & Markov")

metode_list = ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI", "Ensemble AI + Markov"]

with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    mode_auto = st.checkbox("🔎 Cari Putaran Otomatis", value=True)
    putaran = st.slider("🔁 Jumlah Putaran", 50, 1000, 200, step=50, disabled=mode_auto)
    jumlah_uji = st.number_input("📊 Data Uji Akurasi", 5, 100, 10)
    metode = st.selectbox("🧠 Metode Prediksi", metode_list)
    min_conf = 0.0005
    power = 1.5
    if metode in ["LSTM AI", "Ensemble AI + Markov"]:
        min_conf = st.slider("🔎 Minimum Confidence", 0.0001, 0.001, 0.0005, step=0.0001, format="%.4f")
        power = st.slider("📈 Confidence Weight Power", 0.5, 3.0, 1.5, step=0.1)

angka_list = []
if selected_lokasi:
    if mode_auto:
        with st.spinner("🔎 Menganalisis putaran terbaik..."):
            best_acc, best_data = 0.0, []
            for p in range(50, 301, 10):
                data = ambil_data(selected_lokasi, p)
                if len(data) < 30: continue
                temp_df = pd.DataFrame({"angka": data})
                uji_df = temp_df.tail(min(jumlah_uji, len(temp_df)))
                total, benar = 0, 0
                for i in range(len(uji_df)):
                    subset = temp_df.iloc[:-(len(uji_df)-i)]
                    if len(subset) < 30: continue
                    try:
                        pred = top6_lstm(subset, lokasi=selected_lokasi)
                        if pred is None: continue
                        actual = f"{int(uji_df.iloc[i]['angka']):04d}"
                        for j in range(4):
                            if int(actual[j]) in pred[j]:
                                benar += 1
                            total += 1
                    except: continue
                acc = benar/total*100 if total > 0 else 0
                if acc > best_acc:
                    best_acc = acc
                    best_data = data
        angka_list = best_data
        if len(angka_list) >= 30:
            st.success(f"🏆 Putaran terbaik: {len(angka_list)} data (akurasi {best_acc:.1f}%)")
        else:
            st.warning("⚠️ Tidak cukup data valid ditemukan.")
    else:
        angka_list = ambil_data(selected_lokasi, putaran)
        st.info(f"📦 Data digunakan: {len(angka_list)} angka terakhir")

df = pd.DataFrame({"angka": angka_list})
if angka_list:
    with st.expander("📥 Lihat Data"):
        st.code("\n".join(angka_list))

if metode == "LSTM AI":
    with st.expander("⚙️ Manajemen Model LSTM"):
        for i in range(4):
            path = f"saved_models/{selected_lokasi.lower().replace(' ', '_')}_digit{i}.h5"
            col1, col2 = st.columns([2,1])
            with col1:
                if os.path.exists(path): st.info(f"📂 Model Digit-{i} tersedia.")
                else: st.warning(f"⚠️ Model Digit-{i} belum tersedia.")
            with col2:
                if os.path.exists(path):
                    if st.button(f"🗑 Hapus Digit-{i}", key=f"hapus{i}"):
                        os.remove(path)
                        st.warning(f"✅ Model Digit-{i} dihapus.")
        if st.button("📚 Latih & Simpan Semua Model"):
            with st.spinner("🔄 Melatih semua model..."):
                train_and_save_lstm(df, selected_lokasi)
            st.success("✅ Semua model berhasil disimpan.")

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
                    with (col1 if i%2==0 else col2):
                        st.markdown(f"**{label}:** {', '.join(map(str, result[i]))}")
            if metode in ["LSTM AI", "Ensemble AI + Markov"]:
                with st.spinner("🔢 Kombinasi 4D terbaik..."):
                    top_komb = kombinasi_4d(df, lokasi=selected_lokasi, top_n=10, min_conf=min_conf, power=power)
                    if top_komb:
                        with st.expander("💡 Simulasi Kombinasi 4D Terbaik"):
                            sim_col = st.columns(2)
                            for i, (komb, score) in enumerate(top_komb):
                                with sim_col[i%2]:
                                    st.markdown(f"`{komb}` - ⚡️ Confidence: `{score:.4f}`")

        with st.spinner("📏 Menghitung akurasi..."):
            uji_df = df.tail(min(jumlah_uji, len(df)))
            total, benar = 0, 0
            akurasi_list = []
            digit_acc = {"Ribuan": [], "Ratusan": [], "Puluhan": [], "Satuan": []}
            for i in range(len(uji_df)):
                subset_df = df.iloc[:-(len(uji_df)-i)]
                if len(subset_df) < 30: continue
                try:
                    pred = (
                        top6_markov(subset_df)[0] if metode == "Markov" else
                        top6_markov_order2(subset_df) if metode == "Markov Order-2" else
                        top6_markov_hybrid(subset_df) if metode == "Markov Gabungan" else
                        top6_lstm(subset_df, lokasi=selected_lokasi) if metode == "LSTM AI" else
                        top6_ensemble(subset_df, lokasi=selected_lokasi)
                    )
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
                    akurasi_list.append(skor/4*100)
                except: continue
            if total > 0:
                st.success(f"📈 Akurasi {metode}: {benar/total*100:.2f}%")
                with st.expander("📊 Grafik Akurasi"):
                    st.line_chart(pd.DataFrame({"Akurasi (%)": akurasi_list}))
                with st.expander("🔥 Heatmap Akurasi per Digit"):
                    heat_df = pd.DataFrame({k:[sum(v)/len(v)*100 if v else 0] for k,v in digit_acc.items()})
                    fig, ax = plt.subplots()
                    sns.heatmap(heat_df, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
                    st.pyplot(fig)
            else:
                st.warning("⚠️ Tidak cukup data untuk evaluasi akurasi.")
