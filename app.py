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

def ambil_data(lokasi, putaran):
    try:
        url = f"https://wysiwygscan.com/api?pasaran={lokasi.lower()}&hari=harian&putaran={putaran}&format=json&urut=asc"
        headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
        response = requests.get(url, headers=headers)
        data = response.json()
        angka_list = [item["result"] for item in data.get("data", []) if len(item["result"]) == 4 and item["result"].isdigit()]
        return angka_list
    except:
        return []

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_predict = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_kkflmtur.json")
st_lottie(lottie_predict, speed=1, height=150, key="prediksi")

st.title("🔮 Prediksi 4D - AI & Markov")

# Sidebar
metode_list = ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI", "Ensemble AI + Markov"]
selected_lokasi = st.sidebar.selectbox("🌍 Pilih Pasaran", lokasi_list)
metode = st.sidebar.selectbox("🧠 Metode Prediksi", metode_list)
jumlah_uji = st.sidebar.number_input("📊 Jumlah Data Uji Akurasi", min_value=5, max_value=300, value=50)
mode_auto = st.sidebar.checkbox("⚡ Cari Putaran Otomatis")
min_conf = st.sidebar.slider("🔎 Minimum Confidence", 0.0001, 0.001, 0.0005, step=0.0001, format="%.4f") if "LSTM" in metode else 0.0005
power = st.sidebar.slider("📈 Confidence Weight Power", 0.5, 3.0, 1.5, step=0.1) if "LSTM" in metode else 1.5

angka_list = []
if selected_lokasi:
    if mode_auto:
        with st.spinner("🔎 Menganalisis putaran terbaik..."):
            best_acc, best_data = 0.0, []
            for p in range(50, 301, 10):
                data = ambil_data(selected_lokasi, p)
                if len(data) < 30:
                    continue
                df = pd.DataFrame({"angka": data})
                pred = top6_lstm(df, lokasi=selected_lokasi) if metode == "LSTM AI" else top6_ensemble(df, lokasi=selected_lokasi)
                if not pred:
                    continue
                uji_df = df.tail(10)
                total, benar = 0, 0
                for i in range(len(uji_df)):
                    sub = df.iloc[:-(len(uji_df) - i)]
                    if len(sub) < 30:
                        continue
                    hasil = top6_lstm(sub, lokasi=selected_lokasi) if metode == "LSTM AI" else top6_ensemble(sub, lokasi=selected_lokasi)
                    if not hasil: continue
                    actual = f"{int(uji_df.iloc[i]['angka']):04d}"
                    for j in range(4):
                        total += 1
                        if int(actual[j]) in hasil[j]: benar += 1
                if total > 0:
                    acc = benar / total
                    if acc > best_acc:
                        best_acc = acc
                        best_data = data
        if best_data and len(best_data) >= 30:
            angka_list = best_data
            st.success(f"🏆 Putaran terbaik ditemukan: {len(angka_list)} data (akurasi {best_acc*100:.1f}%)")
        else:
            st.error("❌ Gagal menemukan putaran terbaik. Gunakan mode manual.")
    else:
        putaran = st.sidebar.slider("🔁 Jumlah Putaran", 10, 300, 100)
        with st.spinner("🔄 Mengambil data..."):
            angka_list = ambil_data(selected_lokasi, putaran)

if angka_list:
    st.success(f"✅ {len(angka_list)} angka berhasil diambil.")
    with st.expander("📥 Lihat Data"):
        st.code("\n".join(angka_list))
else:
    st.warning("⚠️ Data belum tersedia atau gagal diambil.")

df = pd.DataFrame({"angka": angka_list})

# Manajemen Model
if metode == "LSTM AI":
    with st.expander("⚙️ Manajemen Model LSTM"):
        for i in range(4):
            path = f"saved_models/{selected_lokasi.lower().replace(' ', '_')}_digit{i}.h5"
            col1, col2 = st.columns([2, 1])
            with col1:
                st.info(f"📂 Digit-{i} model {'tersedia' if os.path.exists(path) else '❌ belum ada'}")
            with col2:
                if os.path.exists(path):
                    if st.button(f"🗑 Hapus Digit-{i}", key=f"hapus{i}"):
                        os.remove(path)
                        st.warning(f"✅ Model Digit-{i} dihapus.")
        if st.button("📚 Latih Semua Model"):
            with st.spinner("🔁 Melatih model per digit..."):
                train_and_save_lstm(df, selected_lokasi)
            st.success("✅ Model berhasil dilatih dan disimpan.")

# Prediksi
if st.button("🔮 Prediksi"):
    if len(df) < 30:
        st.warning("❌ Minimal 30 data diperlukan.")
    else:
        with st.spinner("⏳ Memproses prediksi..."):
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
            with st.expander("🎯 Top 6 Prediksi per Digit"):
                for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                    st.markdown(f"**{label}:** {', '.join(map(str, result[i]))}")

            if metode in ["LSTM AI", "Ensemble AI + Markov"]:
                with st.spinner("🔢 Menghitung kombinasi 4D terbaik..."):
                    top_komb = kombinasi_4d(df, lokasi=selected_lokasi, top_n=10, min_conf=min_conf, power=power)
                    if top_komb:
                        with st.expander("💡 Simulasi Kombinasi 4D Terbaik"):
                            for k, (komb, score) in enumerate(top_komb):
                                st.markdown(f"`{komb}` - ⚡ Confidence: `{score:.4f}``")

# Akurasi
if len(df) >= 40:
    with st.spinner("📏 Menghitung akurasi prediksi..."):
        uji_df = df.tail(jumlah_uji)
        total, benar = 0, 0
        akurasi_list = []
        digit_acc = {"Ribuan": [], "Ratusan": [], "Puluhan": [], "Satuan": []}
        for i in range(len(uji_df)):
            subset = df.iloc[:-(len(uji_df) - i)]
            if len(subset) < 30:
                continue
            pred = (
                top6_markov(subset)[0] if metode == "Markov" else
                top6_markov_order2(subset) if metode == "Markov Order-2" else
                top6_markov_hybrid(subset) if metode == "Markov Gabungan" else
                top6_lstm(subset, lokasi=selected_lokasi) if metode == "LSTM AI" else
                top6_ensemble(subset, lokasi=selected_lokasi)
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
else:
    st.info("ℹ️ Minimal 40 data untuk mengevaluasi akurasi.")
