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

st_lottie(load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_kkflmtur.json"), height=150, key="prediksi")
st.title("🔮 Prediksi 4D - AI & Markov")

# Sidebar
hari_list = ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"]
metode_list = ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI", "Ensemble AI + Markov"]

with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    selected_hari = st.selectbox("📅 Pilih Hari", hari_list)
    putaran = st.slider("🔁 Jumlah Putaran", 1, 1000, 100)
    jumlah_uji = st.number_input("📊 Data Uji Akurasi", min_value=1, max_value=200, value=10)
    metode = st.selectbox("🧠 Metode Prediksi", metode_list)

# Ambil Data
angka_list = []
if selected_lokasi and selected_hari:
    try:
        with st.spinner("🔄 Mengambil data dari API..."):
            url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran={putaran}&format=json&urut=asc"
            headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
            response = requests.get(url, headers=headers)
            data = response.json()
            angka_list = [item["result"] for item in data.get("data", []) if len(item["result"]) == 4 and item["result"].isdigit()]
            st.success(f"✅ {len(angka_list)} angka berhasil diambil.")
            with st.expander("📥 Lihat Data"):
                st.code("\n".join(angka_list), language="text")
    except Exception as e:
        st.error(f"❌ Gagal ambil data API: {e}")

df = pd.DataFrame({"angka": angka_list})

# Manajemen Model LSTM
if metode == "LSTM AI":
    with st.expander("⚙️ Manajemen Model LSTM"):
        all_exist = model_exists(selected_lokasi)
        if all_exist:
            st.info(f"📂 Model per digit tersedia untuk pasaran {selected_lokasi}")
        else:
            st.warning("⚠️ Belum semua model digit tersedia.")

        uploaded_files = st.file_uploader("📤 Upload Semua Model Digit (.h5)", type=["h5"], accept_multiple_files=True)
        if uploaded_files:
            for file in uploaded_files:
                name = file.name
                path = f"saved_models/{name}"
                with open(path, "wb") as f:
                    f.write(file.read())
            st.success("✅ Semua model digit berhasil diupload.")

        if st.button("📚 Latih & Simpan Semua Model"):
            with st.spinner("🔄 Melatih model per digit..."):
                train_and_save_lstm(df, selected_lokasi)
            st.success("✅ Semua model digit berhasil dilatih.")

        for i in range(4):
            path = f"saved_models/{selected_lokasi.lower().replace(' ', '_')}_digit{i}.h5"
            if os.path.exists(path):
                with open(path, "rb") as f:
                    st.download_button(f"⬇️ Download Model Digit {i}", f, file_name=os.path.basename(path))
                if st.button(f"🗑 Hapus Model Digit {i}"):
                    os.remove(path)
                    st.warning(f"🗑 Model digit {i} dihapus.")

# Tombol Prediksi
if st.button("🔮 Prediksi"):
    if len(df) < 11:
        st.warning("❌ Minimal 11 data diperlukan.")
    else:
        with st.spinner("⏳ Melakukan prediksi..."):
            result, info = None, {}
            if metode == "Markov":
                result, info = top6_markov(df)
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
                    top_komb = kombinasi_4d(df, lokasi=selected_lokasi, top_n=10)
                    if top_komb:
                        with st.expander("💡 Simulasi Kombinasi 4D Terbaik"):
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
                if len(subset_df) < 11:
                    continue
                pred = (
                    top6_markov(subset_df)[0] if metode == "Markov" else
                    top6_markov_order2(subset_df) if metode == "Markov Order-2" else
                    top6_markov_hybrid(subset_df) if metode == "Markov Gabungan" else
                    top6_lstm(subset_df, lokasi=selected_lokasi) if metode == "LSTM AI" else
                    top6_ensemble(subset_df, lokasi=selected_lokasi)
                )
                actual = f"{int(uji_df.iloc[i]['angka']):04d}"
                if pred is None: continue
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
