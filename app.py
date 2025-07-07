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

st.set_page_config(page_title="Prediksi 4D AI", layout="wide")

# Sidebar
with st.sidebar:
    st.title("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("📍 Pilih Pasaran", lokasi_list)
    mode_auto = st.toggle("🔍 Cari Putaran Terbaik Otomatis", value=True)
    if not mode_auto:
        putaran = st.slider("🔁 Jumlah Putaran", 30, 1000, 100)
    jumlah_uji = st.number_input("📊 Data Uji Akurasi", 5, 200, 10)
    metode = st.selectbox("🤖 Metode Prediksi", [
        "Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI", "Ensemble AI + Markov"
    ])
    min_conf = 0.0005
    power = 1.5
    if "LSTM" in metode:
        min_conf = st.slider("🔎 Minimum Confidence", 0.0001, 0.001, 0.0005, step=0.0001, format="%.4f")
        power = st.slider("📈 Confidence Power", 0.5, 3.0, 1.5, step=0.1)

# Ambil data dari API
@st.cache_data(show_spinner=False)
def ambil_data(lokasi, jumlah):
    try:
        url = f"https://wysiwygscan.com/api?pasaran={lokasi.lower()}&hari=harian&putaran={jumlah}&format=json&urut=asc"
        headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
        response = requests.get(url, headers=headers)
        data = response.json().get("data", [])
        hasil = [x["result"] for x in data if len(x["result"]) == 4 and x["result"].isdigit()]
        return hasil
    except:
        return []

angka_list = []
df = pd.DataFrame()

if selected_lokasi:
    st.header("🔮 Prediksi 4D - AI & Markov")
    if mode_auto:
        st.info("🔎 Menganalisis putaran terbaik... (mohon tunggu)")
        best_acc, best_data = 0.0, []
        best_n = 100
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
                    pred = top6_lstm(subset, lokasi=selected_lokasi) if "LSTM" in metode else top6_markov(subset)[0]
                    actual = f"{int(uji_df.iloc[i]['angka']):04d}"
                    for j in range(4):
                        if int(actual[j]) in pred[j]:
                            benar += 1
                        total += 1
                except:
                    continue
            acc = benar/total*100 if total > 0 else 0
            if acc > best_acc:
                best_acc = acc
                best_data = data
                best_n = p
        angka_list = best_data
        st.success(f"🏆 Putaran terbaik: {best_n} (akurasi {best_acc:.1f}%)")
        st.info(f"📦 Data digunakan: {len(angka_list)} angka terakhir")
    else:
        angka_list = ambil_data(selected_lokasi, putaran)
        st.success(f"✅ {len(angka_list)} data berhasil diambil.")

    if angka_list:
        df = pd.DataFrame({"angka": angka_list})
        with st.expander("📄 Lihat Data"):
            st.code("\n".join(angka_list), language="text")

# Manajemen Model
if metode == "LSTM AI":
    st.subheader("🧠 Model LSTM")
    for i in range(4):
        model_path = f"saved_models/{selected_lokasi.lower().replace(' ', '_')}_digit{i}.h5"
        col1, col2 = st.columns([3, 1])
        with col1:
            if os.path.exists(model_path):
                st.info(f"✅ Model Digit-{i} tersedia.")
            else:
                st.warning(f"⚠️ Model Digit-{i} belum tersedia.")
        with col2:
            if os.path.exists(model_path):
                if st.button(f"Hapus", key=f"del{i}"):
                    os.remove(model_path)
                    st.warning(f"🗑️ Model Digit-{i} dihapus.")

    if not model_exists(selected_lokasi):
        if st.button("📚 Latih & Simpan Semua Model"):
            with st.spinner("Melatih semua model per digit..."):
                train_and_save_lstm(df, selected_lokasi)
            st.success("✅ Semua model berhasil dilatih.")

# Prediksi
if st.button("🔮 Prediksi Sekarang"):
    if len(df) < 30:
        st.warning("❗ Minimal 30 data dibutuhkan.")
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
            st.error("❌ Gagal prediksi.")
        else:
            st.subheader("🎯 Top 6 Tiap Digit")
            col1, col2 = st.columns(2)
            label = ["Ribuan", "Ratusan", "Puluhan", "Satuan"]
            for i in range(4):
                with (col1 if i % 2 == 0 else col2):
                    st.markdown(f"**{label[i]}**: {', '.join(map(str, result[i]))}")

            if "LSTM" in metode:
                with st.spinner("🔢 Simulasi Kombinasi 4D Terbaik..."):
                    top_komb = kombinasi_4d(df, lokasi=selected_lokasi, top_n=10, min_conf=min_conf, power=power)
                    if top_komb:
                        with st.expander("💡 Kombinasi 4D"):
                            for k, (komb, score) in enumerate(top_komb):
                                st.markdown(f"`{komb}` - ⚡ Confidence: `{score:.4f}`")

        with st.spinner("📏 Menghitung akurasi..."):
            uji_df = df.tail(min(jumlah_uji, len(df)))
            total, benar = 0, 0
            akurasi_list = []
            digit_acc = {"Ribuan": [], "Ratusan": [], "Puluhan": [], "Satuan": []}
            for i in range(len(uji_df)):
                subset_df = df.iloc[:-(len(uji_df)-i)]
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
                    actual = f"{int(uji_df.iloc[i]['angka']):04d}"
                    skor = 0
                    for j, key in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                        if int(actual[j]) in pred[j]:
                            skor += 1
                            digit_acc[key].append(1)
                        else:
                            digit_acc[key].append(0)
                    total += 4
                    benar += skor
                    akurasi_list.append(skor / 4 * 100)
                except:
                    continue

            if total > 0:
                st.success(f"📈 Akurasi: {benar/total*100:.2f}%")
                with st.expander("📊 Grafik Akurasi"):
                    st.line_chart(pd.DataFrame({"Akurasi (%)": akurasi_list}))
                with st.expander("🔥 Heatmap per Digit"):
                    heat_df = pd.DataFrame({k: [sum(v)/len(v)*100 if v else 0] for k, v in digit_acc.items()})
                    fig, ax = plt.subplots()
                    sns.heatmap(heat_df, annot=True, cmap="YlGnBu", fmt=".1f", ax=ax)
                    st.pyplot(fig)
            else:
                st.warning("⚠️ Tidak cukup data untuk evaluasi.")
