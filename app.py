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

st.set_page_config(page_title="Prediksi Togel AI", layout="wide")

st.title("🔮 Prediksi 4D - AI & Markov")

# Sidebar
with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    cari_otomatis = st.checkbox("🔍 Cari Putaran Terbaik Otomatis", value=True)
    manual_putaran = st.slider("🔁 Jumlah Putaran Manual", 50, 1000, 200, step=50, disabled=cari_otomatis)
    jumlah_uji = st.number_input("📊 Data Uji Akurasi", min_value=1, max_value=100, value=10)
    metode = st.selectbox("🧠 Metode Prediksi", ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI", "Ensemble AI + Markov"])
    min_conf = st.slider("🔎 Minimum Confidence", 0.0001, 0.0010, 0.0005, step=0.0001, format="%.4f") if "LSTM" in metode else 0.0005
    power = st.slider("⚡️ Confidence Power", 0.5, 3.0, 1.5, 0.1) if "LSTM" in metode else 1.5

# Ambil data
angka_list = []
try:
    with st.spinner("🔄 Mengambil data..."):
        url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari=harian&putaran=1000&format=json&urut=asc"
        headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
        response = requests.get(url, headers=headers)
        data = response.json()
        angka_list = [d["result"] for d in data.get("data", []) if len(d["result"]) == 4 and d["result"].isdigit()]
except:
    st.error("❌ Gagal mengambil data.")
    st.stop()

df_full = pd.DataFrame({"angka": angka_list})
st.success(f"✅ {len(df_full)} data berhasil diambil.")

# Cari putaran terbaik otomatis
selected_putaran = manual_putaran
df = df_full.copy()
if cari_otomatis:
    best_score, best_p = 0, 0
    st.info("🔎 Menganalisis putaran terbaik... (mohon tunggu)")
    for p in range(50, 501, 50):
        subset = df_full.tail(p)
        if len(subset) < 30:
            continue
        pred = top6_lstm(subset, lokasi=selected_lokasi)
        if pred is None:
            continue
        # Evaluasi cepat 5 data
        uji = subset.tail(5)
        benar, total = 0, 0
        for i in range(len(uji)):
            sub = subset.iloc[:-(len(uji)-i)]
            if len(sub) < 30:
                continue
            p_pred = top6_lstm(sub, lokasi=selected_lokasi)
            if p_pred is None:
                continue
            actual = f"{int(uji.iloc[i]['angka']):04d}"
            for j in range(4):
                if int(actual[j]) in p_pred[j]:
                    benar += 1
                total += 1
        if total > 0 and benar/total > best_score:
            best_score = benar/total
            best_p = p
    selected_putaran = best_p or 100
    st.success(f"🏆 Putaran terbaik: {selected_putaran} (akurasi {best_score*100:.1f}%)")

# Gunakan data final
df = df_full.tail(selected_putaran)
st.markdown(f"📦 Data digunakan: `{selected_putaran}` angka terakhir")

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
                if st.button(f"🗑 Hapus Digit-{i}", key=f"hapus_{i}"):
                    os.remove(model_path)
                    st.warning(f"🚮 Model Digit-{i} dihapus.")

    if st.button("📚 Latih Model Sekarang"):
        with st.spinner("🔧 Melatih model per digit..."):
            train_and_save_lstm(df, selected_lokasi)
        st.success("✅ Semua model berhasil dilatih.")

# Tombol Prediksi
if st.button("🔮 Prediksi"):
    if len(df) < 30:
        st.warning("❌ Minimal 30 data diperlukan.")
        st.stop()

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
        st.error("❌ Gagal memprediksi.")
        st.stop()

    st.subheader("🎯 Prediksi Top-6 per Digit")
    col1, col2 = st.columns(2)
    for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
        with (col1 if i % 2 == 0 else col2):
            st.markdown(f"**{label}**: {', '.join(map(str, result[i]))}")

    # Kombinasi 4D
    if "LSTM" in metode:
        with st.spinner("🔢 Menghitung kombinasi 4D..."):
            kombinasi = kombinasi_4d(df, lokasi=selected_lokasi, top_n=10, min_conf=min_conf, power=power)
            if kombinasi:
                st.subheader("💡 Top Kombinasi 4D")
                for k, s in kombinasi:
                    st.markdown(f"`{k}` - ⚡️ Confidence: `{s:.4f}`")

    # Akurasi
    with st.spinner("📏 Evaluasi akurasi..."):
        uji_df = df.tail(jumlah_uji)
        benar, total = 0, 0
        akurasi_list = []
        digit_log = {"Ribuan": [], "Ratusan": [], "Puluhan": [], "Satuan": []}

        for i in range(len(uji_df)):
            sub = df.iloc[:-(len(uji_df)-i)]
            if len(sub) < 30:
                continue
            pred = (
                top6_markov(sub)[0] if metode == "Markov" else
                top6_markov_order2(sub) if metode == "Markov Order-2" else
                top6_markov_hybrid(sub) if metode == "Markov Gabungan" else
                top6_lstm(sub, lokasi=selected_lokasi) if metode == "LSTM AI" else
                top6_ensemble(sub, lokasi=selected_lokasi)
            )
            actual = f"{int(uji_df.iloc[i]['angka']):04d}"
            skor = 0
            for j, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                digit_log[label].append(1 if int(actual[j]) in pred[j] else 0)
                if int(actual[j]) in pred[j]:
                    skor += 1
            total += 4
            benar += skor
            akurasi_list.append(skor / 4 * 100)

        if total > 0:
            st.success(f"🎯 Akurasi Total: {benar / total * 100:.2f}%")
            st.line_chart(pd.DataFrame({"Akurasi (%)": akurasi_list}))
            heat_df = pd.DataFrame({k: [sum(v)/len(v)*100 if v else 0] for k, v in digit_log.items()})
            fig, ax = plt.subplots()
            sns.heatmap(heat_df, annot=True, cmap="YlOrBr", fmt=".1f", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("⚠️ Tidak cukup data untuk evaluasi akurasi.")
