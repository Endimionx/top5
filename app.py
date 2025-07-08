import streamlit as st
import pandas as pd
import requests
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from markov_model import (
    top6_markov,
    top6_markov_order2,
    top6_markov_hybrid,
    kombinasi_4d_markov_hybrid
)
from ai_model import (
    top6_lstm,
    kombinasi_4d,
    top6_ensemble,
    preprocess_data,
    build_model,
    model_exists
)
from lokasi_list import lokasi_list
from user_manual import tampilkan_user_manual
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping


# ============================ UI SETUP ============================
st.set_page_config(page_title="🎯 Prediksi Togel AI", layout="wide")
tampilkan_user_manual()

# ============================ SIDEBAR ============================
with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    selected_hari = st.selectbox("📅 Pilih Hari", ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"])
    metode = st.selectbox("🧠 Metode Prediksi", ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI", "Ensemble AI + Markov"])
    jumlah_uji = st.number_input("📊 Jumlah Uji Akurasi", 3, 100, 7)
    cari_otomatis = st.toggle("🔍 Cari Putaran Otomatis", value=False)

    if cari_otomatis:
        max_putaran = st.number_input("🧮 Max Putaran", 50, 1000, 200)

    digit_weight_input = [1.0] * 4
    if metode == "Markov Gabungan":
        st.markdown("🎯 **Bobot Confidence Tiap Digit**")
        digit_weight_input = [
            st.slider("📌 Ribuan", 0.1, 3.0, 1.0, 0.1),
            st.slider("📌 Ratusan", 0.1, 3.0, 1.0, 0.1),
            st.slider("📌 Puluhan", 0.1, 3.0, 1.0, 0.1),
            st.slider("📌 Satuan", 0.1, 3.0, 1.0, 0.1),
        ]

# ============================ DATA FETCH ============================
putaran = 100
df_all = pd.DataFrame()
if selected_lokasi and selected_hari:
    try:
        with st.spinner("📥 Mengambil data awal..."):
            url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran=1000&format=json&urut=asc"
            headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
            response = requests.get(url, headers=headers)
            data = response.json().get("data", [])
            angka_list_all = [d["result"] for d in data if len(d["result"]) == 4 and d["result"].isdigit()]
            df_all = pd.DataFrame({"angka": angka_list_all})
    except Exception as e:
        st.error(f"❌ Gagal ambil data: {e}")

    if cari_otomatis and not df_all.empty:
        from app_utils import cari_putaran_terbaik
        with st.spinner("🔍 Menganalisis putaran terbaik..."):
            best_n, best_score, _ = cari_putaran_terbaik(
                df_all,
                lokasi=selected_lokasi,
                metode=metode,
                jumlah_uji=jumlah_uji,
                max_putaran=max_putaran,
                digit_weights=digit_weight_input if metode == "Markov Gabungan" else None
            )
        if best_n > 0:
            putaran = best_n
            st.success(f"✅ Putaran terbaik: {best_n} (Akurasi: {best_score:.2f}%)")
        else:
            st.warning("⚠️ Gagal menemukan putaran terbaik.")
    else:
        putaran = st.number_input("🔁 Jumlah Putaran", 20, 1000, 100)

df = df_all.tail(putaran).reset_index(drop=True) if not df_all.empty else pd.DataFrame()

# ============================ TAB AREA ============================
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediksi", "📊 Heatmap", "📈 Grafik Akurasi", "🧠 Manajemen Model"])

# ========== TAB 1: PREDIKSI ==========
with tab1:
    st.subheader("🔮 Hasil Prediksi Top 6 Digit")
    if st.button("🔁 Jalankan Prediksi"):
        if len(df) < 30:
            st.warning("❌ Minimal 30 data diperlukan.")
        else:
            with st.spinner("⏳ Memproses..."):
                result = (
                    top6_markov(df)[0] if metode == "Markov" else
                    top6_markov_order2(df) if metode == "Markov Order-2" else
                    top6_markov_hybrid(df, digit_weights=digit_weight_input) if metode == "Markov Gabungan" else
                    top6_lstm(df, lokasi=selected_lokasi) if metode == "LSTM AI" else
                    top6_ensemble(df, lokasi=selected_lokasi)
                )
            if result:
                col1, col2 = st.columns(2)
                for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                    with (col1 if i % 2 == 0 else col2):
                        st.success(f"**{label}:** {', '.join(map(str, result[i]))}")

                # Kombinasi 4D
                with st.expander("💡 Kombinasi 4D Tertinggi"):
                    top_komb = (
                        kombinasi_4d(df, lokasi=selected_lokasi, top_n=10) if metode == "LSTM AI" else
                        kombinasi_4d_markov_hybrid(df, top_n=10, digit_weights={
                            "ribuan": digit_weight_input[0],
                            "ratusan": digit_weight_input[1],
                            "puluhan": digit_weight_input[2],
                            "satuan": digit_weight_input[3],
                        }) if metode == "Markov Gabungan" else None
                    )
                    if top_komb:
                        for komb, score in top_komb:
                            st.markdown(f"**{komb}** — ⚡ Confidence: `{score:.4f}`")

# ========== TAB 2: HEATMAP ==========
with tab2:
    st.subheader("🌡️ Heatmap Akurasi Digit")
    if not df.empty:
        heatmap_data = pd.DataFrame([
            len(set([angka[i] for angka in df['angka']])) / len(df)
            for i in range(4)
        ], index=["Ribuan", "Ratusan", "Puluhan", "Satuan"], columns=["Akurasi"])
        fig, ax = plt.subplots()
        sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
        st.pyplot(fig)

# ========== TAB 3: GRAFIK AKURASI ==========
with tab3:
    st.subheader("📈 Grafik Akurasi terhadap Jumlah Putaran")
    if not df.empty:
        steps = list(range(30, min(len(df), 300), 10))
        akurasi = []
        for n in steps:
            subset = df.tail(n).reset_index(drop=True)
            acc_total, acc_benar = 0, 0
            for i in range(min(jumlah_uji, len(subset) - 30)):
                train_df = subset.iloc[:-(jumlah_uji - i)]
                if len(train_df) < 30: continue
                pred = (
                    top6_lstm(train_df, lokasi=selected_lokasi) if metode == "LSTM AI" else
                    top6_markov(train_df)[0] if metode == "Markov" else
                    top6_markov_order2(train_df) if metode == "Markov Order-2" else
                    top6_markov_hybrid(train_df, digit_weights=digit_weight_input) if metode == "Markov Gabungan" else
                    top6_ensemble(train_df, lokasi=selected_lokasi)
                )
                actual = f"{int(subset.iloc[-(jumlah_uji - i)]['angka']):04d}"
                acc = sum(int(actual[j]) in pred[j] for j in range(4))
                acc_benar += acc
                acc_total += 4
            akurasi.append(acc_benar / acc_total * 100 if acc_total else 0)
        df_chart = pd.DataFrame({"Putaran": steps, "Akurasi": akurasi})
        st.line_chart(df_chart.set_index("Putaran"))

# ========== TAB 4: MANAJEMEN MODEL ==========
with tab4:
    st.subheader("🧠 Manajemen Model LSTM per Digit")
    if metode == "LSTM AI" and not df.empty:
        for i, digit in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
            model_path = f"saved_models/{selected_lokasi.lower().replace(' ', '_')}_digit{i}.h5"
            st.markdown(f"### 🔢 Digit {digit}")
            if os.path.exists(model_path):
                st.success("✅ Model tersedia")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"🔁 Latih Ulang {digit}"):
                        X, y_all = preprocess_data(df)
                        y = y_all[i]
                        model = build_model(input_len=X.shape[1])
                        model.fit(X, y, epochs=50, batch_size=16, validation_split=0.2)
                        model.save(model_path)
                        st.success(f"✅ {digit} dilatih ulang.")
                with col2:
                    if st.button(f"🗑️ Hapus Model {digit}"):
                        os.remove(model_path)
                        st.warning(f"🧹 Model {digit} dihapus.")
            else:
                if st.button(f"📈 Latih Model {digit}"):
                    X, y_all = preprocess_data(df)
                    y = y_all[i]
                    model = build_model(input_len=X.shape[1])
                    model.fit(X, y, epochs=50, batch_size=16, validation_split=0.2)
                    model.save(model_path)
                    st.success(f"✅ Model {digit} dilatih.")
