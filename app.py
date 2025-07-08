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
    model_exists,
)
from lokasi_list import lokasi_list
from user_manual import tampilkan_user_manual
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping


def load_training_history(path):
    return pd.read_csv(path)


def cari_putaran_terbaik(df_all, lokasi, metode, jumlah_uji=10, max_putaran=200, digit_weights=None):
    best_score, best_n, hasil_all = 0, 0, {}
    for n in range(30, min(len(df_all), max_putaran)):
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
                    top6_markov_hybrid(train_df, digit_weights=digit_weights) if metode == "Markov Gabungan" else
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


st.set_page_config(page_title="Prediksi Togel AI", layout="wide")
tampilkan_user_manual()

hari_list = ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"]
metode_list = ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI", "Ensemble AI + Markov"]

with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    selected_hari = st.selectbox("📅 Pilih Hari", hari_list)
    jumlah_uji = st.number_input("📊 Data Uji Akurasi", min_value=3, max_value=100, value=7)
    metode = st.selectbox("🧠 Metode Prediksi", metode_list)
    cari_otomatis = st.toggle("🔍 Cari Putaran Otomatis", value=False)

    if cari_otomatis:
        max_putaran = st.number_input("🧮 Max Putaran untuk Dicoba", min_value=50, max_value=1000, value=200)

    digit_weight_input = [1.0, 1.0, 1.0, 1.0]
    if metode == "Markov Gabungan":
        st.markdown("🎯 **Bobot Confidence Tiap Digit (Markov Gabungan)**")
        digit_weight_input = [
            st.slider("📌 Ribuan", 0.1, 3.0, 1.0, 0.1),
            st.slider("📌 Ratusan", 0.1, 3.0, 1.0, 0.1),
            st.slider("📌 Puluhan", 0.1, 3.0, 1.0, 0.1),
            st.slider("📌 Satuan", 0.1, 3.0, 1.0, 0.1)
        ]

putaran = 100
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
    elif not cari_otomatis:
        putaran = st.number_input("🔁 Jumlah Putaran", min_value=20, max_value=1000, value=100)

angka_list, df = [], pd.DataFrame()
try:
    if not df_all.empty:
        df = df_all.tail(putaran).reset_index(drop=True)
        angka_list = df["angka"].tolist()
    else:
        url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran={putaran}&format=json&urut=asc"
        headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
        response = requests.get(url, headers=headers)
        data = response.json()
        angka_list = [item["result"] for item in data.get("data", []) if len(item["result"]) == 4 and item["result"].isdigit()]
        df = pd.DataFrame({"angka": angka_list})
except Exception as e:
    st.error(f"❌ Gagal ambil data: {e}")

# 🔮 Prediksi
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
                result = top6_markov_hybrid(df, digit_weights=digit_weight_input)
            elif metode == "LSTM AI":
                result = top6_lstm(df, lokasi=selected_lokasi)
            elif metode == "Ensemble AI + Markov":
                result = top6_ensemble(df, lokasi=selected_lokasi)

        if result:
            with st.expander("🎯 Hasil Prediksi Top 6 Digit"):
                col1, col2 = st.columns(2)
                for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                    with (col1 if i % 2 == 0 else col2):
                        st.markdown(f"**{label}:** {', '.join(map(str, result[i]))}")

            if metode in ["LSTM AI", "Markov Gabungan"]:
                with st.spinner("🔢 Menghitung kombinasi 4D terbaik..."):
                    if metode == "LSTM AI":
                        top_komb = kombinasi_4d(result, mode="average")
                    else:
                        top_komb = kombinasi_4d_markov_hybrid(
                            df,
                            top_n=10,
                            mode="average",
                            digit_weights={
                                "ribuan": digit_weight_input[0],
                                "ratusan": digit_weight_input[1],
                                "puluhan": digit_weight_input[2],
                                "satuan": digit_weight_input[3],
                            }
                        )
                    if top_komb:
                        with st.expander("💡 Simulasi Kombinasi 4D"):
                            for komb, score in top_komb:
                                st.markdown(f"**{komb}** - ⚡ Confidence: `{score:.4f}`")

# 📈 Grafik Akurasi
if metode == "LSTM AI":
    with st.expander("📊 Grafik Riwayat Akurasi per Digit"):
        for i, digit in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
            log_path = f"training_logs/history_{selected_lokasi.lower().replace(' ', '_')}_digit{i}.csv"
            if os.path.exists(log_path):
                df_log = load_training_history(log_path)
                fig, ax = plt.subplots()
                sns.lineplot(data=df_log, x=df_log.index, y="accuracy", label="Akurasi", ax=ax)
                sns.lineplot(data=df_log, x=df_log.index, y="val_accuracy", label="Val Akurasi", ax=ax)
                ax.set_title(f"📈 Akurasi Digit {digit}")
                st.pyplot(fig)

# 🌡️ Heatmap Akurasi
if metode in ["LSTM AI", "Markov Gabungan"] and not df.empty:
    with st.expander("🌡️ Heatmap Akurasi per Digit"):
        heatmap_data = pd.DataFrame([len(set(col)) / len(col) for col in zip(*df["angka"])],
                                    index=["Ribuan", "Ratusan", "Puluhan", "Satuan"],
                                    columns=["Akurasi"])
        fig, ax = plt.subplots()
        sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
        st.pyplot(fig)

# 🧠 Manajemen Model LSTM
if metode == "LSTM AI" and not df.empty:
    with st.expander("🧠 Manajemen Model LSTM per Digit"):
        st.markdown("Kelola model LSTM secara terpisah untuk tiap digit:")
        for i, digit in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
            model_path = f"saved_models/{selected_lokasi.lower().replace(' ', '_')}_digit{i}.h5"
            st.markdown(f"### 🔢 Digit {digit}")
            if os.path.exists(model_path):
                st.success("✅ Model tersedia")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"🔁 Latih Ulang {digit}"):
                        with st.spinner(f"Melatih ulang model digit {digit}..."):
                            X, y_all = preprocess_data(df)
                            model = build_model(input_len=X.shape[1])
                            y = y_all[i]
                            callbacks = [
                                CSVLogger(f"training_logs/history_{selected_lokasi.lower().replace(' ', '_')}_digit{i}.csv"),
                                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                            ]
                            model.fit(X, y, epochs=50, batch_size=16, verbose=0, validation_split=0.2, callbacks=callbacks)
                            model.save(model_path)
                            st.success(f"✅ Model digit {digit} berhasil dilatih ulang.")
                with col2:
                    if st.button(f"🗑️ Hapus Model {digit}"):
                        os.remove(model_path)
                        st.warning(f"🧹 Model digit {digit} telah dihapus.")
            else:
                st.error("❌ Model belum tersedia")
                if st.button(f"📈 Latih Model {digit}"):
                    with st.spinner(f"Melatih model digit {digit}..."):
                        X, y_all = preprocess_data(df)
                        model = build_model(input_len=X.shape[1])
                        y = y_all[i]
                        callbacks = [
                            CSVLogger(f"training_logs/history_{selected_lokasi.lower().replace(' ', '_')}_digit{i}.csv"),
                            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                        ]
                        model.fit(X, y, epochs=50, batch_size=16, verbose=0, validation_split=0.2, callbacks=callbacks)
                        model.save(model_path)
                        st.success(f"✅ Model digit {digit} berhasil dilatih.")
