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

def cari_putaran_terbaik(df_all, lokasi, metode, jumlah_uji=10):
    best_score, best_n, hasil_all = 0, 0, {}
    for n in range(30, min(len(df_all), 200)):
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
                    top6_markov_hybrid(train_df) if metode == "Markov Gabungan" else
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

lottie_predict = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_kkflmtur.json")
st_lottie(lottie_predict, speed=1, height=150, key="prediksi")

st.title("🔮 Prediksi 4D - AI & Markov")

# Sidebar
hari_list = ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"]
metode_list = ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI", "Ensemble AI + Markov"]

with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    selected_hari = st.selectbox("📅 Pilih Hari", hari_list)
    jumlah_uji = st.number_input("📊 Data Uji Akurasi", min_value=5, max_value=100, value=10)
    metode = st.selectbox("🧠 Metode Prediksi", metode_list)

    cari_otomatis = st.toggle("🔍 Cari Putaran Otomatis", value=False)
    putaran = 100
    best_score = None
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
            best_n, best_score, _ = cari_putaran_terbaik(df_all, selected_lokasi, metode, jumlah_uji)
        if best_n > 0:
            putaran = best_n
            st.success(f"✅ Putaran terbaik: {best_n} (Akurasi: {best_score:.2f}%)")
        else:
            st.warning("⚠️ Gagal menemukan putaran terbaik.")
    elif not cari_otomatis:
        putaran = st.slider("🔁 Jumlah Putaran", 20, 1000, 100, step=10)

    min_conf = 0.0005
    power = 1.5
    if metode in ["LSTM AI", "Ensemble AI + Markov"]:
        min_conf = st.slider("🔎 Minimum Confidence", 0.0001, 0.001, 0.0005, step=0.0001, format="%.4f")
        power = st.slider("📈 Confidence Weight Power", 0.5, 3.0, 1.5, step=0.1)

# Ambil data sesuai putaran hasil analisis
angka_list, riwayat_input = [], ""
df = pd.DataFrame()
if selected_lokasi and selected_hari:
    try:
        with st.spinner("📦 Mengambil data berdasarkan putaran..."):
            url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran={putaran}&format=json&urut=asc"
            headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
            response = requests.get(url, headers=headers)
            data = response.json()
            angka_list = [item["result"] for item in data.get("data", []) if len(item["result"]) == 4 and item["result"].isdigit()]
            df = pd.DataFrame({"angka": angka_list})
            riwayat_input = "\n".join(angka_list)
            st.success(f"✅ {len(angka_list)} angka berhasil diambil.")
            with st.expander("📥 Lihat Data"):
                st.code(riwayat_input, language="text")
    except Exception as e:
        st.error(f"❌ Gagal ambil data API: {e}")

# Manajemen Model
if metode == "LSTM AI":
    with st.expander("⚙️ Manajemen Model LSTM"):
        for i in range(4):
            model_path = f"saved_models/{selected_lokasi.lower().replace(' ', '_')}_digit{i}.h5"
            col1, col2 = st.columns([2, 1])
            with col1:
                if os.path.exists(model_path):
                    st.info(f"📂 Model Digit-{i} tersedia.")
                else:
                    st.warning(f"⚠️ Model Digit-{i} belum tersedia.")
            with col2:
                if os.path.exists(model_path):
                    if st.button(f"🗑 Hapus Digit-{i}", key=f"hapus_digit_{i}"):
                        os.remove(model_path)
                        st.warning(f"✅ Model Digit-{i} dihapus.")

        if st.button("📚 Latih & Simpan Semua Model"):
            with st.spinner("🔄 Melatih model per digit..."):
                train_and_save_lstm(df, selected_lokasi)
            st.success("✅ Semua model berhasil dilatih dan disimpan.")

# Prediksi
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
            st.subheader("🎯 Hasil Prediksi")
            col1, col2 = st.columns(2)
            for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                with (col1 if i % 2 == 0 else col2):
                    st.markdown(f"**{label}:** {', '.join(map(str, result[i]))}")

            if metode in ["LSTM AI", "Ensemble AI + Markov"]:
                with st.spinner("🔢 Kombinasi 4D terbaik..."):
                    top_komb = kombinasi_4d(df, lokasi=selected_lokasi, top_n=10, min_conf=min_conf, power=power)
                    with st.expander("💡 Kombinasi 4D Confidence Tinggi"):
                        for komb, score in top_komb:
                            st.markdown(f"`{komb}` - ⚡️ {score:.4f}")

        # Akurasi
        with st.spinner("📏 Evaluasi akurasi..."):
            uji_df = df.tail(min(jumlah_uji, len(df)))
            total, benar = 0, 0
            digit_acc = {k: [] for k in ["Ribuan", "Ratusan", "Puluhan", "Satuan"]}
            akurasi_list = []

            for i in range(len(uji_df)):
                subset = df.iloc[:-(len(uji_df) - i)]
                if len(subset) < 30:
                    continue
                try:
                    pred = (
                        top6_markov(subset)[0] if metode == "Markov" else
                        top6_markov_order2(subset) if metode == "Markov Order-2" else
                        top6_markov_hybrid(subset) if metode == "Markov Gabungan" else
                        top6_lstm(subset, lokasi=selected_lokasi) if metode == "LSTM AI" else
                        top6_ensemble(subset, lokasi=selected_lokasi)
                    )
                    actual = f"{int(uji_df.iloc[i]['angka']):04d}"
                    skor = 0
                    for j, label in enumerate(digit_acc.keys()):
                        digit = int(actual[j])
                        if digit in pred[j]:
                            skor += 1
                            digit_acc[label].append(1)
                        else:
                            digit_acc[label].append(0)
                    total += 4
                    benar += skor
                    akurasi_list.append(skor / 4 * 100)
                except:
                    continue

            if total > 0:
                st.success(f"📈 Akurasi Total: {benar / total * 100:.2f}%")
                st.line_chart(pd.DataFrame({"Akurasi (%)": akurasi_list}))
                heat_df = pd.DataFrame({k: [sum(v)/len(v)*100 if v else 0] for k, v in digit_acc.items()})
                st.subheader("🔥 Heatmap Akurasi per Digit")
                fig, ax = plt.subplots()
                sns.heatmap(heat_df, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
                st.pyplot(fig)
            else:
                st.warning("⚠️ Tidak cukup data untuk evaluasi akurasi.")
