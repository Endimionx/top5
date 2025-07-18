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
    model_exists,
    evaluate_lstm_accuracy_all_digits
)
from lokasi_list import lokasi_list
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Prediksi Togel AI", layout="wide")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

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
    putaran = st.number_input("🔁 Jumlah Putaran", min_value=1, max_value=1000, value=100)
    jumlah_uji = st.number_input("📊 Data Uji Akurasi", min_value=1, max_value=200, value=10)
    metode = st.selectbox("🧠 Metode Prediksi", metode_list)

    min_conf = 0.005
    power = 1.5
    temperature = 0.5
    voting_mode = "product"
    model_type = "lstm"
    window_size = 7

    if metode in ["LSTM AI", "Ensemble AI + Markov"]:
        window_size = st.slider("🪟 Window Size", 3, 30, 7)
        min_conf = st.slider("🔎 Minimum Confidence", 0.0001, 0.01, 0.0005, step=0.0001, format="%.4f")
        power = st.slider("📈 Confidence Power", 0.5, 3.0, 1.5, step=0.1)
        temperature = st.slider("🌡️ Temperature Scaling", 0.1, 2.0, 0.5, step=0.1)
        voting_mode = st.selectbox("⚖️ Kombinasi Mode", ["product", "average"])
        use_transformer = st.checkbox("🧠 Gunakan Transformer")
        model_type = "transformer" if use_transformer else "lstm"
        mode_prediksi = st.selectbox("🎯 Mode Prediksi Top6", ["confidence", "ranked", "hybrid"])

# Ambil & Edit Data (persisten dengan session_state)
if "angka_list" not in st.session_state:
    st.session_state.angka_list = []

if st.button("🔄 Ambil Data dari API"):
    try:
        with st.spinner("🔄 Mengambil data dari API..."):
            url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran={putaran}&format=json&urut=asc"
            headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
            response = requests.get(url, headers=headers)
            data = response.json()
            angka_api = [item["result"] for item in data.get("data", []) if len(item["result"]) == 4 and item["result"].isdigit()]
            st.session_state.angka_list = angka_api
            st.success(f"✅ {len(angka_api)} angka berhasil diambil.")
    except Exception as e:
        st.error(f"❌ Gagal ambil data API: {e}")

riwayat_input = "\n".join(st.session_state.angka_list)
riwayat_input = st.text_area("📝 Edit Data Angka Manual (1 per baris):", value=riwayat_input, height=300)
st.session_state.angka_list = [x.strip() for x in riwayat_input.splitlines() if x.strip().isdigit() and len(x.strip()) == 4]
df = pd.DataFrame({"angka": st.session_state.angka_list})

# Manajemen Model
if metode == "LSTM AI":
    with st.expander("⚙️ Manajemen Model"):
        lokasi_id = selected_lokasi.lower().strip().replace(" ", "_")
        digit_labels = ["ribuan", "ratusan", "puluhan", "satuan"]
        for i, label in enumerate(digit_labels):
            model_path = f"saved_models/{lokasi_id}_{label}_{model_type}.h5"
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                if os.path.exists(model_path):
                    st.info(f"📂 Model {label.upper()} tersedia ({model_type}).")
                else:
                    st.warning(f"⚠️ Model {label.upper()} belum tersedia.")
            with col2:
                if os.path.exists(model_path):
                    if st.button(f"🗑 Hapus {label.upper()}", key=f"hapus_model_{label}"):
                        os.remove(model_path)
                        st.warning(f"✅ Model {label.upper()} dihapus.")
            with col3:
                log_path = f"training_logs/history_{lokasi_id}_{label}_{model_type}.csv"
                if os.path.exists(log_path):
                    if st.button(f"🧹 Hapus Log {label.upper()}", key=f"hapus_log_{label}"):
                        os.remove(log_path)
                        st.info(f"🧾 Log training {label.upper()} dihapus.")
        if st.button("📚 Latih & Simpan Semua Model"):
            with st.spinner(f"🔄 Melatih semua model per digit ({model_type})..."):
                train_and_save_model(df, selected_lokasi, model_type=model_type, window_size=window_size)
            st.success("✅ Semua model berhasil dilatih dan disimpan.")

# Tombol Prediksi
if st.button("🔮 Prediksi"):
    if len(df) < window_size + 1:
        st.warning(f"❌ Minimal {window_size + 1} data diperlukan.")
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
                pred = top6_model(df, lokasi=selected_lokasi, model_type=model_type, return_probs=True,
                                  temperature=temperature, mode_prediksi=mode_prediksi, window_size=window_size)
                if pred: result, probs = pred
            elif metode == "Ensemble AI + Markov":
                pred = top6_model(df, lokasi=selected_lokasi, model_type=model_type, return_probs=True,
                                  temperature=temperature, mode_prediksi=mode_prediksi, window_size=window_size)
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

        digit_labels = ["Ribuan", "Ratusan", "Puluhan", "Satuan"]
        if result is None:
            st.error("❌ Gagal melakukan prediksi.")
        else:
            with st.expander("🎯 Hasil Prediksi Top 6 Digit"):
                col1, col2 = st.columns(2)
                for i, label in enumerate(digit_labels):
                    col = col1 if i < 2 else col2
                    with col:
                        st.markdown(f"**{label}:** {', '.join(map(str, result[i]))}")

            if metode in ["LSTM AI", "Ensemble AI + Markov"] and probs:
                with st.expander("📊 Confidence Bar per Digit"):
                    for i, label in enumerate(digit_labels):
                        st.markdown(f"**🔢 {label}**")
                        digit_data = pd.DataFrame({
                            "Digit": [str(d) for d in result[i]],
                            "Confidence": probs[i]
                        }).sort_values(by="Confidence", ascending=True)
                        st.bar_chart(digit_data.set_index("Digit"))

            if metode in ["LSTM AI", "Ensemble AI + Markov"]:
                with st.spinner("🔢 Menghitung kombinasi 4D terbaik..."):
                    top_komb = kombinasi_4d(df, lokasi=selected_lokasi, model_type=model_type,
                                            top_n=10, min_conf=min_conf, power=power,
                                            mode=voting_mode, window_size=window_size, mode_prediksi=mode_prediksi)
                    if top_komb:
                        with st.expander("💡 Simulasi Kombinasi 4D Terbaik"):
                            sim_col = st.columns(2)
                            for i, (komb, score) in enumerate(top_komb):
                                with sim_col[i % 2]:
                                    st.markdown(f"`{komb}` - ⚡️ Confidence: `{score:.4f}`")

        with st.expander("📊 Evaluasi Akurasi LSTM per Digit"):
            with st.spinner("🔄 Mengevaluasi akurasi model LSTM..."):
                acc_top1_list, acc_top6_list, top1_labels_list = evaluate_lstm_accuracy_all_digits(
                    df, selected_lokasi, model_type=model_type, window_size=window_size
                )
                if acc_top1_list is not None:
                    for i in range(4):
                        label = digit_labels[i]
                        top1_digit = top1_labels_list[i] if top1_labels_list and i < len(top1_labels_list) else "-"
                        st.info(f"🎯 {label} (Digit {i+1})\nTop-1 ({top1_digit}) Accuracy: {acc_top1_list[i]:.2%}, Top-6 Accuracy: {acc_top6_list[i]:.2%}")
                else:
                    st.warning("⚠️ Tidak bisa mengevaluasi akurasi. Model belum tersedia atau data tidak cukup.")
