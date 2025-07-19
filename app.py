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
    evaluate_lstm_accuracy_all_digits,
    preprocess_data,
    find_best_window_size_with_model,
    find_best_window_size_with_model_fast,
    find_best_window_size_with_model_true,
    build_lstm_model,
    build_transformer_model
)
from lokasi_list import lokasi_list
from streamlit_lottie import st_lottie
import time

st.set_page_config(page_title="Prediksi Togel AI", layout="wide")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_predict = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_kkflmtur.json")
st_lottie(lottie_predict, speed=1, height=150, key="prediksi")

st.title("🔮 Prediksi 4D - AI & Markov")

DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

# Sidebar
with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    hari_list = ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"]
    selected_hari = st.selectbox("📅 Pilih Hari", hari_list)
    metode_list = ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI", "Ensemble AI + Markov"]
    metode = st.selectbox("🧠 Metode Prediksi", metode_list)
    putaran = st.number_input("🔁 Jumlah Putaran", min_value=1, max_value=1000, value=100)
    jumlah_uji = st.number_input("📊 Data Uji Akurasi", min_value=1, max_value=200, value=10)

# Init default state
if "angka_list" not in st.session_state:
    st.session_state.angka_list = []
if "ws_result_table" not in st.session_state:
    st.session_state.ws_result_table = None
if "window_per_digit" not in st.session_state:
    st.session_state.window_per_digit = {}

# Tabs utama
tab1, tab2 = st.tabs(["📊 Prediksi & Model", "🪟 Pencarian Window Size"])

with tab1:
    # Ambil & Edit Data
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

    # Parameter lanjutan
    min_conf = 0.005
    power = 1.5
    temperature = 0.5
    voting_mode = "product"
    model_type = "lstm"
    window_size = 7
    window_per_digit = {}

    if metode in ["LSTM AI", "Ensemble AI + Markov"]:
        st.markdown("### 🪟 Window Size per Digit")
        for label in DIGIT_LABELS:
            window_per_digit[label] = st.slider(f"🪟 {label.upper()}", 3, 30, 7, key=f"win_{label}")
        min_conf = st.slider("🔎 Minimum Confidence", 0.0001, 0.01, 0.0005, step=0.0001, format="%.4f")
        power = st.slider("📈 Confidence Power", 0.5, 3.0, 1.5, step=0.1)
        temperature = st.slider("🌡️ Temperature Scaling", 0.1, 2.0, 0.5, step=0.1)
        voting_mode = st.selectbox("⚖️ Kombinasi Mode", ["product", "average"])
        use_transformer = st.checkbox("🧠 Gunakan Transformer")
        model_type = "transformer" if use_transformer else "lstm"
        mode_prediksi = st.selectbox("🎯 Mode Prediksi Top6", ["confidence", "ranked", "hybrid"])

    # Manajemen Model
    if metode == "LSTM AI":
        with st.expander("⚙️ Manajemen Model"):
            lokasi_id = selected_lokasi.lower().strip().replace(" ", "_")
            for label in DIGIT_LABELS:
                model_path = f"saved_models/{lokasi_id}_{label}_{model_type}.h5"
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.info(f"📂 Model {label.upper()} tersedia.") if os.path.exists(model_path) else st.warning(f"⚠️ Model {label.upper()} belum tersedia.")
                with col2:
                    if os.path.exists(model_path) and st.button(f"🗑 Hapus {label.upper()}", key=f"hapus_model_{label}"):
                        os.remove(model_path)
                        st.warning(f"✅ Model {label.upper()} dihapus.")
                with col3:
                    log_path = f"training_logs/history_{lokasi_id}_{label}_{model_type}.csv"
                    if os.path.exists(log_path) and st.button(f"🧹 Hapus Log {label.upper()}", key=f"hapus_log_{label}"):
                        os.remove(log_path)
                        st.info(f"🧾 Log training {label.upper()} dihapus.")
            if st.button("📚 Latih & Simpan Semua Model"):
                with st.spinner("🔄 Melatih semua model..."):
                    train_and_save_model(df, selected_lokasi, window_dict=window_per_digit, model_type=model_type)
                st.success("✅ Model berhasil dilatih.")

    # Prediksi
    if st.button("🔮 Prediksi"):
        if len(df) < max(window_per_digit.values()) + 1:
            st.warning("❌ Jumlah data tidak mencukupi untuk prediksi.")
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
                                      temperature=temperature, mode_prediksi=mode_prediksi, window_dict=window_per_digit)
                    if pred: result, probs = pred
                elif metode == "Ensemble AI + Markov":
                    pred = top6_model(df, lokasi=selected_lokasi, model_type=model_type, return_probs=True,
                                      temperature=temperature, mode_prediksi=mode_prediksi, window_dict=window_per_digit)
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

            if result:
                with st.expander("🎯 Hasil Prediksi Top 6 Digit"):
                    col1, col2 = st.columns(2)
                    for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                        with col1 if i < 2 else col2:
                            st.markdown(f"**{label}:** {', '.join(map(str, result[i]))}")
                if probs:
                    with st.expander("📊 Confidence Bar per Digit"):
                        for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                            st.markdown(f"**🔢 {label}**")
                            df_conf = pd.DataFrame({"Digit": [str(d) for d in result[i]], "Confidence": probs[i]})
                            st.bar_chart(df_conf.set_index("Digit"))
                if metode in ["LSTM AI", "Ensemble AI + Markov"]:
                    with st.spinner("🔢 Menghitung kombinasi 4D terbaik..."):
                        top_komb = kombinasi_4d(df, lokasi=selected_lokasi, model_type=model_type,
                                                top_n=10, min_conf=min_conf, power=power,
                                                mode=voting_mode, window_dict=window_per_digit, mode_prediksi=mode_prediksi)
                        if top_komb:
                            with st.expander("💡 Simulasi Kombinasi 4D Terbaik"):
                                col_sim = st.columns(2)
                                for i, (komb, score) in enumerate(top_komb):
                                    with col_sim[i % 2]:
                                        st.markdown(f"`{komb}` - ⚡️ Confidence: `{score:.4f}`")
            with st.expander("📊 Evaluasi Akurasi LSTM per Digit"):
                with st.spinner("🔄 Mengevaluasi akurasi model..."):
                    acc1, acc6, top1 = evaluate_lstm_accuracy_all_digits(
                        df, selected_lokasi, model_type=model_type, window_size=window_per_digit
                    )
                    if acc1:
                        for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                            st.info(f"🎯 {label} Accuracy:\nTop-1: {acc1[i]:.2%}, Top-6: {acc6[i]:.2%}")
                    else:
                        st.warning("⚠️ Tidak bisa evaluasi. Model belum tersedia atau data kurang.")

with tab2:
    min_ws = st.number_input("🔁 Min Window Size", min_value=3, max_value=20, value=4)
    max_ws = st.number_input("🔁 Max Window Size", min_value=3, max_value=30, value=14)

    if st.button("🔍 Cari Window Size Terbaik"):
        with st.spinner("🔎 Mencari window size terbaik per digit..."):
            window_per_digit = {}
            ws_info_data = []
            progress_bar = st.progress(0)
            for idx, label in enumerate(DIGIT_LABELS):
                try:
                    st.markdown(f"#### 🔧 Proses: {label.upper()}")
                    time.sleep(0.1)
                    best_ws, top6_digits = find_best_window_size_with_model_true(
                        df,
                        label=label,
                        lokasi=selected_lokasi,
                        model_type=model_type,
                        min_ws=min_ws,
                        max_ws=max_ws,
                        temperature=temperature
                    )
                    window_per_digit[label] = best_ws
                    st.session_state[f"win_{label}"] = best_ws
                    ws_info_data.append({
                        "Digit": label.upper(),
                        "Best WS": best_ws,
                        "Top6": ", ".join(map(str, top6_digits)) if top6_digits else "-"
                    })
                except Exception as e:
                    st.error(f"❌ Gagal {label.upper()} WS: {e}")
                    window_per_digit[label] = None
                    ws_info_data.append({
                        "Digit": label.upper(),
                        "Best WS": "-",
                        "Top6": "-"
                    })
                progress_bar.progress((idx + 1) / len(DIGIT_LABELS))
                time.sleep(0.2)
            progress_bar.empty()
            st.session_state.window_per_digit = window_per_digit
            st.session_state.ws_result_table = pd.DataFrame(ws_info_data)

    if st.session_state.ws_result_table is not None:
        st.markdown("### ✅ Hasil Pencarian Window Size Terbaik")
        st.dataframe(st.session_state.ws_result_table, use_container_width=True)
        try:
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.axis('off')
            tbl = ax.table(cellText=st.session_state.ws_result_table.values,
                           colLabels=st.session_state.ws_result_table.columns,
                           cellLoc='center', loc='center')
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1, 1.5)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Gagal menyimpan sebagai gambar: {e}")
