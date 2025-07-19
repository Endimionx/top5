import streamlit as st
import pandas as pd
import requests
import os
import numpy as np
import time
import matplotlib.pyplot as plt

from markov_model import top6_markov, top6_markov_order2, top6_markov_hybrid
from ai_model import (
    top6_model,
    train_and_save_model,
    kombinasi_4d,
    evaluate_lstm_accuracy_all_digits,
    preprocess_data,
    find_best_window_size_with_model_true,
    build_lstm_model,
    build_transformer_model
)
from lokasi_list import lokasi_list

st.set_page_config(page_title="Prediksi Togel AI", layout="wide")

st.title("🔮 Prediksi 4D - AI & Markov")

DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

# ====== Inisialisasi session_state window_per_digit ======
for label in DIGIT_LABELS:
    key = f"win_{label}"
    if key not in st.session_state:
        st.session_state[key] = 7  # default value

# ======== Ambil Data API dan Input Manual ========
st.markdown("## 📥 Data Angka Masuk")

# ======== Sidebar Pengaturan ========
with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    selected_hari = st.selectbox("📅 Hari", ["harian", "kemarin", "2hari", "3hari"])
    putaran = st.number_input("🔁 Putaran", 10, 1000, 100)
    metode = st.selectbox("🧠 Metode", ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI", "Ensemble AI + Markov"])
    jumlah_uji = st.number_input("📊 Data Uji", 1, 200, 10)
    temperature = st.slider("🌡️ Temperature", 0.1, 2.0, 0.5, step=0.1)
    voting_mode = st.selectbox("⚖️ Kombinasi", ["product", "average"])
    power = st.slider("📈 Confidence Power", 0.5, 3.0, 1.5, 0.1)
    min_conf = st.slider("🔎 Min Confidence", 0.0001, 0.01, 0.0005, 0.0001, format="%.4f")
    use_transformer = st.checkbox("🤖 Gunakan Transformer")
    model_type = "transformer" if use_transformer else "lstm"
    mode_prediksi = st.selectbox("🎯 Mode Prediksi", ["confidence", "ranked", "hybrid"])

    st.markdown("### 🪟 Window Size per Digit")
    window_per_digit = {}
    for label in DIGIT_LABELS:
        window_per_digit[label] = st.slider(
            f"{label.upper()}", 3, 30, st.session_state[f"win_{label}"], key=f"win_{label}"
        )

# ======== Manajemen Model ========
# ======== Manajemen Model (khusus metode AI) ========
if metode in ["LSTM AI", "Ensemble AI + Markov"]:
    with st.expander("⚙️ Manajemen Model", expanded=True):
        lokasi_id = selected_lokasi.lower().strip().replace(" ", "_")
        digit_labels = ["ribuan", "ratusan", "puluhan", "satuan"]
        for label in digit_labels:
            model_path = f"saved_models/{lokasi_id}_{label}_{model_type}.h5"
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                if os.path.exists(model_path):
                    st.info(f"📂 Model {label.upper()} tersedia.")
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
            with st.spinner("🔄 Melatih semua model..."):
                train_and_save_model(df, selected_lokasi, window_dict=window_per_digit, model_type=model_type)
            st.success("✅ Model berhasil dilatih.")
# ======== Ambil Data API ========
if "angka_list" not in st.session_state:
    st.session_state.angka_list = []

col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🔄 Ambil Data dari API"):
        try:
            with st.spinner("🔄 Mengambil data..."):
                url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran={putaran}&format=json&urut=asc"
                headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
                data = requests.get(url, headers=headers).json()
                angka_api = [d["result"] for d in data["data"] if len(d["result"]) == 4 and d["result"].isdigit()]
                st.session_state.angka_list = angka_api
                st.success(f"{len(angka_api)} angka berhasil diambil.")
        except Exception as e:
            st.error(f"❌ Gagal ambil data: {e}")

with col2:
    st.caption("📌 Data angka akan digunakan untuk pelatihan dan prediksi")

with st.expander("✏️ Edit Data Angka Manual", expanded=True):
    riwayat_input = "\n".join(st.session_state.angka_list)
    riwayat_input = st.text_area("📝 1 angka per baris:", value=riwayat_input, height=300)
    st.session_state.angka_list = [x.strip() for x in riwayat_input.splitlines() if x.strip().isdigit() and len(x.strip()) == 4]
    df = pd.DataFrame({"angka": st.session_state.angka_list})

# ======== Tabs Utama ========
tab1, tab2 = st.tabs(["🔮 Prediksi & Evaluasi", "🪟 Cari Window Size"])

# ======== TAB 1 ========
with tab1:
    if st.button("🔮 Prediksi"):
        if len(df) < max(window_per_digit.values()) + 1:
            st.warning("❌ Data tidak cukup.")
        else:
            with st.spinner("⏳ Memproses..."):
                result, probs = None, None
                if metode == "Markov":
                    result, _ = top6_markov(df)
                elif metode == "Markov Order-2":
                    result = top6_markov_order2(df)
                elif metode == "Markov Gabungan":
                    result = top6_markov_hybrid(df)
                elif metode == "LSTM AI":
                    result, probs = top6_model(df, lokasi=selected_lokasi, model_type=model_type,  
                                               return_probs=True, temperature=temperature,  
                                               mode_prediksi=mode_prediksi, window_dict=window_per_digit)  
                elif metode == "Ensemble AI + Markov":
                    lstm_result, probs = top6_model(df, lokasi=selected_lokasi, model_type=model_type,  
                                                    return_probs=True, temperature=temperature,  
                                                    mode_prediksi=mode_prediksi, window_dict=window_per_digit)  
                    markov_result, _ = top6_markov(df)  
                    result = []  
                    for i in range(4):  
                        merged = lstm_result[i] + markov_result[i]  
                        freq = {x: merged.count(x) for x in set(merged)}  
                        top6 = sorted(freq.items(), key=lambda x: -x[1])[:6]  
                        result.append([x[0] for x in top6])

            if result:
                st.subheader("🎯 Hasil Prediksi Top 6")
                for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                    st.markdown(f"**{label}:** {', '.join(map(str, result[i]))}")

            if probs:
                st.subheader("📊 Confidence Bar")
                for i, label in enumerate(DIGIT_LABELS):
                    st.markdown(f"**{label.upper()}**")
                    dconf = pd.DataFrame({
                        "Digit": [str(d) for d in result[i]],
                        "Confidence": probs[i]
                    }).sort_values("Confidence", ascending=True)
                    st.bar_chart(dconf.set_index("Digit"))

            if metode in ["LSTM AI", "Ensemble AI + Markov"]:
                with st.spinner("🔢 Kombinasi 4D..."):
                    top_komb = kombinasi_4d(df, lokasi=selected_lokasi, model_type=model_type,
                                            top_n=10, min_conf=min_conf, power=power,
                                            mode=voting_mode, window_dict=window_per_digit,
                                            mode_prediksi=mode_prediksi)
                    st.subheader("💡 Kombinasi 4D Top")
                    for komb, score in top_komb:
                        st.markdown(f"`{komb}` - Confidence: `{score:.4f}`")

    st.subheader("📊 Evaluasi Akurasi")
    acc1, acc6, top1 = evaluate_lstm_accuracy_all_digits(
        df, selected_lokasi, model_type=model_type, window_size=window_per_digit
    )
    for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
        st.info(f"🎯 {label}: Top-1 = {acc1[i]:.2%}, Top-6 = {acc6[i]:.2%}")

# ======== TAB 2 ========
with tab2:
    min_ws = st.number_input("🔁 Min WS", 3, 10, 4)
    max_ws = st.number_input("🔁 Max WS", 4, 20, 12)

    if "ws_result_table" not in st.session_state:
        st.session_state.ws_result_table = None
    if "window_per_digit" not in st.session_state:
        st.session_state.window_per_digit = {}

    if st.button("🔍 Cari Window Size Terbaik"):
        with st.spinner("Mencari window size..."):
            ws_info = []
            progress = st.progress(0)
            for idx, label in enumerate(DIGIT_LABELS):
                try:
                    best_ws, top6 = find_best_window_size_with_model_true(
                        df, label, lokasi=selected_lokasi, model_type=model_type,
                        min_ws=min_ws, max_ws=max_ws, temperature=temperature, use_cv=False, cv_folds=2
                    )
                    st.session_state.window_per_digit[label] = best_ws
                    ws_info.append({
                        "Digit": label.upper(),
                        "Best WS": best_ws,
                        "Top6": ", ".join(map(str, top6)) if top6 else "-"
                    })
                except Exception as e:
                    ws_info.append({
                        "Digit": label.upper(),
                        "Best WS": "-",
                        "Top6": "-"
                    })
                    st.error(f"❌ Gagal {label.upper()} WS: {e}")
                progress.progress((idx + 1) / len(DIGIT_LABELS))
            st.session_state.ws_result_table = pd.DataFrame(ws_info)

    if st.session_state.ws_result_table is not None:
        st.subheader("✅ Hasil Window Size")
        st.dataframe(st.session_state.ws_result_table)

        try:
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.axis('off')
            tbl = ax.table(cellText=st.session_state.ws_result_table.values,
                           colLabels=st.session_state.ws_result_table.columns,
                           cellLoc='center',
                           loc='center')
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1, 1.5)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Gagal simpan gambar: {e}")
