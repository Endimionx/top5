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
    putaran = st.slider("🔁 Jumlah Putaran", 1, 1000, 100)
    jumlah_uji = st.number_input("📊 Data Uji Akurasi", min_value=1, max_value=200, value=10)
    metode = st.selectbox("🧠 Metode Prediksi", metode_list)

    min_conf = 0.005
    power = 1.5
    temperature = 0.5
    voting_mode = "product"
    model_type = "lstm"

    if metode in ["LSTM AI", "Ensemble AI + Markov"]:
        min_conf = st.slider("🔎 Minimum Confidence", 0.0001, 0.01, 0.0005, step=0.0001, format="%.4f")
        power = st.slider("📈 Confidence Power", 0.5, 3.0, 1.5, step=0.1)
        temperature = st.slider("🌡️ Temperature Scaling", 0.1, 2.0, 0.5, step=0.1)
        voting_mode = st.selectbox("⚖️ Kombinasi Mode", ["product", "average"])
        use_transformer = st.checkbox("🧠 Gunakan Transformer")
        model_type = "transformer" if use_transformer else "lstm"

# Ambil Data
angka_list = []
riwayat_input = ""
if selected_lokasi and selected_hari:
    try:
        with st.spinner("🔄 Mengambil data dari API..."):
            url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran={putaran}&format=json&urut=asc"
            headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
            response = requests.get(url, headers=headers)
            data = response.json()
            angka_list = [item["result"] for item in data.get("data", []) if len(item["result"]) == 4 and item["result"].isdigit()]
            riwayat_input = "\n".join(angka_list)
            st.success(f"✅ {len(angka_list)} angka berhasil diambil.")
            with st.expander("📥 Lihat Data"):
                st.code(riwayat_input, language="text")
    except Exception as e:
        st.error(f"❌ Gagal ambil data API: {e}")

df = pd.DataFrame({"angka": angka_list})

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
                train_and_save_model(df, selected_lokasi, model_type=model_type)
            st.success("✅ Semua model berhasil dilatih dan disimpan.")

# Evaluasi Otomatis Banyak Putaran
with st.expander("🧪 Evaluasi Putaran Terbaik"):
    enable_eval = st.checkbox("🔍 Uji Banyak Putaran", value=False)
    if enable_eval:
        start = st.number_input("Putaran Awal", 50, 5000, 50, step=50)
        end = st.number_input("Putaran Akhir", start + 50, 5000, 200, step=50)
        step = st.number_input("Step", 10, 500, 50)

        eval_hasil = []
        putaran_range = list(range(start, end + 1, step))

        with st.spinner("🧮 Menguji akurasi untuk berbagai jumlah putaran..."):
            for p in putaran_range:
                try:
                    url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran={p}&format=json&urut=asc"
                    headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
                    r = requests.get(url, headers=headers)
                    data = r.json()
                    angka_p = [item["result"] for item in data.get("data", []) if len(item["result"]) == 4 and item["result"].isdigit()]
                    dfp = pd.DataFrame({"angka": angka_p})
                    if len(dfp) < 30:
                        continue
                    uji_df = dfp.tail(min(jumlah_uji, len(dfp)))
                    total, benar = 0, 0
                    for i in range(len(uji_df)):
                        subset = dfp.iloc[:-(len(uji_df) - i)]
                        if len(subset) < 20:
                            continue
                        pred = (
                            top6_markov(subset)[0] if metode == "Markov" else
                            top6_markov_order2(subset) if metode == "Markov Order-2" else
                            top6_markov_hybrid(subset) if metode == "Markov Gabungan" else
                            top6_model(subset, lokasi=selected_lokasi, model_type=model_type) if metode == "LSTM AI" else
                            top6_ensemble(subset, lokasi=selected_lokasi, model_type=model_type)
                        )
                        if pred is None:
                            continue
                        if metode in ["Markov", "Markov Order-2", "Markov Gabungan"]:
                            pred[1], pred[2] = pred[2], pred[1]
                        actual = f"{int(uji_df.iloc[i]['angka']):04d}"
                        skor = sum([int(actual[j]) in pred[j] for j in range(4)])
                        total += 4
                        benar += skor
                    if total > 0:
                        akurasi = benar / total * 100
                        eval_hasil.append((p, akurasi))
                except:
                    continue

        if eval_hasil:
            df_hasil = pd.DataFrame(eval_hasil, columns=["Putaran", "Akurasi (%)"])
            st.line_chart(df_hasil.set_index("Putaran"))
            st.dataframe(df_hasil.style.format({"Akurasi (%)": "{:.2f}"}))
            best_row = max(eval_hasil, key=lambda x: x[1])
            st.success(f"🎯 Putaran terbaik: {best_row[0]} dengan akurasi {best_row[1]:.2f}%")
        else:
            st.warning("⚠️ Tidak ada hasil yang bisa dievaluasi.")
