import streamlit as st
import pandas as pd
import requests
import os
from markov_model import top6_markov, top6_markov_order2, top6_markov_hybrid
from ai_model import top6_lstm, train_and_save_lstm, kombinasi_4d, top6_ensemble, model_exists
from lokasi_list import lokasi_list
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Prediksi Togel AI", layout="wide", initial_sidebar_state="expanded")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_predict = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_kkflmtur.json")
st_lottie(lottie_predict, speed=1, height=150, key="prediksi")

st.title("🔮 Prediksi 4D - AI & Markov")

hari_list = ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"]
metode_list = ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI", "Ensemble AI + Markov"]

with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    selected_hari = st.selectbox("📅 Pilih Hari", hari_list)
    putaran = st.slider("🔁 Jumlah Putaran", 1, 1000, 100)
    jumlah_uji = st.number_input("📊 Jumlah Data Uji Akurasi", min_value=1, max_value=1000, value=5)
    metode = st.selectbox("🧠 Pilih Metode Prediksi", metode_list)

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

# LSTM Model Management
if metode == "LSTM AI":
    with st.expander("⚙️ LSTM AI - Manajemen Model"):
        model_path = f"saved_models/lstm_{selected_lokasi.lower().replace(' ', '_')}.h5"
        if model_exists(selected_lokasi):
            st.info(f"📁 Model tersedia: `{model_path}`")
        else:
            uploaded = st.file_uploader("📤 Upload Model (.h5)", type=["h5"])
            if uploaded:
                os.makedirs("saved_models", exist_ok=True)
                with open(model_path, "wb") as f:
                    f.write(uploaded.read())
                st.success("✅ Model berhasil diupload.")

        if st.button("📚 Latih & Simpan Model"):
            with st.spinner("🔄 Melatih model..."):
                train_and_save_lstm(df, selected_lokasi)
            st.success("✅ Model berhasil dilatih dan disimpan.")

        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                st.download_button("⬇️ Download Model", f, file_name=os.path.basename(model_path))
            if st.button("🗑 Hapus Model"):
                os.remove(model_path)
                st.warning("🗑 Model berhasil dihapus.")

# Prediksi
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
            st.error("❌ Gagal prediksi.")
        else:
            st.subheader("🎯 Prediksi Top 6 Digit per Posisi")
            col1, col2 = st.columns(2)
            for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                with (col1 if i % 2 == 0 else col2):
                    st.markdown(f"**{label}:** {', '.join(str(d) for d in result[i])}")

            if metode == "Markov" and isinstance(info, dict):
                with st.expander("🔥 Kombinasi 4D Terpopuler (Markov)"):
                    kombinasi_populer = info.get("kombinasi_populer", [])
                    if kombinasi_populer:
                        gabung_populer = " * ".join([row[0] for row in kombinasi_populer])
                        st.code(gabung_populer, language="text")

            if metode in ["LSTM AI", "Ensemble AI + Markov"]:
                with st.spinner("🔢 Menghitung kombinasi 4D..."):
                    top_komb = kombinasi_4d(df, lokasi=selected_lokasi, top_n=10)
                    if top_komb:
                        with st.expander("🔢 Top 10 Kombinasi 4D (AI)"):
                            gabungan = " * ".join([row[0] for row in top_komb])
                            st.code(gabungan, language="text")

        # Akurasi
        with st.spinner("📏 Menghitung akurasi..."):
            uji_df = df.tail(min(jumlah_uji, len(df)))
            total = benar = 0
            list_akurasi = []
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
                if pred is None or len(pred) != 4:
                    continue
                actual = f"{int(uji_df.iloc[i]['angka']):04d}"
                skor = sum(int(actual[j]) in pred[j] for j in range(4))
                total += 4
                benar += skor
                list_akurasi.append(skor / 4 * 100)
            if total > 0:
                st.success(f"📈 Akurasi {metode}: {benar / total * 100:.2f}%")
                with st.expander("📊 Grafik Akurasi"):
                    st.line_chart(pd.DataFrame({"Akurasi (%)": list_akurasi}))
            else:
                st.warning("⚠️ Tidak cukup data untuk evaluasi akurasi.")
