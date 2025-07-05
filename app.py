import streamlit as st
import pandas as pd
import requests
import os
from markov_model import top6_markov, top6_markov_order2, top6_markov_hybrid
from ai_model import top6_lstm, train_and_save_lstm, model_exists, anti_top6_lstm, low6_lstm, kombinasi_4d
from lokasi_list import lokasi_list
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prediksi Togel AI", layout="wide")
st.markdown("<h4>Prediksi Togel 4D - AI & Markov</h4>", unsafe_allow_html=True)

hari_list = ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"]

selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
selected_hari = st.selectbox("📅 Pilih Hari", hari_list)
putaran = st.slider("🔁 Jumlah Putaran", 1, 1000, 10)
jumlah_uji = st.number_input("📊 Jumlah Data Uji Akurasi", min_value=1, max_value=1000, value=5, step=1)

angka_list = []
riwayat_input = ""
if selected_lokasi and selected_hari:
    try:
        with st.spinner("🔄 Mengambil data dari API..."):
            url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran={putaran}&showpasaran=yes&showtgl=yes&format=json&urut=asc"
            headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
            response = requests.get(url, headers=headers)
            data = response.json()
            angka_list = [item["result"] for item in data.get("data", []) if len(item["result"]) == 4 and item["result"].isdigit()]
            riwayat_input = "\n".join(angka_list)
            st.success(f"✅ {len(angka_list)} angka berhasil diambil.")
            with st.expander("📥 Lihat Data"):
                st.code(riwayat_input)
    except Exception as e:
        st.error(f"❌ Gagal ambil data API: {e}")

df = pd.DataFrame({"angka": angka_list})

metode = st.selectbox("🧠 Pilih Metode Prediksi", ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI"])

if metode == "LSTM AI":
    with st.expander("⚙️ LSTM AI - Manajemen Model"):
        tab1, tab2 = st.tabs(["🔧 Model", "📉 Grafik Pelatihan"])

        with tab1:
            model_path = f"saved_models/lstm_{selected_lokasi.lower().replace(' ', '_')}.h5"
            if st.button("📚 Latih & Simpan Model"):
                with st.spinner("🔄 Melatih model..."):
                    train_and_save_lstm(df, selected_lokasi)
                st.success("✅ Model berhasil dilatih dan disimpan.")

            if os.path.exists(model_path):
                st.success(f"📁 Model ditemukan: {model_path}")
                with open(model_path, "rb") as f:
                    st.download_button("⬇️ Download Model", f, file_name=os.path.basename(model_path))
                if st.button("🗑 Hapus Model"):
                    os.remove(model_path)
                    st.warning("🗑 Model berhasil dihapus.")
            else:
                uploaded_model = st.file_uploader("📤 Upload Model (.h5)", type=["h5"])
                if uploaded_model is not None:
                    with st.spinner("📦 Menyimpan model yang diunggah..."):
                        with open(model_path, "wb") as f:
                            f.write(uploaded_model.read())
                    st.success("✅ Model berhasil diunggah.")
                    st.experimental_rerun()

        with tab2:
            log_file = f"training_logs/history_{selected_lokasi.lower().replace(' ', '_')}.csv"
            if os.path.exists(log_file):
                st.subheader("📈 Riwayat Pelatihan")
                df_log = pd.read_csv(log_file)
                st.line_chart(df_log[["loss", "output_0_accuracy", "output_1_accuracy", "output_2_accuracy", "output_3_accuracy"]])
                st.caption("output_0 = ribuan, output_1 = ratusan, output_2 = puluhan, output_3 = satuan")

if st.button("🔮 Prediksi"):
    if len(df) < 11:
        st.warning("❌ Minimal 11 data diperlukan.")
    else:
        with st.spinner("⏳ Melakukan prediksi..."):
            result = (
                top6_markov(df) if metode == "Markov" else
                top6_markov_order2(df) if metode == "Markov Order-2" else
                top6_markov_hybrid(df) if metode == "Markov Gabungan" else
                top6_lstm(df, lokasi=selected_lokasi)
            )

        if result is None:
            st.error("❌ Gagal prediksi.")
        else:
            st.markdown("### 🎯 Prediksi Top 6 Digit per Posisi")
            for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                st.markdown(f"**{label}:** {', '.join(str(d) for d in result[i])}")

            # Prediksi Kombinasi 4D
            if metode == "LSTM AI":
                with st.spinner("🔢 Menghitung kombinasi 4D..."):
                    st.markdown("### 🔢 Top 10 Kombinasi 4D Berdasarkan Confidence")
                    top_komb = kombinasi_4d(df, lokasi=selected_lokasi, top_n=10)
                    if top_komb:
                        df_komb = pd.DataFrame(top_komb, columns=["Kombinasi", "Confidence"])
                        st.dataframe(df_komb.style.format({"Confidence": "{:.5f}"}), use_container_width=True)

            # Akurasi
            with st.spinner("📏 Menghitung akurasi..."):
                list_akurasi = []
                uji_df = df.tail(min(jumlah_uji, len(df)))
                total = benar = 0
                for i in range(len(uji_df)):
                    subset_df = df.iloc[:-(len(uji_df) - i)]
                    if len(subset_df) < 11:
                        continue
                    pred_uji = (
                        top6_markov(subset_df) if metode == "Markov" else
                        top6_markov_order2(subset_df) if metode == "Markov Order-2" else
                        top6_markov_hybrid(subset_df) if metode == "Markov Gabungan" else
                        top6_lstm(subset_df, lokasi=selected_lokasi)
                    )
                    if pred_uji is None:
                        continue
                    actual = f"{int(uji_df.iloc[i]['angka']):04d}"
                    skor = sum(int(actual[j]) in pred_uji[j] for j in range(4))
                    total += 4
                    benar += skor
                    list_akurasi.append(skor / 4 * 100)

                if total > 0:
                    akurasi_total = (benar / total) * 100
                    st.info(f"📈 Akurasi {metode}: {akurasi_total:.2f}%")
                    with st.expander("📊 Grafik Akurasi"):
                        st.line_chart(pd.DataFrame({"Akurasi (%)": list_akurasi}))
                else:
                    st.warning("⚠️ Tidak cukup data valid untuk evaluasi akurasi.")
