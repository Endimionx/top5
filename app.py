
import streamlit as st
import pandas as pd
from markov_model import top5_markov
from ai_model import top5_lstm

st.title("🎰 Prediksi Togel 4 Digit - AI & Markov (Top-5 Alternatif)")

riwayat_input = st.text_area("📝 Masukkan data history togel (1 angka per baris):", height=200)
data_lines = [line.strip() for line in riwayat_input.split("\n") if line.strip().isdigit() and len(line.strip()) == 4]
df = pd.DataFrame({"angka": data_lines})

metode = st.selectbox("🧠 Pilih Metode Prediksi", ["Markov", "LSTM AI"])

if st.button("🔮 Prediksi"):
    if len(df) < 11:
        st.warning("❌ Minimal 11 data diperlukan untuk prediksi.")
    else:
        if metode == "Markov":
            hasil = top5_markov(df)
        else:
            hasil = top5_lstm(df)

        st.subheader("🎯 Top-5 Prediksi per Posisi:")
        for i, posisi in enumerate(["Digit 1 (Ribuan)", "Digit 2 (Ratusan)", "Digit 3 (Puluhan)", "Digit 4 (Satuan)"]):
            st.markdown(f"**{posisi}:** {', '.join(str(d) for d in hasil[i])}")
