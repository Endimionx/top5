import streamlit as st
import pandas as pd
from markov_model import top5_markov, top5_markov_order2, top5_markov_hybrid
from ai_model import top5_lstm

st.title("🎰 Prediksi Togel 4 Digit - AI & Markov (Top-5 Alternatif + Uji Akurasi)")

riwayat_input = st.text_area("🧾 Masukkan data history togel (1 angka per baris):", height=200)
data_lines = [line.strip() for line in riwayat_input.split("\n") if line.strip().isdigit() and len(line.strip()) == 4]
df = pd.DataFrame({"angka": data_lines})

angka_aktual = st.text_input("❓ Masukkan angka aktual (untuk uji akurasi, opsional):", "")
jumlah_uji = st.number_input("📊 Jumlah data uji terakhir:", min_value=1, max_value=500, value=5, step=1)

metode = st.selectbox("🧠 Pilih Metode Prediksi", ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI"])

if st.button("🔮 Prediksi"):
    if len(df) < 11:
        st.warning("❌ Minimal 11 data diperlukan untuk prediksi.")
    else:
        if metode == "Markov":
            hasil = top5_markov(df)
        elif metode == "Markov Order-2":
            hasil = top5_markov_order2(df)
        elif metode == "Markov Gabungan":
            hasil = top5_markov_hybrid(df)
        else:
            hasil = top5_lstm(df)

        st.subheader("🎯 Prediksi Posisi (Top 5 Alternatif per Digit):")
        for i, posisi in enumerate(["Digit 1 (Ribuan)", "Digit 2 (Ratusan)", "Digit 3 (Puluhan)", "Digit 4 (Satuan)"]):
            st.markdown(f"**{posisi}:** {', '.join(str(d) for d in hasil[i])}")

        if angka_aktual and angka_aktual.isdigit() and len(angka_aktual) == 4:
            uji_df = df.tail(jumlah_uji)
            total = 0
            benar = 0

            for i in range(len(uji_df)):
                subset_df = df.iloc[:-(len(uji_df) - i)]
                if len(subset_df) >= 11:
                    if metode == "LSTM AI":
                        prediksi = top5_lstm(subset_df)
                    elif metode == "Markov Order-2":
                        prediksi = top5_markov_order2(subset_df)
                    elif metode == "Markov Gabungan":
                        prediksi = top5_markov_hybrid(subset_df)
                    else:
                        prediksi = top5_markov(subset_df)

                    actual = f"{int(uji_df.iloc[i]['angka']):04d}"
                    for j in range(4):
                        if int(actual[j]) in prediksi[j]:
                            benar += 1
                    total += 4

            if total > 0:
                akurasi_total = (benar / total) * 100
                st.info(f"📈 Akurasi per digit (dari {len(uji_df)} data): {akurasi_total:.2f}%")
            else:
                st.warning("⚠️ Tidak cukup data untuk menghitung akurasi.")
