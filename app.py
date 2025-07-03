import streamlit as st
import pandas as pd
from markov_model import top5_markov, top5_markov_order2, top5_markov_hybrid
from ai_model import top5_lstm

st.set_page_config(page_title="Prediksi Togel AI", layout="centered")

st.title("🎰 Prediksi Togel 4 Digit - AI & Markov")

# Input history data
riwayat_input = st.text_area("🧾 Masukkan data history togel (1 angka per baris):", height=200)
data_lines = [line.strip() for line in riwayat_input.split("\n") if line.strip().isdigit() and len(line.strip()) == 4]
df = pd.DataFrame({"angka": data_lines})

# Selectbox lokasi pasaran (tanpa icon)
lokasi_list = [
    "ARMENIA", "ATLANTIC DAY", "ATLANTIC MORNING", "ATLANTIC NIGHT", "AZERBAIJAN", "BAHRAIN",
    "BARCELONA", "BATAVIA", "BHUTAN", "BIRMINGHAM", "BRISBANE", "BRITANIA", "BRUNEI", "BUCHAREST",
    "BUDAPEST", "BULLSEYE", "CALIFORNIA", "CAMBODIA", "CAMBRIDGE", "CANBERRA", "CHILE", "CHINA",
    "COLOMBIA", "COLORADO DAY", "COLORADO EVENING", "COLORADO MORNING", "COPENHAGEN", "CYPRUS",
    "DARWIN", "DELAWARE DAY", "DELAWARE NIGHT", "DUBLIN DAY", "DUBLIN MORNING", "DUBLIN NIGHT",
    "EMIRATES DAY", "EMIRATES NIGHT", "EURO", "FLORIDA EVENING", "FLORIDA MIDDAY", "GEORGIA EVENING",
    "GEORGIA MIDDAY", "GEORGIA NIGHT", "GERMANY PLUS5", "GREENLAND", "HELSINKI", "HONGKONG",
    "HONGKONG LOTTO", "HOUSTON", "ILLINOIS EVENING", "ILLINOIS MIDDAY", "INDIANA EVENING",
    "INDIANA MIDDAY", "IVORY COAST", "JAPAN", "JORDAN", "KENTUCKY EVENING", "KENTUCKY MIDDAY",
    "KUWAIT", "LAOS", "LEBANON", "MAGNUM4D", "MANHATTAN DAY", "MANHATTAN NIGHT", "MARYLAND EVENING",
    "MARYLAND MIDDAY", "MASSACHUSETTS EVENING", "MASSACHUSETTS MIDDAY", "MICHIGAN EVENING",
    "MICHIGAN MIDDAY", "MIDWAY", "MILAN", "MISSOURI EVENING", "MISSOURI MIDDAY", "MONACO",
    "MOROCCO QUATRO 00:00", "MOROCCO QUATRO 01:00", "MOROCCO QUATRO 02:00", "MOROCCO QUATRO 03:00",
    "MOROCCO QUATRO 04:00", "MOROCCO QUATRO 05:00", "MOROCCO QUATRO 06:00", "MOROCCO QUATRO 15:00",
    "MOROCCO QUATRO 16:00", "MOROCCO QUATRO 17:00", "MOROCCO QUATRO 18:00", "MOROCCO QUATRO 19:00",
    "MOROCCO QUATRO 20:00", "MOROCCO QUATRO 21:00", "MOROCCO QUATRO 22:00", "MOROCCO QUATRO 23:00",
    "MUNICH", "MYANMAR", "NAMIBIA", "NEVADA", "NEVADA DAY", "NEVADA EVENING", "NEVADA MORNING",
    "NEW JERSEY EVENING", "NEW JERSEY MIDDAY", "NEW YORK EVENING", "NEW YORK MIDDAY", "NICOSIA",
    "NORTH CAROLINA DAY", "NORTH CAROLINA EVENING", "OHIO EVENING", "OHIO MIDDAY", "OKLAHOMA DAY",
    "OKLAHOMA EVENING", "OKLAHOMA MORNING", "OMAN", "OREGON 1", "OREGON 2", "OREGON 3", "OREGON 4",
    "ORLANDO", "OSLO", "PACIFIC", "PCSO", "PENNSYLVANIA DAY", "PENNSYLVANIA EVENING", "QATAR",
    "QUEENSLAND", "RHODE ISLAND MIDDAY", "ROTTERDAM", "SINGAPORE", "SINGAPORE 6D",
    "SOUTH CAROLINA MIDDAY", "STOCKHOLM", "SYDNEY", "SYDNEY LOTTO", "TAIWAN", "TENNESSE EVENING",
    "TENNESSE MIDDAY", "TENNESSE MORNING", "TEXAS DAY", "TEXAS EVENING", "TEXAS MORNING",
    "TEXAS NIGHT", "THAILAND", "TOTO MACAU 00:00", "TOTO MACAU 13:00", "TOTO MACAU 16:00",
    "TOTO MACAU 19:00", "TOTO MACAU 22:00", "TURKEY", "UAE", "USA DAY", "USA NIGHT", "UTAH DAY",
    "UTAH EVENING", "UTAH MORNING", "VENEZIA", "VIRGINIA DAY", "VIRGINIA NIGHT",
    "WASHINGTON DC EVENING", "WASHINGTON DC MIDDAY", "WEST VIRGINIA", "WISCONSIN", "YAMAN", "ZURIC"
]

selected_lokasi = st.selectbox("🌍 Pilih Lokasi Pasaran", lokasi_list)

# Tampilkan angka valid dari input
with st.expander("✅ Daftar Angka Valid (4 Digit)"):
    if data_lines:
        st.code("\n".join(data_lines), language='text')
        st.success(f"Total: {len(data_lines)} data valid.")
    else:
        st.warning("Belum ada data valid yang dimasukkan.")

# Input pengujian
jumlah_uji = st.number_input("📊 Jumlah data uji terakhir:", min_value=1, max_value=500, value=5, step=1)
metode = st.selectbox("🧠 Pilih Metode Prediksi", ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI"])

# Tombol prediksi
if st.button("🔮 Prediksi"):
    if len(df) < 11:
        st.warning("❌ Minimal 11 data diperlukan untuk prediksi.")
    else:
        # Lakukan prediksi berdasarkan metode
        if metode == "Markov":
            hasil = top5_markov(df)
        elif metode == "Markov Order-2":
            hasil = top5_markov_order2(df)
        elif metode == "Markov Gabungan":
            hasil = top5_markov_hybrid(df)
        else:
            hasil = top5_lstm(df)

        # Tampilkan hasil prediksi
        st.subheader("🎯 Prediksi Posisi (Top 5 Alternatif per Digit):")
        for i, posisi in enumerate(["Digit 1 (Ribuan)", "Digit 2 (Ratusan)", "Digit 3 (Puluhan)", "Digit 4 (Satuan)"]):
            st.markdown(f"**{posisi}:** {', '.join(str(d) for d in hasil[i])}")

        # Uji akurasi jika cukup data
        uji_df = df.tail(jumlah_uji)
        total, benar = 0, 0

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
