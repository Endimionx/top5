import streamlit as st
import pandas as pd
import requests
from markov_model import top5_markov, top5_markov_order2, top5_markov_hybrid
from ai_model import top5_lstm

st.set_page_config(page_title="Prediksi Togel AI", layout="centered")
st.title("🎰 Prediksi Togel 4 Digit - AI & Markov")

# --- Pilihan Pasaran dan Hari ---
lokasi_list = [
    "GERMANY", "HONGKONG", "SINGAPORE", "MAGNUM4D", "TOTO MACAU 00:00",
    "USA DAY", "USA NIGHT", "SYDNEY", "PCSO", "BRUNEI", "CAMBODIA"
]
hari_list = ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"]

selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
selected_hari = st.selectbox("📅 Pilih Hari", hari_list)
putaran = st.slider("🔁 Jumlah Putaran", min_value=1, max_value=1000, value=5)

# --- Hit API dan Ambil Data ---
riwayat_input = ""
angka_list = []

if selected_lokasi and selected_hari:
    with st.spinner(f"🔄 Mengambil data dari pasaran '{selected_lokasi}' ({selected_hari})..."):
        try:
            url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran={putaran}&showpasaran=yes&showtgl=yes&format=json"
            headers = {
                "Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"
            }
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and isinstance(data["data"], list):
                    angka_list = [
                        item["result"]
                        for item in data["data"]
                        if isinstance(item, dict) and "result" in item and len(item["result"]) == 4 and item["result"].isdigit()
                    ]
                    if angka_list:
                        riwayat_input = "\n".join(angka_list)
                        st.success(f"✅ {len(angka_list)} angka berhasil diambil dari API.")
                        with st.expander("📥 Hasil Angka dari API"):
                            st.code(riwayat_input)
                    else:
                        st.warning("⚠️ Tidak ada angka valid ditemukan di field 'result'.")
                else:
                    st.warning("⚠️ Format respons tidak valid: tidak ada key 'data'.")
            else:
                st.error(f"❌ Gagal mengakses API. Status code: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Terjadi error saat request API: {e}")

# --- Text Area Input
riwayat_input = st.text_area("🧾 Masukkan data history togel (1 angka per baris):", value=riwayat_input, height=200)
data_lines = [line.strip() for line in riwayat_input.split("\n") if line.strip().isdigit() and len(line.strip()) == 4]
df = pd.DataFrame({"angka": data_lines})

# --- Tampilkan Angka Valid
with st.expander("✅ Daftar Angka Valid"):
    if data_lines:
        st.code("\n".join(data_lines))
        st.success(f"Total: {len(data_lines)} data valid.")
    else:
        st.warning("Belum ada angka valid.")

# --- Pilih Metode
metode = st.selectbox("🧠 Pilih Metode Prediksi", ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI"])

# --- Prediksi & Uji Akurasi
if st.button("🔮 Prediksi"):
    if len(df) < 11:
        st.warning("❌ Minimal 11 data diperlukan untuk prediksi.")
    else:
        # Prediksi
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

        # --- Uji Akurasi berdasarkan jumlah putaran
        uji_df = df.tail(putaran)
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
