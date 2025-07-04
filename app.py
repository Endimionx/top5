import streamlit as st
import pandas as pd
import requests
import os
from dotenv import load_dotenv
from markov_model import top6_markov, top6_markov_order2, top6_markov_hybrid
from ai_model import top6_lstm, train_and_save_lstm, model_exists

load_dotenv()
st.set_page_config(page_title="Prediksi Togel AI", layout="wide")
st.markdown("<h4>Prediksi Togel 4D - AI & Markov</h4>", unsafe_allow_html=True)

# ======================= PASARAN ========================
lokasi_list = sorted(set([
    "ARMENIA", "ATLANTIC DAY", "ATLANTIC MORNING", "ATLANTIC NIGHT", "AZERBAIJAN",
    "BAHRAIN", "BARCELONA", "BATAVIA", "BHUTAN", "BIRMINGHAM", "BRISBANE",
    "BRITANIA", "BRUNEI", "BUCHAREST", "BUDAPEST", "BULLSEYE", "CALIFORNIA",
    "CAMBODIA", "CAMBRIDGE", "CANBERRA", "CHILE", "CHINA", "COLOMBIA",
    "COLORADO DAY", "COLORADO EVENING", "COLORADO MORNING", "COPENHAGEN",
    "CYPRUS", "DARWIN", "DELAWARE DAY", "DELAWARE NIGHT", "DUBLIN DAY",
    "DUBLIN MORNING", "DUBLIN NIGHT", "EMIRATES DAY", "EMIRATES NIGHT", "EURO",
    "FLORIDA EVENING", "FLORIDA MIDDAY", "GEORGIA EVENING", "GEORGIA MIDDAY",
    "GEORGIA NIGHT", "GERMANY PLUS5", "GREENLAND", "HELSINKI", "HONGKONG",
    "HONGKONG LOTTO", "HOUSTON", "ILLINOIS EVENING", "ILLINOIS MIDDAY",
    "INDIANA EVENING", "INDIANA MIDDAY", "IVORY COAST", "JAPAN", "JORDAN",
    "KENTUCKY EVENING", "KENTUCKY MIDDAY", "KUWAIT", "LAOS", "LEBANON",
    "MAGNUM4D", "MANHATTAN DAY", "MANHATTAN NIGHT", "MARYLAND EVENING",
    "MARYLAND MIDDAY", "MASSACHUSETTS EVENING", "MASSACHUSETTS MIDDAY",
    "MICHIGAN EVENING", "MICHIGAN MIDDAY", "MIDWAY", "MILAN", "MISSOURI EVENING",
    "MISSOURI MIDDAY", "MONACO", "MOROCCO QUATRO 00:00", "MOROCCO QUATRO 01:00",
    "MOROCCO QUATRO 02:00", "MOROCCO QUATRO 03:00", "MOROCCO QUATRO 04:00",
    "MOROCCO QUATRO 05:00", "MOROCCO QUATRO 06:00", "MOROCCO QUATRO 15:00",
    "MOROCCO QUATRO 16:00", "MOROCCO QUATRO 17:00", "MOROCCO QUATRO 18:00",
    "MOROCCO QUATRO 19:00", "MOROCCO QUATRO 20:00", "MOROCCO QUATRO 21:00",
    "MOROCCO QUATRO 22:00", "MOROCCO QUATRO 23:00", "MUNICH", "MYANMAR",
    "NAMIBIA", "NEVADA", "NEVADA DAY", "NEVADA EVENING", "NEVADA MORNING",
    "NEW JERSEY EVENING", "NEW JERSEY MIDDAY", "NEW YORK EVENING",
    "NEW YORK MIDDAY", "NICOSIA", "NORTH CAROLINA DAY", "NORTH CAROLINA EVENING",
    "OHIO EVENING", "OHIO MIDDAY", "OKLAHOMA DAY", "OKLAHOMA EVENING",
    "OKLAHOMA MORNING", "OMAN", "OREGON 1", "OREGON 2", "OREGON 3", "OREGON 4",
    "ORLANDO", "OSLO", "PACIFIC", "PCSO", "PENNSYLVANIA DAY",
    "PENNSYLVANIA EVENING", "QATAR", "QUEENSLAND", "RHODE ISLAND MIDDAY",
    "ROTTERDAM", "SINGAPORE", "SINGAPORE 6D", "SOUTH CAROLINA MIDDAY",
    "STOCKHOLM", "SYDNEY", "SYDNEY LOTTO", "TAIWAN", "TENNESSE EVENING",
    "TENNESSE MIDDAY", "TENNESSE MORNING", "TEXAS DAY", "TEXAS EVENING",
    "TEXAS MORNING", "TEXAS NIGHT", "THAILAND", "TOTO MACAU 00:00",
    "TOTO MACAU 13:00", "TOTO MACAU 16:00", "TOTO MACAU 19:00",
    "TOTO MACAU 22:00", "TURKEY", "UAE", "USA DAY", "USA NIGHT", "UTAH DAY",
    "UTAH EVENING", "UTAH MORNING", "VENEZIA", "VIRGINIA DAY", "VIRGINIA NIGHT",
    "WASHINGTON DC EVENING", "WASHINGTON DC MIDDAY", "WEST VIRGINIA",
    "WISCONSIN", "YAMAN", "ZURICH"
]))

hari_list = ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"]

selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
selected_hari = st.selectbox("📅 Pilih Hari", hari_list)
putaran = st.slider("🔁 Jumlah Putaran", 1, 1000, 10)
jumlah_uji = st.number_input("📊 Jumlah Data Uji Akurasi", min_value=1, max_value=1000, value=5)

# ======================= AMBIL DATA ========================
angka_list = []
riwayat_input = ""
if selected_lokasi and selected_hari:
    try:
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

# ======================= PREDIKSI ========================
metode = st.selectbox("🧠 Pilih Metode Prediksi", ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI"])

# ======= Khusus Manajemen Model LSTM =======
if metode == "LSTM AI":
    with st.expander("🛠️ Manajemen Model LSTM"):
        save_dir = os.path.join(os.getcwd(), "saved_models")
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"lstm_{selected_lokasi.lower().replace(' ', '_')}.h5")

        uploaded = st.file_uploader("📤 Upload Model (.h5)", type=["h5"])
        if uploaded is not None:
            with open(model_path, "wb") as f:
                f.write(uploaded.read())
            st.success("✅ Model berhasil diunggah.")

        if st.button("📚 Latih & Simpan Model"):
            if len(df) < 20:
                st.warning("Minimal 20 data untuk latih model.")
            else:
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
            st.warning("⚠️ Model belum tersedia. Silakan latih atau upload.")

# ======================= Prediksi dan Akurasi ========================
if st.button("🔮 Prediksi"):
    if len(df) < 11:
        st.warning("❌ Minimal 11 data diperlukan.")
    else:
        pred = (
            top6_markov(df) if metode == "Markov" else
            top6_markov_order2(df) if metode == "Markov Order-2" else
            top6_markov_hybrid(df) if metode == "Markov Gabungan" else
            top6_lstm(df, lokasi=selected_lokasi)
        )
        if pred is None:
            st.error("❌ Gagal prediksi.")
        else:
            st.markdown("#### 🎯 Prediksi Top-6 Digit per Posisi")
            for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                st.markdown(f"**{label}:** {', '.join(str(d) for d in pred[i])}")

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
