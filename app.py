import streamlit as st
import pandas as pd
import requests
import os
from dotenv import load_dotenv
from markov_model import top5_markov, top5_markov_order2, top5_markov_hybrid
from ai_model import top5_lstm

# Load API key dari .env
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Togel AI + Chat", layout="centered")
st.title("🎰 Prediksi Togel 4 Digit - AI & Markov + Together.ai Assistant")

# --- Pilihan Pasaran dan Hari
lokasi_list = ["GERMANY", "HONGKONG", "SINGAPORE", "MAGNUM4D", "TOTO MACAU 00:00"]
hari_list = ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"]

selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
selected_hari = st.selectbox("📅 Pilih Hari", hari_list)
putaran = st.slider("🔁 Jumlah Putaran (Ambil dari API)", 1, 1000, 10)
jumlah_uji = st.slider("📊 Jumlah Data Uji Akurasi", 1, 1000, 5)

# --- Ambil Data dari API
angka_list = []
riwayat_input = ""

if selected_lokasi and selected_hari:
    try:
        url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran={putaran}&showpasaran=yes&showtgl=yes&format=json&urut=asc"
        headers = {
            "Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"
        }
        response = requests.get(url, headers=headers)
        data = response.json()
        angka_list = [
            item["result"]
            for item in data.get("data", [])
            if isinstance(item, dict) and len(item["result"]) == 4 and item["result"].isdigit()
        ]
        riwayat_input = "\n".join(angka_list)
        st.success(f"✅ {len(angka_list)} angka berhasil diambil dari API.")
        with st.expander("📥 Lihat Angka dari API"):
            st.code(riwayat_input)
    except Exception as e:
        st.error(f"❌ Gagal ambil data API: {e}")

# --- Parse Data
data_lines = [x.strip() for x in riwayat_input.split("\n") if x.strip().isdigit() and len(x.strip()) == 4]
df = pd.DataFrame({"angka": data_lines})

with st.expander("✅ Daftar Angka Valid"):
    st.code("\n".join(data_lines))

# --- Pilih Metode
metode = st.selectbox("🧠 Pilih Metode Prediksi", ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI"])
hasil = None
akurasi_total = None

# --- Prediksi & Akurasi
if st.button("🔮 Prediksi"):
    if len(df) < 11:
        st.warning("❌ Minimal 11 data diperlukan.")
    else:
        if metode == "Markov":
            hasil = top5_markov(df)
        elif metode == "Markov Order-2":
            hasil = top5_markov_order2(df)
        elif metode == "Markov Gabungan":
            hasil = top5_markov_hybrid(df)
        else:
            hasil = top5_lstm(df)

        st.markdown("#### 🎯 Prediksi Top-5 Digit")
        for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
            st.markdown(f"**{label}:** {', '.join(str(d) for d in hasil[i])}")

        # --- Uji Akurasi
        list_akurasi = []
        if metode == "LSTM AI":
            if len(df) >= jumlah_uji + 11:
                uji_df = df.tail(jumlah_uji)
                train_df = df.iloc[:-jumlah_uji]
                prediksi = top5_lstm(train_df)
                if prediksi:
                    total = benar = 0
                    for i in range(len(uji_df)):
                        actual = f"{int(uji_df.iloc[i]['angka']):04d}"
                        skor = sum(int(actual[j]) in prediksi[j] for j in range(4))
                        total += 4
                        benar += skor
                        list_akurasi.append(skor / 4 * 100)
                    akurasi_total = (benar / total) * 100
                    st.info(f"📈 Akurasi LSTM AI: {akurasi_total:.2f}%")
        else:
            uji_df = df.tail(min(jumlah_uji, len(df)))
            total = benar = 0
            for i in range(len(uji_df)):
                subset_df = df.iloc[:-(len(uji_df) - i)]
                if len(subset_df) < 11: continue
                if metode == "Markov":
                    pred = top5_markov(subset_df)
                elif metode == "Markov Order-2":
                    pred = top5_markov_order2(subset_df)
                elif metode == "Markov Gabungan":
                    pred = top5_markov_hybrid(subset_df)
                actual = f"{int(uji_df.iloc[i]['angka']):04d}"
                skor = sum(int(actual[j]) in pred[j] for j in range(4))
                total += 4
                benar += skor
                list_akurasi.append(skor / 4 * 100)
            if total > 0:
                akurasi_total = (benar / total) * 100
                st.info(f"📈 Akurasi {metode}: {akurasi_total:.2f}%")

        if list_akurasi:
            with st.expander("📊 Grafik Akurasi per Data"):
                st.line_chart(pd.DataFrame({"Akurasi (%)": list_akurasi}))

# -----------------------------
# Together AI Chat Assistant
st.markdown("---")
st.markdown("### 💬 Chat Assistant (via Together.ai)")

if not TOGETHER_API_KEY:
    st.error("❌ API Key Together.ai tidak ditemukan di file .env.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Tanya soal prediksi, akurasi, metode...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        context = f"""
        Jumlah data: {len(df)}
        Metode: {metode}
        Prediksi:
        Ribuan: {hasil[0] if hasil else []}
        Ratusan: {hasil[1] if hasil else []}
        Puluhan: {hasil[2] if hasil else []}
        Satuan: {hasil[3] if hasil else []}
        Akurasi: {akurasi_total if akurasi_total else 'Belum tersedia'}
        """

        with st.chat_message("assistant"):
            with st.spinner("🤖 Menjawab dari Together.ai..."):
                try:
                    headers = {
                        "Authorization": f"Bearer {TOGETHER_API_KEY}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": "mistralai/Mistral-7B-Instruct-v0.1",
                        "messages": [
                            {"role": "system", "content": "Kamu adalah asisten AI untuk prediksi angka dan analisis statistik."},
                            {"role": "user", "content": f"{context}\n\nPertanyaan: {prompt}"}
                        ],
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 512
                    }

                    response = requests.post("https://api.together.ai/v1/chat/completions", headers=headers, json=payload)
                    response.raise_for_status()
                    reply = response.json()["choices"][0]["message"]["content"]

                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.error(f"❌ Gagal menjawab dari Together.ai: {e}")
