import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from dotenv import load_dotenv
from pathlib import Path
from urllib.parse import unquote
from markov_model import top5_markov, top5_markov_order2, top5_markov_hybrid
from ai_model import top5_lstm, train_and_save_lstm, model_exists
import streamlit.components.v1 as components

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Prediksi Togel AI + Chat", layout="centered")
st.markdown("<h4>Prediksi Togel 4 Digit - AI & Markov + Chat</h4>", unsafe_allow_html=True)

# ======================= PASARAN ========================
lokasi_list = [
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
]
hari_list = ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"]

selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
selected_hari = st.selectbox("📅 Pilih Hari", hari_list)
putaran = st.slider("🔁 Jumlah Putaran", 1, 1000, 10)
jumlah_uji = st.slider("📊 Jumlah Data Uji Akurasi", 1, 1000, 5)

# ======================= AMBIL DATA ========================
angka_list = []
riwayat_input = ""
if selected_lokasi and selected_hari:
    try:
        url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran={putaran}&format=json&urut=asc"
        headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
        response = requests.get(url, headers=headers)
        data = response.json()
        angka_list = [item["result"] for item in data.get("data", []) if len(item["result"]) == 4 and item["result"].isdigit()]
        riwayat_input = "\n".join(angka_list)
        st.success(f"✅ {len(angka_list)} angka berhasil diambil dari API.")
        with st.expander("📥 Lihat Data"):
            st.code(riwayat_input)
    except Exception as e:
        st.error(f"❌ Gagal ambil data API: {e}")

data_lines = [x.strip() for x in riwayat_input.split("\n") if x.strip().isdigit() and len(x.strip()) == 4]
df = pd.DataFrame({"angka": data_lines})
with st.expander("✅ Daftar Angka Valid"):
    st.code("\n".join(data_lines))

# ======================= PREDIKSI ========================
metode = st.selectbox("🧠 Pilih Metode Prediksi", ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI"])

if metode == "LSTM AI":
    if model_exists(selected_lokasi):
        st.success("📁 Model tersedia untuk pasaran ini.")
    else:
        st.warning("⚠️ Model belum tersedia. Latih model terlebih dahulu.")

    if st.button("💾 Latih & Simpan Model"):
        if len(df) < 11:
            st.warning("❌ Minimal 11 data diperlukan untuk pelatihan.")
        else:
            train_and_save_lstm(df, lokasi=selected_lokasi)
            st.success("✅ Model berhasil dilatih & disimpan.")

if st.button("🔮 Prediksi"):
    if len(df) < 11:
        st.warning("❌ Minimal 11 data diperlukan.")
    else:
        hasil = None
        if metode == "Markov":
            hasil = top5_markov(df)
        elif metode == "Markov Order-2":
            hasil = top5_markov_order2(df)
        elif metode == "Markov Gabungan":
            hasil = top5_markov_hybrid(df)
        else:
            hasil = top5_lstm(df, lokasi=selected_lokasi)

        if hasil is None:
            st.error("❌ Gagal memproses prediksi. Cek apakah model sudah dilatih.")
            st.stop()

        st.markdown("#### 🎯 Prediksi Top-5 Digit")
        for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
            st.markdown(f"**{label}:** {', '.join(str(d) for d in hasil[i])}")

        # Hitung Akurasi
        list_akurasi = []
        uji_df = df.tail(min(jumlah_uji, len(df)))
        total = benar = 0
        for i in range(len(uji_df)):
            subset_df = df.iloc[:-(len(uji_df) - i)]
            if len(subset_df) < 11:
                continue
            pred = (
                top5_markov(subset_df) if metode == "Markov" else
                top5_markov_order2(subset_df) if metode == "Markov Order-2" else
                top5_markov_hybrid(subset_df) if metode == "Markov Gabungan" else
                top5_lstm(subset_df, lokasi=selected_lokasi)
            )
            if pred is None: continue
            actual = f"{int(uji_df.iloc[i]['angka']):04d}"
            skor = sum(int(actual[j]) in pred[j] for j in range(4))
            total += 4
            benar += skor
            list_akurasi.append(skor / 4 * 100)

        if total > 0:
            akurasi_total = (benar / total) * 100
            st.info(f"📈 Akurasi {metode}: {akurasi_total:.2f}%")
        else:
            st.warning("⚠️ Tidak cukup data untuk hitung akurasi.")

        if list_akurasi:
            with st.expander("📊 Grafik Akurasi"):
                st.line_chart(pd.DataFrame({"Akurasi (%)": list_akurasi}))

# ======================= MANAJEMEN MODEL ========================
if metode == "LSTM AI":
    st.markdown("## 📁 Manajemen Model Tersimpan")
    model_files = list(MODEL_DIR.glob("*.h5"))
    if not model_files:
        st.info("Tidak ada model tersimpan di folder `saved_models/`.")
    else:
        for model_file in model_files:
            col1, col2, col3 = st.columns([3, 2, 2])
            with col1:
                st.markdown(f"📄 **{model_file.name}**")
            with col2:
                with open(model_file, "rb") as f:
                    st.download_button("⬇️ Download", f, file_name=model_file.name, mime="application/octet-stream", key=f"dl-{model_file.name}")
            with col3:
                if st.button("🗑️ Hapus", key=f"del-{model_file.name}"):
                    try:
                        os.remove(model_file)
                        st.success(f"{model_file.name} dihapus.")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Gagal hapus: {e}")

# ======================= FLOATING CHAT ========================
components.html("""
<style>
#open-chat-btn {
    position: fixed; bottom: 25px; right: 25px;
    background-color: #25d366; color: white;
    padding: 14px; border: none; border-radius: 50%;
    font-size: 22px; cursor: pointer; z-index: 9999;
}
#chat-box {
    position: fixed; bottom: 90px; right: 25px;
    width: 320px; max-height: 400px; background-color: white;
    border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    display: none; flex-direction: column; z-index: 9999;
    padding: 10px; overflow-y: auto;
}
#chat-box textarea {
    width: 100%; border: none; border-top: 1px solid #ccc;
    resize: none; padding: 8px; margin-top: 5px; border-radius: 5px;
}
</style>
<button id="open-chat-btn">💬</button>
<div id="chat-box">
    <div><b>AI Assistant</b></div>
    <div id="chat-log" style="font-size:14px; margin: 10px 0; max-height:300px; overflow-y:auto;"></div>
    <textarea id="chat-input" rows="2" placeholder="Tulis pertanyaan..."></textarea>
</div>
<script>
const chatBox = document.getElementById("chat-box");
const chatBtn = document.getElementById("open-chat-btn");
const chatInput = document.getElementById("chat-input");
const chatLog = document.getElementById("chat-log");
chatBtn.onclick = () => { chatBox.style.display = chatBox.style.display === "none" ? "flex" : "none"; };
chatInput.addEventListener("keydown", async function(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        const msg = chatInput.value.trim();
        if (!msg) return;
        chatLog.innerHTML += `<div style='text-align:right;'>🧑‍💬 ${msg}</div>`;
        chatInput.value = "...";
        const response = await fetch(`/chat?q=${encodeURIComponent(msg)}`);
        const result = await response.text();
        chatLog.innerHTML += `<div style='text-align:left;'>🤖 ${result}</div>`;
        chatInput.value = "";
    }
});
</script>
""", height=0)

if "q" in st.query_params:
    q = st.query_params["q"]
    try:
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "mistralai/Mistral-7B-Instruct-v0.1",
            "messages": [
                {"role": "system", "content": "Kamu adalah asisten AI statistik dan prediksi angka."},
                {"role": "user", "content": unquote(q)}
            ],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 512
        }
        res = requests.post("https://api.together.ai/v1/chat/completions", headers=headers, json=payload)
        st.write(res.json()["choices"][0]["message"]["content"])
    except Exception as e:
        st.write(f"[Error]: {e}")
    st.stop()
