import streamlit as st
import pandas as pd
import requests
import os
from dotenv import load_dotenv
from markov_model import top5_markov, top5_markov_order2, top5_markov_hybrid
from ai_model import top5_lstm
from urllib.parse import unquote

# Load API key
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

st.set_page_config(page_title="Prediksi Togel AI", layout="centered")
st.title("🎰 Prediksi Togel 4D - AI + Markov + Chat")

# Pilihan Pasaran dan Hari
lokasi_list = ["GERMANY", "HONGKONG", "SINGAPORE", "MAGNUM4D", "TOTO MACAU 00:00"]
hari_list = ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"]

selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
selected_hari = st.selectbox("📅 Pilih Hari", hari_list)
putaran = st.slider("🔁 Jumlah Putaran", 1, 1000, 10)
jumlah_uji = st.slider("📊 Jumlah Data Uji Akurasi", 1, 1000, 5)

# Ambil Data dari API
angka_list = []
riwayat_input = ""

if selected_lokasi and selected_hari:
    try:
        url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran={putaran}&showpasaran=yes&showtgl=yes&format=json&urut=asc"
        headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
        response = requests.get(url, headers=headers)
        data = response.json()
        angka_list = [
            item["result"]
            for item in data.get("data", [])
            if isinstance(item, dict) and len(item["result"]) == 4 and item["result"].isdigit()
        ]
        riwayat_input = "\n".join(angka_list)
        st.success(f"✅ {len(angka_list)} angka berhasil diambil.")
        with st.expander("📥 Lihat Angka dari API"):
            st.code(riwayat_input)
    except Exception as e:
        st.error(f"❌ Gagal ambil data API: {e}")

# Parse angka menjadi DataFrame
data_lines = [x.strip() for x in riwayat_input.split("\n") if x.strip().isdigit() and len(x.strip()) == 4]
df = pd.DataFrame({"angka": data_lines})

with st.expander("✅ Daftar Angka Valid"):
    st.code("\n".join(data_lines))

# Pilih metode prediksi
metode = st.selectbox("🧠 Pilih Metode", ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI"])
hasil = None
akurasi_total = None

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

        # Uji akurasi
        list_akurasi = []
        uji_df = df.tail(min(jumlah_uji, len(df)))
        total = benar = 0

        for i in range(len(uji_df)):
            subset_df = df.iloc[:-(len(uji_df) - i)]
            if len(subset_df) < 11:
                continue

            if metode == "Markov":
                pred = top5_markov(subset_df)
            elif metode == "Markov Order-2":
                pred = top5_markov_order2(subset_df)
            elif metode == "Markov Gabungan":
                pred = top5_markov_hybrid(subset_df)
            elif metode == "LSTM AI":
                pred = top5_lstm(subset_df)

            actual = f"{int(uji_df.iloc[i]['angka']):04d}"
            skor = sum(int(actual[j]) in pred[j] for j in range(4))
            total += 4
            benar += skor
            list_akurasi.append(skor / 4 * 100)

        if total > 0:
            akurasi_total = (benar / total) * 100
            st.info(f"📈 Akurasi {metode}: {akurasi_total:.2f}%")

        if list_akurasi:
            with st.expander("📊 Grafik Akurasi"):
                st.line_chart(pd.DataFrame({"Akurasi (%)": list_akurasi}))

# Chat Assistant Melayang
st.markdown("""
<style>
#open-chat-btn {
    position: fixed;
    bottom: 25px;
    right: 25px;
    background-color: #25d366;
    color: white;
    padding: 14px;
    border: none;
    border-radius: 50%;
    font-size: 22px;
    cursor: pointer;
    z-index: 100;
}
#chat-box {
    position: fixed;
    bottom: 90px;
    right: 25px;
    width: 320px;
    max-height: 400px;
    background-color: white;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    display: none;
    flex-direction: column;
    z-index: 100;
    padding: 10px;
    overflow-y: auto;
}
#chat-box textarea {
    width: 100%;
    border: none;
    border-top: 1px solid #ccc;
    resize: none;
    padding: 8px;
    margin-top: 5px;
    border-radius: 5px;
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

chatBtn.onclick = () => {
    chatBox.style.display = chatBox.style.display === "none" ? "flex" : "none";
};

chatInput.addEventListener("keydown", async function(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        const msg = chatInput.value.trim();
        if (!msg) return;
        chatLog.innerHTML += `<div style='text-align:right;'>🧑‍💬 ${msg}</div>`;
        chatInput.value = "...";
        const response = await fetch("/chat?q=" + encodeURIComponent(msg));
        const result = await response.text();
        chatLog.innerHTML += `<div style='text-align:left;'>🤖 ${result}</div>`;
        chatInput.value = "";
    }
});
</script>
""", unsafe_allow_html=True)

# Endpoint untuk Chat Assistant
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
