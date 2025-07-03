import streamlit as st
import pandas as pd
import requests
import os
from dotenv import load_dotenv
from markov_model import top5_markov, top5_markov_order2, top5_markov_hybrid
from ai_model import top5_lstm, prepare_lstm_data, train_lstm_model
import streamlit.components.v1 as components

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

st.set_page_config(page_title="Prediksi Togel AI + Chat", layout="centered")
st.markdown("<h4>Prediksi Togel 4 Digit - AI & Markov + Chat</h4>", unsafe_allow_html=True)

menu = st.sidebar.radio("📂 Menu", ["Prediksi", "Latih Model"])

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
        url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran={putaran}&format=json"
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

data_lines = [x.strip() for x in riwayat_input.split("\n") if x.strip().isdigit() and len(x.strip()) == 4]
df = pd.DataFrame({"angka": data_lines})

# ======================= MENU ========================
if menu == "Prediksi":
    metode = st.selectbox("🧠 Pilih Metode Prediksi", ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI"])
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
                if hasil is None:
                    st.error("❌ Prediksi LSTM gagal. Data tidak mencukupi.")
                    st.stop()

            st.markdown("#### 🎯 Prediksi Top-5 Digit")
            for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                st.markdown(f"**{label}:** {', '.join(str(d) for d in hasil[i])}")

            list_akurasi = []
            uji_df = df.tail(min(jumlah_uji, len(df)))
            total = benar = 0
            for i in range(len(uji_df)):
                subset_df = df.iloc[:-(len(uji_df) - i)]
                if len(subset_df) < 11:
                    continue
                pred = top5_markov(subset_df) if metode == "Markov" else \
                       top5_markov_order2(subset_df) if metode == "Markov Order-2" else \
                       top5_markov_hybrid(subset_df) if metode == "Markov Gabungan" else \
                       top5_lstm(subset_df)
                if pred is None:
                    continue
                actual = f"{int(uji_df.iloc[i]['angka']):04d}"
                skor = sum(int(actual[j]) in pred[j] for j in range(4))
                total += 4
                benar += skor
                list_akurasi.append(skor / 4 * 100)

            if total > 0:
                st.info(f"📈 Akurasi {metode}: {benar / total * 100:.2f}%")
            if list_akurasi:
                with st.expander("📊 Grafik Akurasi"):
                    st.line_chart(pd.DataFrame({"Akurasi (%)": list_akurasi}))

elif menu == "Latih Model":
    if len(df) < 20:
        st.warning("❌ Minimal 20 data diperlukan untuk pelatihan.")
    else:
        X, y = prepare_lstm_data(df)
        model = train_lstm_model(X, y)
        st.success("✅ Model LSTM berhasil dilatih.")

# ======================= FLOATING CHAT ========================
components.html("""
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
    z-index: 9999;
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
    z-index: 9999;
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
</script>
""", height=0)
