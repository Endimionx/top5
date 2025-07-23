import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def split_digits(num_str):
    return [int(d) for d in str(num_str).zfill(4)]

def analyze_frequency(data, pos):
    digits = [split_digits(num)[pos] for num in data]
    return Counter(digits)

def analyze_delay(data, pos):
    delay = {i: 0 for i in range(10)}
    last_seen = {i: -1 for i in range(10)}
    for idx, num in enumerate(data):
        d = split_digits(num)[pos]
        for i in range(10):
            if i == d:
                if last_seen[i] != -1:
                    delay[i] = idx - last_seen[i]
                last_seen[i] = idx
    return delay

def analyze_trend(data, pos):
    trend = {'Naik': 0, 'Turun': 0, 'Tetap': 0}
    prev = None
    for num in data:
        curr = split_digits(num)[pos]
        if prev is not None:
            if curr > prev:
                trend['Naik'] += 1
            elif curr < prev:
                trend['Turun'] += 1
            else:
                trend['Tetap'] += 1
        prev = curr
    return trend

def even_odd_analysis(data, pos):
    return {
        'Genap': sum(1 for num in data if split_digits(num)[pos] % 2 == 0),
        'Ganjil': sum(1 for num in data if split_digits(num)[pos] % 2 == 1),
    }

def big_small_analysis(data, pos):
    return {
        'Besar (≥5)': sum(1 for num in data if split_digits(num)[pos] >= 5),
        'Kecil (<5)': sum(1 for num in data if split_digits(num)[pos] < 5),
    }

def digit_position_heatmap(data):
    pos_counts = np.zeros((4, 10))
    for num in data:
        digits = split_digits(num)
        for i, d in enumerate(digits):
            pos_counts[i][d] += 1
    return pos_counts

def zigzag_pattern(data):
    count = 0
    for i in range(2, len(data)):
        a, b, c = int(str(data[i-2])[-1]), int(str(data[i-1])[-1]), int(str(data[i])[-1])
        if (a < b > c) or (a > b < c):
            count += 1
    return count

def predict_next_pattern(freq, delay, heatmap):
    pred = []
    sorted_delay = sorted(delay.items(), key=lambda x: x[1], reverse=True)
    pred += [d for d, _ in sorted_delay[:2]]
    sorted_freq = sorted(freq.items(), key=lambda x: x[1])
    pred += [d for d, _ in sorted_freq[:2]]
    pred += [int(np.argmin(row)) for row in heatmap]
    return list(dict.fromkeys(pred))[:6]

def render_digit_badge(d):
    return f"""<div style='background:#0e1117;color:white;border-radius:50%;width:48px;height:48px;
                display:flex;align-items:center;justify-content:center;font-size:20px;
                box-shadow:0 0 6px rgba(0,255,255,0.4);margin:4px;'>{d}</div>"""

def tab4(df):
    st.title("📊 Analisis Pola Angka 4D")

    if "angka" not in df.columns:
        st.error("❌ Kolom 'angka' tidak ditemukan.")
        return

    angka_data = df["angka"].dropna().astype(int).tolist()
    if not angka_data:
        st.warning("⚠️ Data kosong.")
        return

    posisi = ["Ribuan", "Ratusan", "Puluhan", "Satuan"]
    tabs = st.tabs(posisi)

    for i, tab in enumerate(tabs):
        with tab:
            st.subheader(f"📌 Posisi: {posisi[i]}")

            freq = analyze_frequency(angka_data, i)
            freq_df = pd.DataFrame(freq.items(), columns=["Digit", "Frekuensi"]).sort_values("Digit")
            st.markdown("**Frekuensi Digit**")
            st.bar_chart(freq_df.set_index("Digit"))

            delay = analyze_delay(angka_data, i)
            st.markdown("**⏱️ Delay Kemunculan Digit**")
            st.write(pd.DataFrame(delay.items(), columns=["Digit", "Delay"]).set_index("Digit").style.bar(subset=["Delay"], color='#00f0ff'))

            trend = analyze_trend(angka_data, i)
            st.markdown("**📈 Tren Naik / Turun**")
            st.write(pd.DataFrame(trend.items(), columns=["Arah", "Jumlah"]).set_index("Arah").style.bar(color='#ffcf00'))

            eo = even_odd_analysis(angka_data, i)
            st.markdown("**🧮 Ganjil / Genap**")
            st.write(pd.DataFrame(eo.items(), columns=["Tipe", "Jumlah"]).set_index("Tipe").style.bar(color='#ff9b00'))

            bs = big_small_analysis(angka_data, i)
            st.markdown("**🔢 Besar / Kecil**")
            st.write(pd.DataFrame(bs.items(), columns=["Kategori", "Jumlah"]).set_index("Kategori").style.bar(color='#00ff88'))

    st.markdown("### 🔥 Heatmap Posisi Digit")
    heatmap = digit_position_heatmap(angka_data)
    fig, ax = plt.subplots(figsize=(8, 2))
    sns.heatmap(heatmap, annot=True, fmt=".0f", cmap="YlGnBu", 
                xticklabels=list(range(10)), yticklabels=posisi, ax=ax)
    st.pyplot(fig)

    st.markdown("### 🔀 Pola Zigzag")
    st.info(f"🔁 Ditemukan pola zigzag sebanyak **{zigzag_pattern(angka_data)}** kali.")

    st.markdown("### 🧠 Insight Otomatis")
    freq_all = analyze_frequency(angka_data, 0)
    delay_all = analyze_delay(angka_data, 0)
    most_common = max(freq_all.items(), key=lambda x: x[1])[0]
    top_delay = max(delay_all.items(), key=lambda x: x[1])
    st.success(f"Digit paling sering muncul (ribuan): `{most_common}`")
    st.info(f"Digit dengan delay tertinggi (ribuan): `{top_delay[0]}` → delay {top_delay[1]} langkah")

    st.markdown("### 🔮 Prediksi Pola Selanjutnya")
    pred = predict_next_pattern(freq_all, delay_all, heatmap)
    st.markdown(f"<div style='display:flex;flex-wrap:wrap'>{''.join(render_digit_badge(d) for d in pred)}</div>", unsafe_allow_html=True)
    st.caption("📌 Prediksi berdasarkan delay, frekuensi, dan distribusi posisi.")
