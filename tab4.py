import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

def split_digits(num_str):
    return [int(d) for d in str(num_str).zfill(4)]

def analyze_frequency_per_position(data):
    position_freq = {i: Counter() for i in range(4)}
    for num in data:
        digits = split_digits(num)
        for i, d in enumerate(digits):
            position_freq[i][d] += 1
    return position_freq

def analyze_delay_per_position(data):
    delays = [{d: 0 for d in range(10)} for _ in range(4)]
    last_seen = [{d: -1 for d in range(10)} for _ in range(4)]

    for idx, num in enumerate(data):
        digits = split_digits(num)
        for pos, d in enumerate(digits):
            for digit in range(10):
                if digit == d:
                    if last_seen[pos][digit] != -1:
                        delays[pos][digit] = idx - last_seen[pos][digit]
                    last_seen[pos][digit] = idx
    return delays

def digit_position_heatmap(data):
    pos_counts = np.zeros((4, 10))
    for num in data:
        digits = split_digits(num)
        for i, d in enumerate(digits):
            pos_counts[i][d] += 1
    return pos_counts

def analyze_trend_per_position(data):
    trends = [{'naik': 0, 'turun': 0, 'tetap': 0} for _ in range(4)]
    prev_digits = None
    for num in data:
        digits = split_digits(num)
        if prev_digits:
            for i in range(4):
                if digits[i] > prev_digits[i]:
                    trends[i]['naik'] += 1
                elif digits[i] < prev_digits[i]:
                    trends[i]['turun'] += 1
                else:
                    trends[i]['tetap'] += 1
        prev_digits = digits
    return trends

def zigzag_pattern(data):
    pattern = 0
    for i in range(2, len(data)):
        a = int(str(data[i-2])[-1])
        b = int(str(data[i-1])[-1])
        c = int(str(data[i])[-1])
        if (a < b > c) or (a > b < c):
            pattern += 1
    return pattern

def even_odd_analysis_per_position(data):
    result = [{'genap': 0, 'ganjil': 0} for _ in range(4)]
    for num in data:
        digits = split_digits(num)
        for i, d in enumerate(digits):
            if d % 2 == 0:
                result[i]['genap'] += 1
            else:
                result[i]['ganjil'] += 1
    return result

def big_small_analysis_per_position(data):
    result = [{'besar': 0, 'kecil': 0} for _ in range(4)]
    for num in data:
        digits = split_digits(num)
        for i, d in enumerate(digits):
            if d >= 5:
                result[i]['besar'] += 1
            else:
                result[i]['kecil'] += 1
    return result

def predict_next_pattern(freq, delay, heatmap):
    predicted_digits = []
    sorted_delay = sorted(delay.items(), key=lambda x: x[1], reverse=True)
    predicted_digits += [d for d, _ in sorted_delay[:2]]

    sorted_freq = sorted(freq.items(), key=lambda x: x[1])
    predicted_digits += [d for d, _ in sorted_freq[:2]]

    rare_pos_digits = []
    for row in heatmap:
        rare_pos_digits.append(int(np.argmin(row)))
    predicted_digits += rare_pos_digits

    predicted_digits = list(dict.fromkeys(predicted_digits))
    return predicted_digits[:6]

def tab4(df):
    st.title("📊 Analisis Pola Angka 4D")
    if "angka" not in df.columns:
        st.error("❌ Kolom 'angka' tidak ditemukan di data.")
        return

    angka_data = df["angka"].dropna().astype(int).tolist()

    if not angka_data:
        st.warning("⚠️ Data 4D kosong.")
        return

    if st.button("🔍 Analisis Pola Sekarang", use_container_width=True):
        pos_labels = ["Ribu", "Ratus", "Puluh", "Satuan"]

        # Frekuensi per posisi
        st.markdown("### 🔁 Frekuensi Digit per Posisi")
        freq_pos = analyze_frequency_per_position(angka_data)
        for i in range(4):
            st.markdown(f"**📍 Posisi {pos_labels[i]}**")
            freq_df = pd.DataFrame(freq_pos[i].items(), columns=["Digit", "Frekuensi"]).sort_values("Digit")
            st.bar_chart(freq_df.set_index("Digit"))

        # Delay per posisi
        st.markdown("### ⏱️ Delay Kemunculan Digit per Posisi")
        delays = analyze_delay_per_position(angka_data)
        for i in range(4):
            st.markdown(f"**📍 Posisi {pos_labels[i]}**")
            st.json(delays[i])

        # Heatmap posisi
        st.markdown("### 🔥 Heatmap Posisi Digit")
        heatmap = digit_position_heatmap(angka_data)
        fig, ax = plt.subplots(figsize=(8, 2))
        sns.heatmap(heatmap, annot=True, fmt=".0f", cmap="YlGnBu",
                    xticklabels=list(range(10)),
                    yticklabels=pos_labels, ax=ax)
        ax.set_title("Heatmap Posisi Digit")
        st.pyplot(fig)

        # Tren per posisi
        st.markdown("### 📈 Tren Naik / Turun per Posisi")
        trends = analyze_trend_per_position(angka_data)
        for i in range(4):
            st.markdown(f"**📍 Posisi {pos_labels[i]}**")
            st.write(trends[i])

        # Zigzag
        st.markdown("### 🔀 Pola Zigzag")
        zz = zigzag_pattern(angka_data)
        st.write(f"Zigzag pattern ditemukan: `{zz}` kali")

        # Ganjil / Genap per posisi
        st.markdown("### 🧮 Statistik Ganjil / Genap per Posisi")
        eo = even_odd_analysis_per_position(angka_data)
        for i in range(4):
            st.markdown(f"**📍 Posisi {pos_labels[i]}**")
            st.write(eo[i])

        # Besar / Kecil per posisi
        st.markdown("### 🔢 Statistik Besar / Kecil per Posisi")
        bs = big_small_analysis_per_position(angka_data)
        for i in range(4):
            st.markdown(f"**📍 Posisi {pos_labels[i]}**")
            st.write(bs[i])

        # Insight
        st.markdown("### 🧠 Insight Otomatis")
        flat_digits = [d for num in angka_data for d in split_digits(num)]
        flat_freq = Counter(flat_digits)
        most_common_digit = max(flat_freq.items(), key=lambda x: x[1])[0]
        flat_delay = analyze_delay_per_position(angka_data)
        flat_delay_all = defaultdict(int)
        for pos in flat_delay:
            for k, v in pos.items():
                flat_delay_all[k] = max(flat_delay_all[k], v)
        delay_sorted = sorted(flat_delay_all.items(), key=lambda x: x[1], reverse=True)
        st.info(f"Digit paling sering muncul: `{most_common_digit}`")
        if delay_sorted:
            st.info(f"Digit dengan delay tertinggi: `{delay_sorted[0][0]}` (delay: {delay_sorted[0][1]} langkah)")

        # Prediksi pola
        st.markdown("### 🔮 Prediksi Pola Selanjutnya")
        prediksi = predict_next_pattern(flat_freq, flat_delay_all, heatmap)
        cols = st.columns(len(prediksi))
        for i, d in enumerate(prediksi):
            cols[i].markdown(
                f"<div style='background-color:#0e1117;color:white;border-radius:50%;width:48px;height:48px;"
                f"display:flex;align-items:center;justify-content:center;font-size:20px;"
                f"box-shadow:0 0 6px rgba(0,255,255,0.4);'>{d}</div>",
                unsafe_allow_html=True
            )
        st.caption("📌 Prediksi ini berbasis statistik delay, frekuensi, dan heatmap posisi.")
