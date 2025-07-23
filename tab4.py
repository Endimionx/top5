import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

def split_digits(num_str):
    return [int(d) for d in str(num_str).zfill(4)]

def analyze_frequency(data):
    digits = [d for num in data for d in split_digits(num)]
    freq = Counter(digits)
    return freq

def analyze_delay(data):
    delay = {i: 0 for i in range(10)}
    last_seen = {i: -1 for i in range(10)}
    delays = []

    for idx, num in enumerate(data):
        digits = set(split_digits(num))
        for d in range(10):
            if d in digits:
                if last_seen[d] != -1:
                    delay[d] = idx - last_seen[d]
                last_seen[d] = idx
        delays.append(delay.copy())

    return delays[-1] if delays else {}

def digit_position_heatmap(data):
    pos_counts = np.zeros((4, 10))
    for num in data:
        digits = split_digits(num)
        for i, d in enumerate(digits):
            pos_counts[i][d] += 1
    return pos_counts

def analyze_trend(data):
    trend = {'naik': 0, 'turun': 0, 'tetap': 0}
    prev = None
    for num in data:
        if prev is not None:
            if num > prev:
                trend['naik'] += 1
            elif num < prev:
                trend['turun'] += 1
            else:
                trend['tetap'] += 1
        prev = num
    return trend

def zigzag_pattern(data):
    pattern = 0
    for i in range(2, len(data)):
        a = int(str(data[i-2])[-1])
        b = int(str(data[i-1])[-1])
        c = int(str(data[i])[-1])
        if (a < b > c) or (a > b < c):
            pattern += 1
    return pattern

def even_odd_analysis(data):
    counts = {'genap': 0, 'ganjil': 0}
    for num in data:
        digits = split_digits(num)
        for d in digits:
            if d % 2 == 0:
                counts['genap'] += 1
            else:
                counts['ganjil'] += 1
    return counts

def big_small_analysis(data):
    counts = {'besar': 0, 'kecil': 0}
    for num in data:
        digits = split_digits(num)
        for d in digits:
            if d >= 5:
                counts['besar'] += 1
            else:
                counts['kecil'] += 1
    return counts

def predict_next_pattern(freq, delay, heatmap):
    predicted_digits = []

    # 1. Tambahkan digit dengan delay tertinggi
    sorted_delay = sorted(delay.items(), key=lambda x: x[1], reverse=True)
    predicted_digits += [d for d, _ in sorted_delay[:2]]

    # 2. Tambahkan digit paling jarang muncul
    sorted_freq = sorted(freq.items(), key=lambda x: x[1])
    predicted_digits += [d for d, _ in sorted_freq[:2]]

    # 3. Tambahkan digit dari posisi yang paling jarang muncul
    rare_pos_digits = []
    for row in heatmap:
        rare_pos_digits.append(int(np.argmin(row)))
    predicted_digits += rare_pos_digits

    predicted_digits = list(dict.fromkeys(predicted_digits))  # unik
    return predicted_digits[:6]

def tab4(df):
    st.title("📊 Analisis Pola Angka 4D")

    if "angka" not in df.columns:
        st.error("❌ Kolom '4D' tidak ditemukan di data.")
        return

    angka_data = df["angka"].dropna().astype(int).tolist()

    if not angka_data:
        st.warning("⚠️ Data 4D kosong.")
        return

    st.markdown("### 🔁 Frekuensi Digit")
    freq = analyze_frequency(angka_data)
    freq_df = pd.DataFrame(freq.items(), columns=["Digit", "Frekuensi"]).sort_values("Digit")
    st.bar_chart(freq_df.set_index("Digit"))

    st.markdown("### ⏱️ Delay Kemunculan Digit")
    delay = analyze_delay(angka_data)
    st.write(delay)

    st.markdown("### 🔥 Heatmap Posisi Digit")
    heatmap = digit_position_heatmap(angka_data)
    fig, ax = plt.subplots(figsize=(8, 2))
    sns.heatmap(heatmap, annot=True, fmt=".0f", cmap="YlGnBu", xticklabels=list(range(10)), yticklabels=["Ribu", "Ratus", "Puluh", "Satuan"], ax=ax)
    ax.set_title("Heatmap Posisi Digit")
    st.pyplot(fig)

    st.markdown("### 📈 Tren Naik / Turun")
    trend = analyze_trend(angka_data)
    st.write(trend)

    st.markdown("### 🔀 Pola Zigzag")
    zz = zigzag_pattern(angka_data)
    st.write(f"Zigzag pattern ditemukan: `{zz}` kali")

    st.markdown("### 🧮 Statistik Ganjil / Genap")
    evenodd = even_odd_analysis(angka_data)
    st.write(evenodd)

    st.markdown("### 🔢 Statistik Besar / Kecil")
    bigsmall = big_small_analysis(angka_data)
    st.write(bigsmall)

    st.markdown("### 🧠 Insight Otomatis")
    most_common_digit = max(freq.items(), key=lambda x: x[1])[0]
    delay_sorted = sorted(delay.items(), key=lambda x: x[1], reverse=True)
    st.info(f"Digit paling sering muncul: `{most_common_digit}`")
    if delay_sorted:
        st.info(f"Digit dengan delay tertinggi: `{delay_sorted[0][0]}` (delay: {delay_sorted[0][1]} langkah)")

    st.markdown("### 🔮 Prediksi Pola Selanjutnya")
    prediksi = predict_next_pattern(freq, delay, heatmap)
    st.success(f"Prediksi Digit Potensial Berikutnya: `{prediksi}`")

    st.caption("📌 Prediksi ini berbasis statistik delay, frekuensi, dan heatmap posisi.")
