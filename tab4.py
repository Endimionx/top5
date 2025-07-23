import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

def split_digits(num_str):
    return [int(d) for d in str(num_str).zfill(4)]

def analyze_frequency(data, pos):
    digits = [split_digits(num)[pos] for num in data]
    freq = Counter(digits)
    return freq

def analyze_delay(data, pos):
    delay = {i: 0 for i in range(10)}
    last_seen = {i: -1 for i in range(10)}
    delays = []

    for idx, num in enumerate(data):
        d = split_digits(num)[pos]
        for i in range(10):
            if i == d:
                if last_seen[i] != -1:
                    delay[i] = idx - last_seen[i]
                last_seen[i] = idx
        delays.append(delay.copy())

    return delays[-1] if delays else {}

def analyze_trend(data, pos):
    trend = {'naik': 0, 'turun': 0, 'tetap': 0}
    prev = None
    for num in data:
        curr = split_digits(num)[pos]
        if prev is not None:
            if curr > prev:
                trend['naik'] += 1
            elif curr < prev:
                trend['turun'] += 1
            else:
                trend['tetap'] += 1
        prev = curr
    return trend

def even_odd_analysis(data, pos):
    counts = {'genap': 0, 'ganjil': 0}
    for num in data:
        d = split_digits(num)[pos]
        if d % 2 == 0:
            counts['genap'] += 1
        else:
            counts['ganjil'] += 1
    return counts

def big_small_analysis(data, pos):
    counts = {'besar': 0, 'kecil': 0}
    for num in data:
        d = split_digits(num)[pos]
        if d >= 5:
            counts['besar'] += 1
        else:
            counts['kecil'] += 1
    return counts

def digit_position_heatmap(data):
    pos_counts = np.zeros((4, 10))
    for num in data:
        digits = split_digits(num)
        for i, d in enumerate(digits):
            pos_counts[i][d] += 1
    return pos_counts

def zigzag_pattern(data):
    pattern = 0
    for i in range(2, len(data)):
        a = int(str(data[i-2])[-1])
        b = int(str(data[i-1])[-1])
        c = int(str(data[i])[-1])
        if (a < b > c) or (a > b < c):
            pattern += 1
    return pattern

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

    predicted_digits = list(dict.fromkeys(predicted_digits))  # unik
    return predicted_digits[:6]

def render_digit_badge(digit):
    return f"""<div style='
        background-color: #0e1117;
        color: white;
        border-radius: 50%;
        width: 48px;
        height: 48px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        box-shadow: 0 0 6px rgba(0, 255, 255, 0.4);
        margin-right: 6px;
        margin-top: 6px;
    '>{digit}</div>"""

def tab4(df):
    st.title("📊 Analisis Pola Angka 4D")

    if "angka" not in df.columns:
        st.error("❌ Kolom 'angka' tidak ditemukan di data.")
        return

    angka_data = df["angka"].dropna().astype(int).tolist()

    if not angka_data:
        st.warning("⚠️ Data 4D kosong.")
        return

    digit_pos_label = ["Ribu", "Ratus", "Puluh", "Satuan"]

    tabs = st.tabs(digit_pos_label)
    for i, tab in enumerate(tabs):
        with tab:
            st.subheader(f"📌 Posisi Digit: {digit_pos_label[i]}")

            freq = analyze_frequency(angka_data, i)
            freq_df = pd.DataFrame(freq.items(), columns=["Digit", "Frekuensi"]).sort_values("Digit")
            st.markdown("**Frekuensi Digit**")
            st.bar_chart(freq_df.set_index("Digit"))

            delay = analyze_delay(angka_data, i)
            st.markdown("**Delay Kemunculan Digit**")
            st.json(delay)

            trend = analyze_trend(angka_data, i)
            st.markdown("**Tren Naik / Turun**")
            st.json(trend)

            evenodd = even_odd_analysis(angka_data, i)
            st.markdown("**Statistik Ganjil / Genap**")
            st.json(evenodd)

            bigsmall = big_small_analysis(angka_data, i)
            st.markdown("**Statistik Besar / Kecil**")
            st.json(bigsmall)

    st.markdown("### 🔥 Heatmap Posisi Digit")
    heatmap = digit_position_heatmap(angka_data)
    fig, ax = plt.subplots(figsize=(8, 2))
    sns.heatmap(heatmap, annot=True, fmt=".0f", cmap="YlGnBu", xticklabels=list(range(10)), yticklabels=digit_pos_label, ax=ax)
    ax.set_title("Heatmap Posisi Digit")
    st.pyplot(fig)

    st.markdown("### 🔀 Pola Zigzag")
    zz = zigzag_pattern(angka_data)
    st.write(f"Zigzag pattern ditemukan: `{zz}` kali")

    st.markdown("### 🧠 Insight Otomatis")
    freq_all = analyze_frequency(angka_data, 0)
    delay_all = analyze_delay(angka_data, 0)
    most_common_digit = max(freq_all.items(), key=lambda x: x[1])[0]
    delay_sorted = sorted(delay_all.items(), key=lambda x: x[1], reverse=True)
    st.info(f"Digit paling sering muncul (ribuan): `{most_common_digit}`")
    if delay_sorted:
        st.info(f"Digit dengan delay tertinggi (ribuan): `{delay_sorted[0][0]}` (delay: {delay_sorted[0][1]} langkah)")

    st.markdown("### 🔮 Prediksi Pola Selanjutnya")
    prediksi = predict_next_pattern(freq_all, delay_all, heatmap)
    badge_html = "".join([render_digit_badge(d) for d in prediksi])
    st.markdown(f"<div style='display:flex; flex-wrap:wrap'>{badge_html}</div>", unsafe_allow_html=True)

    st.caption("📌 Prediksi ini berbasis statistik delay, frekuensi, dan heatmap posisi.")
