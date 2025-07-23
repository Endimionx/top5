import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter

# Helper: Pisahkan angka 4D jadi list digit
def split_digits(num):
    return [int(d) for d in str(num).zfill(4)]

# Delay digit untuk posisi tertentu
def calculate_delay(data, pos):
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

# Tren naik-turun per posisi digit
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

# Genap/ganjil per posisi digit
def even_odd_analysis(data, pos):
    counts = {'genap': 0, 'ganjil': 0}
    for num in data:
        d = split_digits(num)[pos]
        if d % 2 == 0:
            counts['genap'] += 1
        else:
            counts['ganjil'] += 1
    return counts

# Besar/kecil per posisi digit
def big_small_analysis(data, pos):
    counts = {'besar': 0, 'kecil': 0}
    for num in data:
        d = split_digits(num)[pos]
        if d >= 5:
            counts['besar'] += 1
        else:
            counts['kecil'] += 1
    return counts

# Data dummy / ganti dengan data asli
df = pd.DataFrame({
    "angka": [1234, 5678, 4321, 8765, 1111, 9999, 2345, 6789, 3456, 7890]
})
angka_data = df["angka"].dropna().astype(int).tolist()

# Buat tab per posisi digit
positions = ["Ribuan", "Ratusan", "Puluhan", "Satuan"]
tabs = st.tabs(positions)

for i, tab in enumerate(tabs):
    with tab:
        st.subheader(f"📍 Posisi: {positions[i]}")

        # Delay
        delay = calculate_delay(angka_data, i)
        delay_df = pd.DataFrame(delay.items(), columns=["Digit", "Delay"]).sort_values("Digit")
        st.markdown("**⏱️ Delay Kemunculan Digit**")
        st.bar_chart(delay_df.set_index("Digit"))

        # Tren naik/turun
        trend = analyze_trend(angka_data, i)
        st.markdown("**📈 Tren Naik / Turun**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Naik", trend["naik"])
        col2.metric("Turun", trend["turun"])
        col3.metric("Tetap", trend["tetap"])

        # Ganjil/Genap
        evenodd = even_odd_analysis(angka_data, i)
        st.markdown("**🧮 Statistik Ganjil / Genap**")
        st.progress(evenodd["genap"] / (evenodd["genap"] + evenodd["ganjil"]))
        st.text(f"Ganjil: {evenodd['ganjil']} | Genap: {evenodd['genap']}")

        # Besar/Kecil
        bigsmall = big_small_analysis(angka_data, i)
        st.markdown("**🔢 Statistik Besar / Kecil**")
        st.progress(bigsmall["besar"] / (bigsmall["besar"] + bigsmall["kecil"]))
        st.text(f"Besar: {bigsmall['besar']} | Kecil: {bigsmall['kecil']}")
