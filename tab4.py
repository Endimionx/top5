import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ... (fungsi-fungsi pendukung tetap sama seperti sebelumnya, bisa disalin ulang jika belum)

def tab4(df):
    st.title("📊 Analisis Pola Angka 4D")

    if "angka" not in df.columns:
        st.error("❌ Kolom 'angka' tidak ditemukan di data.")
        return

    angka_data = df["angka"].dropna().astype(int).tolist()
    if not angka_data:
        st.warning("⚠️ Data 4D kosong.")
        return

    run_all = st.button("🔍 Jalankan Analisis Lengkap")

    if not run_all:
        st.info("Tekan tombol di atas untuk menampilkan hasil analisis.")
        return

    digit_pos_label = ["Ribu", "Ratus", "Puluh", "Satuan"]
    tabs = st.tabs(digit_pos_label)

    for i, tab in enumerate(tabs):
        with tab:
            st.subheader(f"📌 Posisi Digit: {digit_pos_label[i]}")
            recent_data = angka_data

            freq = analyze_frequency(recent_data, i)
            freq_df = pd.DataFrame(freq.items(), columns=["Digit", "Frekuensi"]).sort_values("Digit")
            st.markdown("**📈 Frekuensi Digit (semua data)**")
            st.bar_chart(freq_df.set_index("Digit"))

            st.markdown("**⏱️ Delay Kemunculan Digit**")
            delay = analyze_delay(recent_data, i)
            render_delay(delay)

            st.markdown("**📉 Tren Naik / Turun**")
            trend = analyze_trend(recent_data, i)
            for key, val in trend.items():
                st.success(f"Jumlah tren `{key}`: `{val}`")
            st.info(f"🔮 Prediksi tren berikutnya: **{predict_trend(recent_data, i)}**")

            st.markdown("**🧮 Statistik Ganjil / Genap**")
            eo = even_odd_analysis(recent_data, i)
            for key, val in eo.items():
                st.success(f"Jumlah digit `{key}`: `{val}`")
            st.info(f"🔮 Prediksi berikutnya: **{predict_even_odd(recent_data, i)}**")

            st.markdown("**🔢 Statistik Besar / Kecil**")
            bs = big_small_analysis(recent_data, i)
            for key, val in bs.items():
                st.success(f"Jumlah digit `{key}`: `{val}`")
            st.info(f"🔮 Prediksi berikutnya: **{predict_big_small(recent_data, i)}**")

            st.markdown("**🔁 Pencarian Pola Historis**")
            pattern_len = st.slider(f"Pilih panjang pola untuk posisi {digit_pos_label[i]}", 2, 6, 3, key=f"pattern_{i}")
            pattern, matches, digits = find_historical_pattern(recent_data, i, pattern_len)
            if matches:
                st.success(f"Pola terakhir: {pattern} pernah muncul sebanyak {len(matches)} kali.")
                st.info(f"Digit setelah pola tersebut:")
                badge = "".join([render_digit_badge(d) for d in matches])
                st.markdown(f"<div style='display:flex;flex-wrap:wrap'>{badge}</div>", unsafe_allow_html=True)
            else:
                st.warning("❌ Pola terakhir belum pernah muncul sebelumnya.")

    st.markdown("### 🔥 Heatmap Posisi Digit")
    heatmap = digit_position_heatmap(angka_data)
    fig, ax = plt.subplots(figsize=(8, 2))
    sns.heatmap(heatmap, annot=True, fmt=".0f", cmap="YlGnBu",
                xticklabels=list(range(10)),
                yticklabels=digit_pos_label,
                ax=ax)
    ax.set_title("Heatmap Posisi Digit")
    st.pyplot(fig)

    st.markdown("### 🔀 Pola Zigzag")
    zz = zigzag_pattern(angka_data)
    st.success(f"Zigzag pattern ditemukan: `{zz}` kali")

    st.markdown("### 🧠 Insight Otomatis")
    freq_all = analyze_frequency(angka_data, 0)
    delay_all = analyze_delay(angka_data, 0)
    most_common_digit = max(freq_all.items(), key=lambda x: x[1])[0]
    delay_sorted = sorted(delay_all.items(), key=lambda x: x[1], reverse=True)
    st.info(f"Digit paling sering muncul (ribuan): `{most_common_digit}`")
    if delay_sorted:
        st.info(f"Digit dengan delay tertinggi (ribuan): `{delay_sorted[0][0]}` selama `{delay_sorted[0][1]}` langkah")

    st.markdown("### 🔮 Prediksi Pola Selanjutnya")
    prediksi = predict_next_pattern(freq_all, delay_all, heatmap)
    badge_html = "".join([render_digit_badge(d) for d in prediksi])
    st.markdown(f"<div style='display:flex; flex-wrap:wrap'>{badge_html}</div>", unsafe_allow_html=True)

    st.caption("📌 Prediksi berdasarkan kombinasi statistik delay, frekuensi, dan posisi digit.")
