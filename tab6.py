# tab6.py
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os

from tab6_fungsi import (
    DIGIT_LABELS,
    build_lstm4d_model,
    train_lstm4d,
    prepare_lstm4d_data,
    predict_lstm4d_top6_per_digit,
    match_top_with_reference,
)

def tab6(df, lokasi):
    st.markdown("## 🔮 Prediksi 4D Langsung per Posisi")
    st.info("Model akan mempelajari data angka untuk memprediksi digit berikutnya.")

    window_size = st.slider("Window Size", 3, 20, 10, key="tab6_ws")
    epochs = st.slider("Epochs", 10, 200, 50, step=10, key="tab6_epochs")
    batch_size = st.slider("Batch Size", 4, 64, 16, step=4, key="tab6_batch")

    # Input referensi prediksi lain
    st.markdown("### 📘 Referensi Prediksi per Posisi")
    referensi_digit = {}
    cols = st.columns(4)
    for i, pos in enumerate(DIGIT_LABELS):
        with cols[i]:
            textarea = st.text_area(f"{pos.upper()} (Referensi)", height=150, key=f"ref_{pos}")
            lines = [int(line.strip()) for line in textarea.strip().splitlines() if line.strip().isdigit()]
            referensi_digit[pos] = lines

    if st.button("🚀 Jalankan Prediksi 4D", key="tab6_run"):
        try:
            X, y = prepare_lstm4d_data(df, window_size)
            model = build_lstm4d_model(window_size)
            model = train_lstm4d(model, X, y, epochs=epochs, batch_size=batch_size)

            top8_per_digit, probs = predict_lstm4d_top6_per_digit(model, df, window_size, top_k=8)
            result = []

            for i, pos in enumerate(DIGIT_LABELS):
                top8 = top8_per_digit[i]
                ref = referensi_digit.get(pos, [])
                matched = match_top_with_reference(top8, ref)
                result.append({
                    "Posisi": pos,
                    "Top-8": ", ".join(str(d) for d in top8),
                    "Referensi": ", ".join(str(r) for r in ref),
                    "Match": "✅" if matched else "❌"
                })

            df_result = pd.DataFrame(result)
            st.markdown("### 🎯 Hasil Prediksi dan Pencocokan")
            st.table(df_result)

            # Simpan otomatis ke file log berdasarkan lokasi & tanggal
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            filename = f"prediksi_tab6_{lokasi}_{now}.txt"
            with open(filename, "w") as f:
                f.write(f"📍 Lokasi: {lokasi} | Waktu: {now}\n")
                for row in result:
                    f.write(f"{row['Posisi'].upper()}: Top-8={row['Top-8']} | Referensi={row['Referensi']} | Match={row['Match']}\n")
            st.success(f"Hasil disimpan ke `{filename}`")

        except Exception as e:
            st.error(f"Gagal prediksi: {e}")
