# tab6.py
import streamlit as st
import pandas as pd
from datetime import datetime
import os
from tab6_fungsi import (
    DIGIT_LABELS,
    build_lstm4d_model,
    train_lstm4d,
    prepare_lstm4d_data,
    predict_lstm4d_top6_per_digit,
    generate_all_4d_combinations,
    filter_by_reference_8digit,
    save_prediction_to_txt
)

def tab6(df, lokasi="lokasi_default"):
    st.title("🎯 Prediksi Langsung 4D (Deep Learning + Auto Save)")

    st.markdown("📌 Model ini mempelajari pola dari urutan angka 4D dan memprediksi semua digit sekaligus (ribuan, ratusan, puluhan, satuan), namun hasil tetap ditampilkan per digit.")

    window_size = st.slider("Window Size", 5, 30, 10, 1, key="tab6_ws")
    epochs = st.slider("Epochs", 1, 100, 20, 1, key="tab6_epochs")
    batch_size = st.slider("Batch Size", 8, 128, 32, 8, key="tab6_batch")

    st.markdown("### 📘 (Opsional) Referensi Data 8-Digit")
    textarea_8digit = st.text_area("Masukkan data referensi 8 digit (pisahkan per baris)", height=150, key="ref8digit")
    data_ref = [line.strip() for line in textarea_8digit.strip().splitlines() if line.strip().isdigit()] if textarea_8digit else []

    if st.button("🚀 Jalankan Prediksi 4D", key="run_tab6"):
        with st.spinner("Melatih model dan memprediksi..."):
            try:
                model = build_lstm4d_model(window_size)
                X, y = prepare_lstm4d_data(df, window_size)
                model = train_lstm4d(model, X, y, epochs=epochs, batch_size=batch_size)

                top6_per_digit, full_probs = predict_lstm4d_top6_per_digit(model, df, window_size)

                if top6_per_digit:
                    st.markdown("### 🔢 Top-6 Prediksi per Posisi")
                    for i, label in enumerate(DIGIT_LABELS):
                        st.write(f"{label.upper()}: `{top6_per_digit[i]}`")

                    all_4d = generate_all_4d_combinations(top6_per_digit)
                    filtered_4d = filter_by_reference_8digit(all_4d, data_ref)

                    st.markdown("### 🔍 Final 4D Prediction (Filtered by Referensi 8-Digit)")
                    if filtered_4d:
                        st.success(f"{len(filtered_4d)} kombinasi cocok ditemukan:")
                        preview = ", ".join(filtered_4d[:30])
                        st.code(preview + (" ..." if len(filtered_4d) > 30 else ""))

                        # ✅ Simpan otomatis
                        save_prediction_to_txt(filtered_4d, lokasi)

                    else:
                        st.warning("Tidak ada kombinasi 4D yang cocok dengan data referensi.")
                        save_prediction_to_txt(all_4d, lokasi, note="(Tanpa kecocokan referensi)")

                else:
                    st.error("Gagal melakukan prediksi.")

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
