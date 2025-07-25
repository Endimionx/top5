# tab6.py

import streamlit as st
import pandas as pd
from tab6_fungsi import (
    DIGIT_LABELS,
    build_lstm4d_model,
    train_lstm4d,
    prepare_lstm4d_data,
    predict_lstm4d_top8,
    parse_manual_input,
    extract_digit_patterns_from_manual_ref,
    refine_top8_with_patterns,
    save_prediction_log,
)

def tab6(df, lokasi):
    st.markdown("## 🔮 Prediksi 4D dengan Bantuan Pola Manual (8 Digit × 50 Baris)")

    st.markdown("### ✍️ Input Data Manual Per Posisi (8 digit per baris, 50 baris total)")
    manual_data = {}
    col1, col2 = st.columns(2)
    for i, pos in enumerate(DIGIT_LABELS):
        with (col1 if i < 2 else col2):
            manual_data[pos] = st.text_area(
                f"{pos.upper()} (50 baris × 8 digit)", height=400, key=f"input_manual_{pos}"
            )

    window_size = st.number_input("Window Size (LSTM)", 5, 20, 10, key="tab6_ws")
    epochs = st.number_input("Epochs", 1, 100, 15, key="tab6_epochs")
    batch_size = st.number_input("Batch Size", 1, 64, 16, key="tab6_bs")

    if st.button("🚀 Jalankan Prediksi Tab6", key="tab6_run"):
        st.info("Melatih model dan menghitung prediksi...")

        model = train_lstm4d(df, window_size, epochs=epochs, batch_size=batch_size)
        top8, probs = predict_lstm4d_top8(model, df, window_size)

        if top8 is None:
            return st.error("Data tidak cukup untuk prediksi.")

        # Parsing dan validasi semua input manual
        manual_digits = {}
        for pos in DIGIT_LABELS:
            parsed = parse_manual_input(manual_data[pos])
            if parsed is None:
                return st.error(f"Input manual posisi {pos.upper()} tidak valid. Harus 50 baris × 8 digit.")
            manual_digits[pos] = parsed

        # Ekstraksi pola dan prediksi dari baris ke-50
        pola_refs = []
        pred_besok = []
        for pos in DIGIT_LABELS:
            pola, pred = extract_digit_patterns_from_manual_ref(manual_digits[pos])
            pola_refs.append(pola)
            pred_besok.append(pred)

        # Gabungkan pola dan prediksi untuk refine
        refined = refine_top8_with_patterns(top8, pola_refs, pred_besok)

        result_dict = {DIGIT_LABELS[i]: refined[i] for i in range(4)}
        filename = save_prediction_log(result_dict, lokasi)

        st.success("✅ Prediksi Selesai dan Disimpan")
        for pos in DIGIT_LABELS:
            st.write(f"**{pos.upper()}**: {refined[DIGIT_LABELS.index(pos)]}")
        st.markdown(f"📄 Log disimpan: `{filename}`")
