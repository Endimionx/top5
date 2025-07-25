# tab6.py

import streamlit as st
from tab6_fungsi import (
    DIGIT_LABELS,
    parse_reference_input,
    get_target_digit_from_df,
    train_and_predict_top6,
    save_prediction_log
)

def tab6(df, lokasi):
    st.header("📊 Prediksi 4D Berdasarkan Referensi Per Posisi")

    st.markdown("Masukkan 49 baris referensi 8-digit yang masing-masing **berasal dari 1 posisi**:")

    input_refs = {}
    for label in DIGIT_LABELS:
        input_refs[label] = st.text_area(
            f"Referensi 8 Digit - {label.capitalize()}",
            height=300,
            key=f"input_{label}"
        )

    if st.button("🔮 Prediksi", key="predict_button"):
        all_refs = {}
        for label in DIGIT_LABELS:
            parsed = parse_reference_input(input_refs[label])
            if parsed is None:
                st.error(f"Referensi {label} tidak valid. Pastikan 49 baris × 8 digit.")
                return
            all_refs[label] = parsed

        hasil = {}
        full_probs = {}

        for idx, label in enumerate(DIGIT_LABELS):
            target_digit = get_target_digit_from_df(df, idx)
            if target_digit is None:
                st.error("Data target tidak valid.")
                return
            top6, probs = train_and_predict_top6(all_refs[label], target_digit)
            hasil[label] = top6
            full_probs[label] = probs

        st.subheader("✅ Hasil Prediksi Top-6 per Posisi")
        for label in DIGIT_LABELS:
            st.write(f"**{label.capitalize()}**: {hasil[label]}")

        st.subheader("📄 Simpan ke Log")
        if st.button("💾 Simpan Log"):
            filename = save_prediction_log(hasil, lokasi)
            st.success(f"Disimpan ke: {filename}")
