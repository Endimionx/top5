# tab6.py

import streamlit as st
from tab6_fungsi import (
    DIGIT_LABELS,
    parse_reference_input,
    prepare_X_y_from_ref_and_df,
    train_digit_model,
    predict_top6,
    save_prediction_log
)

def tab6(df, lokasi):
    st.header("🎯 Tab6 - Mode B: Referensi 8-digit → Target DF")

    st.markdown("Masukkan referensi 8-digit (49 baris + 1 untuk prediksi), **masing-masing posisi digit**:")
    inputs = {}
    for label in DIGIT_LABELS:
        inputs[label] = st.text_area(f"✏️ Input Referensi {label.upper()} (8 digit × 50 baris)", height=300, key=f"ta_{label}")

    if st.button("🚀 Jalankan Prediksi", key="predict_btn_tab6"):
        hasil_prediksi = {}
        probs_all = {}
        success = True

        for i, label in enumerate(DIGIT_LABELS):
            ref_digits = parse_reference_input(inputs[label])
            if ref_digits is None:
                st.error(f"❌ Input tidak valid untuk {label.upper()} (butuh 50 baris 8-digit).")
                success = False
                continue

            X, y = prepare_X_y_from_ref_and_df(ref_digits, df, i)
            if X is None or y is None:
                st.error(f"❌ Data tidak cukup untuk posisi {label.upper()}.")
                success = False
                continue

            model = train_digit_model(X, y)
            top6, probs = predict_top6(model, ref_digits)
            hasil_prediksi[label] = top6
            probs_all[label] = probs

        if success:
            st.success("✅ Prediksi berhasil!")
            st.subheader("🔢 Hasil Prediksi Top-6 per Posisi:")
            for label in DIGIT_LABELS:
                st.write(f"**{label.upper()}**: {hasil_prediksi[label]}")

            if st.button("💾 Simpan Hasil", key="save_btn_tab6"):
                path = save_prediction_log(hasil_prediksi, lokasi)
                st.success(f"✅ Disimpan ke file: {path}")
