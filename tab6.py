# tab6.py

import streamlit as st
import pandas as pd

from tab6_fungsi import (
    DIGIT_LABELS,
    train_lstm4d,
    predict_lstm4d_top8,
    parse_manual_input,
    refine_top8_with_patterns,
    save_prediction_log
)

def tab6(df, lokasi):
    st.markdown("## 🔮 Prediksi 4D Langsung (Dengan Bantuan Referensi 8 Digit)")
    st.info("Masukkan masing-masing 50 digit per posisi (49 data valid sebelumnya + 1 prediksi untuk besok)")

    window_size = st.slider("Window Size (Pelatihan)", 5, 30, 10, key="tab6_ws")
    epoch = st.slider("Epoch", 1, 100, 15, key="tab6_epoch")
    batch = st.slider("Batch Size", 4, 64, 16, key="tab6_batch")

    st.markdown("### ✏️ Input Manual 8-Digit per Posisi")
    col1, col2 = st.columns(2)

    with col1:
        ribuan_txt = st.text_area("Ribuan (50 baris)", height=300, key="tab6_ribuan")
        ratusan_txt = st.text_area("Ratusan (50 baris)", height=300, key="tab6_ratusan")
    with col2:
        puluhan_txt = st.text_area("Puluhan (50 baris)", height=300, key="tab6_puluhan")
        satuan_txt = st.text_area("Satuan (50 baris)", height=300, key="tab6_satuan")

    # Parse input manual
    manual_input = {}
    valid = True
    for label, text in zip(DIGIT_LABELS, [ribuan_txt, ratusan_txt, puluhan_txt, satuan_txt]):
        digits = parse_manual_input(text)
        count = len(digits) if digits else 0
        st.caption(f"{label.upper()}: {count} baris valid")
        if count != 50:
            valid = False
        manual_input[label] = digits if digits else []

    if st.button("🔎 Prediksi Sekarang", key="prediksi_btn_tab6"):
        if not valid:
            st.error("Semua posisi harus memiliki **50 baris angka 0-9**.")
        else:
            with st.spinner("Melatih model & melakukan prediksi..."):
                model = train_lstm4d(df, window_size=window_size, epochs=epoch, batch_size=batch)
                top8_pred, _ = predict_lstm4d_top8(model, df, window_size=window_size)

                hasil_prediksi = {}
                for i, label in enumerate(DIGIT_LABELS):
                    ref_digits = manual_input[label][:49]
                    pred_digit = manual_input[label][-1]
                    refined = refine_top8_with_manual([top8_pred[i]], ref_digits, extra_score=2.0)
                    hasil_prediksi[label] = refined[0]

                st.success("✅ Prediksi berhasil disesuaikan!")
                for k, v in hasil_prediksi.items():
                    st.write(f"{k.upper()}: `{v}`")

                save_prediction_log(hasil_prediksi, lokasi)
