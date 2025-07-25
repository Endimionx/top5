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
from collections import Counter

def tab6(df, lokasi):
    st.markdown("## 🔮 Prediksi 4D Otomatis (Model LSTM + Pola Manual 8 Digit)")

    window_size = st.slider("Window Size", 5, 20, 10, key="tab6_window")
    epochs = st.slider("Epochs", 5, 50, 15, key="tab6_epochs")
    batch_size = st.slider("Batch Size", 8, 64, 16, key="tab6_batch")

    st.markdown("### 📥 Masukkan Data Manual (8 Digit / Posisi, 50 baris)")
    col1, col2, col3, col4 = st.columns(4)
    ribuan_txt = col1.text_area("Ribuan", height=300, key="tab6_ribuan")
    ratusan_txt = col2.text_area("Ratusan", height=300, key="tab6_ratusan")
    puluhan_txt = col3.text_area("Puluhan", height=300, key="tab6_puluhan")
    satuan_txt = col4.text_area("Satuan", height=300, key="tab6_satuan")

    manual_input = {}
    for label, text in zip(DIGIT_LABELS, [ribuan_txt, ratusan_txt, puluhan_txt, satuan_txt]):
        digits = parse_manual_input(text)
        if digits:
            manual_input[label] = digits

    if len(manual_input) < 4:
        st.warning("Mohon isi 50 baris digit untuk semua posisi.")
        return

    if st.button("🔎 Prediksi Sekarang", key="tab6_prediksi"):
        with st.spinner("Melatih model dan memproses prediksi..."):
            model = train_lstm4d(df, window_size=window_size, epochs=epochs, batch_size=batch_size)
            top8, _ = predict_lstm4d_top8(model, df, window_size=window_size)

            pola_refs = {}
            prediksi_manual = []
            for pos in DIGIT_LABELS:
                pola, target = extract_digit_patterns_from_manual_ref(manual_input[pos])
                pola_refs[pos] = pola
                prediksi_manual.append(target)

            hasil_refined = refine_top8_with_patterns(top8, pola_refs, prediksi_manual)

            st.markdown("### ✅ Hasil Prediksi Final")
            result = {}
            for i, pos in enumerate(DIGIT_LABELS):
                st.write(f"**{pos.upper()}**: {hasil_refined[i]}")
                result[pos] = hasil_refined[i]

            filename = save_prediction_log(result, lokasi)
            st.success(f"✅ Prediksi disimpan ke file: `{filename}`")
