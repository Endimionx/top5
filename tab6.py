# tab6.py

import streamlit as st
from tab6_fungsi import (
    DIGIT_LABELS,
    build_lstm4d_model,
    train_lstm4d,
    prepare_lstm4d_data,
    predict_lstm4d_top8,
    parse_manual_input,
    extract_digit_patterns_from_manual_ref,
    refine_top8_with_patterns,
    save_prediction_log
)

def tab6(df, lokasi):
    st.header("📊 Prediksi 4D Akurat - Hybrid LSTM + Referensi 49x8")

    st.markdown("Masukkan 49 baris data referensi untuk masing-masing posisi (8 digit per baris):")

    manual_inputs = {}
    for label in DIGIT_LABELS:
        manual_inputs[label] = st.text_area(
            f"Referensi 49x8 - Posisi {label.upper()}",
            height=200,
            key=f"input_{label}"
        )

    run = st.button("🚀 Jalankan Prediksi", key="run_predict_tab6")

    if run:
        valid_refs = {}
        for label in DIGIT_LABELS:
            parsed = parse_manual_input(manual_inputs[label])
            if not parsed:
                st.error(f"Referensi untuk posisi {label.upper()} tidak valid. Harus 49 baris, tiap baris 8 digit.")
                return
            valid_refs[label] = parsed

        with st.spinner("Melatih model LSTM dan membuat prediksi..."):
            model = train_lstm4d(df)
            top8, full_probs = predict_lstm4d_top8(model, df)

            if not top8:
                st.error("Prediksi gagal. Pastikan jumlah data mencukupi.")
                return

            # Ambil pola referensi
            pola_refs = {label: extract_digit_patterns_from_manual_ref(valid_refs[label]) for label in DIGIT_LABELS}

            # Refine dengan hybrid pattern
            # Gabungkan pola referensi per posisi
            ref_list = [pola_refs[label][i] for i, label in enumerate(DIGIT_LABELS)]

        # Refine dengan hybrid pattern
        refined = refine_top8_with_patterns(top8, ref_list, full_probs)
        st.success("✅ Prediksi selesai!")

        # Tampilkan hasil
        for i, label in enumerate(DIGIT_LABELS):
            st.write(f"**{label.upper()}**: {refined[i]}")

        # Simpan log
        log_dict = {label: refined[i] for i, label in enumerate(DIGIT_LABELS)}
        log_file = save_prediction_log(log_dict, lokasi)
        st.info(f"📁 Hasil prediksi disimpan di: `{log_file}`")
