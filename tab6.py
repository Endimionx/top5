# tab6.py

import streamlit as st
from tab6_fungsi import (
    DIGIT_LABELS,
    build_lstm4d_model,
    train_lstm4d,
    prepare_lstm4d_data,
    predict_lstm4d_top8,
    parse_manual_input,
    extract_digit_pattern_from_8digit_block,
    refine_top8_with_patterns,
    save_prediction_log
)

def tab6(df, lokasi):
    st.header("🔮 Prediksi 4D - LSTM Refinement dengan Data 8 Digit (50 Baris per Posisi)")
    st.markdown("Gunakan 8 digit per baris untuk masing-masing posisi (50 baris total, baris ke-50 opsional).")

    with st.expander("✍️ Input Prediksi Tepat 8-Digit per Posisi (49 referensi + 1 opsional prediksi besok)"):
        text_inputs = {}
        for label in DIGIT_LABELS:
            text_inputs[label] = st.text_area(
                f"Masukkan 8-digit x 50 baris untuk posisi **{label.upper()}**", 
                height=250, key=f"text_input_{label}"
            )

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        ws = st.number_input("Window Size", 5, 20, 10, key="tab6_ws")
    with col2:
        epoch = st.number_input("Epochs", 5, 100, 15, key="tab6_epoch")
    with col3:
        batch_size = st.number_input("Batch Size", 1, 128, 16, key="tab6_batch")

    if st.button("🚀 Jalankan Prediksi LSTM + Refinement", key="run_prediksi_tab6"):
        with st.spinner("Melatih model LSTM dan memproses prediksi..."):
            try:
                model = train_lstm4d(df, window_size=ws, epochs=epoch, batch_size=batch_size)
                top8, _ = predict_lstm4d_top8(model, df, window_size=ws)

                all_manual_data = {}
                for label in DIGIT_LABELS:
                    parsed = parse_manual_input(text_inputs[label])
                    if parsed is None:
                        st.error(f"❌ Format tidak valid pada posisi {label.upper()}. Harus 50 baris, 8 digit per baris.")
                        return
                    all_manual_data[label] = parsed

                # Ambil pattern frekuensi digit dari 49 baris awal
                pola_refs = []
                for label in DIGIT_LABELS:
                    pola = extract_digit_pattern_from_8digit_block(all_manual_data[label])
                    pola_refs.append(pola)

                # Refinement berdasarkan pattern referensi
                refined = refine_top8_with_patterns(top8, pola_refs)

                # Tampilkan hasil akhir
                st.subheader("🎯 Hasil Prediksi Final (Top-6 per Posisi):")
                result = {}
                for idx, label in enumerate(DIGIT_LABELS):
                    st.markdown(f"**{label.upper()}**: {refined[idx]}")
                    result[label] = refined[idx]

                file_saved = save_prediction_log(result, lokasi)
                st.success(f"Hasil prediksi disimpan di: `{file_saved}`")

            except Exception as e:
                st.error(f"Gagal prediksi: {e}")
