# tab6.py

import streamlit as st
from tab6_fungsi import (
    DIGIT_LABELS,
    build_lstm4d_model,
    train_lstm4d,
    predict_lstm4d_top8,
    parse_manual_input,
    extract_digit_patterns_from_manual_ref,
    refine_top8_with_patterns,
    save_prediction_log
)

def tab6(df, lokasi):
    st.header("📊 Prediksi 4D - Refined with Manual 8-Digit Referensi")

    with st.expander("✍️ Masukkan Data Referensi (49 baris 8-digit per posisi)"):
        manual_inputs = {}
        for label in DIGIT_LABELS:
            text = st.text_area(f"Referensi 8 Digit - {label.capitalize()} (49 baris, 8 digit per baris)", key=f"ref_input_{label}")
            parsed = parse_manual_input(text)
            if parsed:
                manual_inputs[label] = parsed
            else:
                st.warning(f"Referensi {label} belum valid (pastikan 49 baris & 8 digit per baris).")

    col1, col2 = st.columns(2)
    with col1:
        ws = st.number_input("Window Size", 5, 20, 10, key="ws_tab6")
        epochs = st.number_input("Epochs", 1, 100, 15, key="epochs_tab6")
    with col2:
        batch_size = st.number_input("Batch Size", 4, 128, 16, key="bs_tab6")
        run_pred = st.button("🔮 Jalankan Prediksi", key="run_tab6")

    if run_pred:
        if len(manual_inputs) != 4:
            st.error("Pastikan semua posisi memiliki referensi 49 baris valid.")
            return

        try:
            with st.spinner("Melatih model dan memproses prediksi..."):
                model = train_lstm4d(df, window_size=ws, epochs=epochs, batch_size=batch_size)
                top8, full_probs = predict_lstm4d_top8(model, df, window_size=ws)

                pola_refs = {label: extract_digit_patterns_from_manual_ref(manual_inputs[label]) for label in DIGIT_LABELS}
                refined = refine_top8_with_patterns(top8, [pola_refs[label] for label in DIGIT_LABELS])

                st.subheader("✅ Hasil Prediksi (Top-6 Per Posisi)")
                result_dict = {}
                for i, label in enumerate(DIGIT_LABELS):
                    st.markdown(f"**{label.capitalize()}:** {refined[i]}")
                    result_dict[label] = refined[i]

                file_path = save_prediction_log(result_dict, lokasi)
                st.success(f"Hasil disimpan ke: `{file_path}`")

        except Exception as e:
            st.error(f"Gagal prediksi: {e}")
