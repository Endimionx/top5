import streamlit as st
from tab6_fungsi import (
    parse_reference_input,
    extract_target_from_df,
    train_model_per_posisi,
    predict_last_row,
    DIGIT_LABELS,
    save_prediction_log
)

def tab6(df, selected_lokasi):
    st.header("📊 Prediksi 4D - Mode B (Referensi Jadi Model)")

    st.markdown("Masukkan data referensi 8 digit untuk masing-masing posisi. Harus 50 baris, masing-masing 8 digit.")

    text_areas = {}
    for label in DIGIT_LABELS:
        text_areas[label] = st.text_area(f"📥 Data Referensi 8 Digit - {label.capitalize()}",
                                         height=300, key=f"ref_{label}")

    if st.button("🔮 Jalankan Prediksi", key="run_prediksi_modeB"):
        all_valid = True
        references = {}

        for label in DIGIT_LABELS:
            parsed = parse_reference_input(text_areas[label])
            if parsed is None:
                st.error(f"Format referensi {label} tidak valid. Harus 50 baris, masing-masing 8 digit.")
                all_valid = False
            else:
                references[label] = parsed

        if not all_valid:
            return

        if len(df) < 49:
            st.error("Data utama (df) kurang dari 49 baris.")
            return

        pred_result = {}
        for idx, label in enumerate(DIGIT_LABELS):
            X = references[label][:49]  # 49 baris pertama
            y = extract_target_from_df(df, idx)
            model = train_model_per_posisi(X, y)
            pred_digit = predict_last_row(references[label], model)
            pred_result[label] = pred_digit

        st.success("✅ Prediksi Selesai!")
        st.write("### Hasil Prediksi Per Posisi:")
        for label in DIGIT_LABELS:
            st.write(f"**{label.capitalize()}**: {pred_result[label]}")

        filename = save_prediction_log(pred_result, selected_lokasi)
        st.info(f"Hasil disimpan di: `{filename}`")
