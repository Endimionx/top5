# tab6.py

import streamlit as st
import pandas as pd
from tab6_fungsi import (
    DIGIT_LABELS,
    train_lstm_for_position,
    predict_top8_per_position,
    parse_manual_8digit_input,
    extract_frequencies_8digit,
    refine_prediction,
    save_prediction_log
)

def tab6(df, lokasi):
    st.header("🔢 Prediksi 4D dengan Bantuan Referensi 8-Digit (Per Posisi)")
    st.info("Masukkan 49 baris referensi (masing-masing baris 8 digit) untuk setiap posisi:")

    window_size = st.number_input("Window Size", min_value=5, max_value=20, value=10, key="ws_tab6")

    col_manuals = {}
    for label in DIGIT_LABELS:
        with st.expander(f"📥 Input 8-Digit Referensi untuk Posisi {label.upper()} (49 baris, 8 digit tiap baris)"):
            col_manuals[label] = st.text_area(f"Input {label.upper()}", height=400, key=f"ta_{label}")

    if st.button("🚀 Jalankan Prediksi", key="predict_btn_tab6"):
        result_dict = {}
        for i, label in enumerate(DIGIT_LABELS):
            st.markdown(f"#### ✴️ Posisi: {label.upper()}")
            manual_data = parse_manual_8digit_input(col_manuals[label])
            if not manual_data:
                st.warning(f"Referensi posisi {label} tidak valid atau kurang dari 49 baris.")
                continue

            # Latih model
            model = train_lstm_for_position(df, i, window_size=window_size)

            # Ambil sequence digit posisi dari df
            seq = df['angka'].astype(str).apply(lambda x: int(x[i])).tolist()

            # Prediksi top8 + probabilitas
            top8, probs = predict_top8_per_position(model, seq, window_size=window_size)

            # Frekuensi digit referensi
            freqs = extract_frequencies_8digit(manual_data, i)

            # Refinement
            refined = refine_prediction(top8, probs, freqs)

            result_dict[label] = refined
            st.write(f"Top-6 Refinement ({label}):", refined)

        if result_dict:
            filename = save_prediction_log(result_dict, lokasi)
            st.success(f"✅ Hasil prediksi disimpan ke file: `{filename}`")
