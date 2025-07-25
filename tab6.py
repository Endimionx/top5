# tab6.py

import streamlit as st
import pandas as pd
from tab6_fungsi import (
    DIGIT_LABELS,
    train_lstm4d,
    predict_lstm4d_top8,
    parse_manual_input,
    extract_digit_patterns_from_manual_ref,
    refine_top8_with_patterns,
    save_prediction_log,
)

def tab6(df, lokasi):
    st.markdown("## 🎯 Prediksi 4D - Model LSTM + Bantuan Pola Manual")

    window_size = st.slider("Window Size (LSTM)", 5, 20, 10, key="tab6_ws")
    epochs = st.slider("Epochs", 1, 100, 15, key="tab6_epochs")
    batch_size = st.slider("Batch Size", 1, 128, 16, key="tab6_batch")

    st.markdown("### ✍️ Input Manual - Prediksi 8 Digit Per Posisi (50 baris per posisi)")
    manual_digits = {}
    valid_input = True

    col1, col2 = st.columns(2)
    for i, pos in enumerate(DIGIT_LABELS):
        with (col1 if i < 2 else col2):
            textarea = st.text_area(
                f"{pos.upper()} (50 baris angka 0-9)",
                height=300,
                key=f"manual_input_{pos}"
            )
            digits = parse_manual_input(textarea)
            if digits is None:
                st.warning(f"❌ Input untuk {pos.upper()} harus tepat 50 baris angka 0-9.")
                valid_input = False
            else:
                manual_digits[pos] = digits

    if st.button("🔮 Jalankan Prediksi Tab6", key="run_tab6_button") and valid_input:
        with st.spinner("Melatih model dan memproses prediksi..."):
            try:
                model = train_lstm4d(df, window_size=window_size, epochs=epochs, batch_size=batch_size)
                top8, _ = predict_lstm4d_top8(model, df, window_size=window_size)
                if not top8:
                    st.error("❌ Gagal membuat prediksi dari model.")
                    return

                # Ambil pola referensi & prediksi besok dari 50 baris per posisi
                pola_refs = []
                pred_besok = []
                for i, pos in enumerate(DIGIT_LABELS):
                    pola_counter, pred_digit = extract_digit_patterns_from_manual_ref(manual_digits[pos])
                    pola_refs.append(pola_counter)
                    pred_besok.append(pred_digit)

                refined = refine_top8_with_patterns(top8, pola_refs, pred_besok)

                st.success("✅ Prediksi 4D Selesai!")
                hasil = {pos: refined[i] for i, pos in enumerate(DIGIT_LABELS)}
                for pos in DIGIT_LABELS:
                    st.write(f"**{pos.upper()}**: `{hasil[pos]}`")

                filename = save_prediction_log(hasil, lokasi)
                st.info(f"📁 Disimpan otomatis ke file: `{filename}`")

            except Exception as e:
                st.error(f"❌ Gagal prediksi: {e}")
