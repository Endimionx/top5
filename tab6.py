# tab6.py

import streamlit as st
import pandas as pd
from tab6_fungsi import (
    DIGIT_LABELS,
    build_lstm4d_model,
    train_lstm4d,
    prepare_lstm4d_data,
    predict_lstm4d_top8,
    refine_top8_with_manual,
    parse_manual_input,
    extract_manual_ref_per_digit,
    save_prediction_log
)

def tab6(df, lokasi):
    st.header("🔮 Prediksi 4D (Model LSTM per Posisi + Referensi Manual)")

    # Input parameter
    window_size = st.number_input("Window Size", min_value=5, max_value=30, value=10, key="ws6")
    epochs = st.number_input("Epochs", min_value=1, max_value=100, value=15, key="ep6")
    batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=16, key="bs6")

    st.subheader("📥 Input Prediksi Manual (49 hari + 1 hari untuk besok)")
    manual_inputs = {}
    cols = st.columns(4)
    for i, label in enumerate(DIGIT_LABELS):
        with cols[i]:
            manual_inputs[label] = st.text_area(
                f"Posisi {label.upper()} (masukkan 50 digit)",
                height=300,
                key=f"manual_{label}"
            )

    if st.button("🚀 Jalankan Prediksi", key="run_tab6"):
        try:
            with st.spinner("Melatih model dan memproses prediksi..."):
                # Latih model
                model = train_lstm4d(df, window_size=window_size, epochs=epochs, batch_size=batch_size)

                # Prediksi top-8
                top8_digits, full_probs = predict_lstm4d_top8(model, df, window_size)
                if top8_digits is None:
                    st.error("Data tidak cukup untuk prediksi.")
                    return

                # Parsing manual input & validasi
                manual_49_refs, manual_digit_besok = extract_manual_ref_per_digit(manual_inputs)

                # Gabungkan digit referensi besok sebagai pembanding refinement
                manual_refs_digit = [manual_digit_besok[pos] for pos in DIGIT_LABELS]

                # Refinement prediksi berdasarkan referensi manual
                final_result = refine_top8_with_manual(top8_digits, manual_refs_digit)

                # Tampilkan hasil
                st.subheader("📊 Hasil Prediksi Akhir per Posisi")
                pred_table = []
                for i, pos in enumerate(DIGIT_LABELS):
                    pred_table.append({
                        "Posisi": pos.upper(),
                        "Top-6": ", ".join(str(d) for d in final_result[i]),
                        "Manual Besok": manual_refs_digit[i],
                        "Match": "✅" if manual_refs_digit[i] in final_result[i] else "❌"
                    })
                st.table(pd.DataFrame(pred_table))

                # Simpan hasil ke log
                result_dict = {DIGIT_LABELS[i]: final_result[i] for i in range(4)}
                filepath = save_prediction_log(result_dict, lokasi)
                st.success(f"Hasil disimpan ke `{filepath}`")

        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")
