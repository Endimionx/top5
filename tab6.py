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
    save_prediction_log
)

def tab6(df, lokasi):
    st.markdown("## 🔮 Prediksi 4D - LSTM Akurat (Per Posisi + Manual Matching)")

    # Sidebar atau input baris manual
    st.markdown("### ✍️ Input Prediksi Manual (100 Hari Terakhir + 1 Prediksi Besok)")
    cols = st.columns(4)
    textarea_inputs = {}
    valid_refs = {}

    for i, label in enumerate(DIGIT_LABELS):
        with cols[i]:
            textarea_inputs[label] = st.text_area(
                f"{label.capitalize()} (100 baris)", height=200, key=f"text_{label}"
            )
            parsed = parse_manual_input(textarea_inputs[label])
            if parsed:
                valid_refs[label] = parsed
            else:
                st.warning(f"❗ Data {label} tidak valid atau tidak 100 baris.")

    st.markdown("---")
    window_size = st.number_input("🪟 Window Size", 5, 30, 10, key="ws_tab6")
    epochs = st.number_input("🧠 Epochs", 1, 100, 15, key="ep_tab6")
    batch_size = st.number_input("📦 Batch Size", 4, 64, 16, key="bs_tab6")

    if st.button("🚀 Jalankan Prediksi", use_container_width=True, key="run_pred_tab6"):
        try:
            st.info("🔧 Melatih model sementara...")
            model = train_lstm4d(df, window_size=window_size, epochs=epochs, batch_size=batch_size)

            st.info("🔮 Melakukan prediksi 4D (top-8 per posisi)...")
            top8_per_pos, _ = predict_lstm4d_top8(model, df, window_size)

            if not top8_per_pos:
                return st.error("Gagal memprediksi. Pastikan data cukup.")

            st.subheader("📈 Hasil Prediksi Awal (Top-8 per Posisi)")
            for i, label in enumerate(DIGIT_LABELS):
                st.write(f"{label.upper()}: {top8_per_pos[i]}")

            if len(valid_refs) == 4:
                # Gunakan baris ke-100 (baris ke-99, index ke-99) sebagai prediksi besok
                manual_refs = [valid_refs[label][99] for label in DIGIT_LABELS]
                refined = refine_top8_with_manual(top8_per_pos, manual_refs, extra_score=2.0)

                st.subheader("✅ Refinement dengan Referensi Manual (Top-6)")
                for i, label in enumerate(DIGIT_LABELS):
                    st.success(f"{label.upper()}: {refined[i]} (ref: {manual_refs[i]})")

                # Logging
                log_file = save_prediction_log(dict(zip(DIGIT_LABELS, refined)), lokasi)
                st.info(f"📁 Hasil prediksi disimpan ke: `{log_file}`")

            else:
                st.warning("📄 Input referensi manual belum lengkap atau valid. Refinement dilewati.")
                for i, label in enumerate(DIGIT_LABELS):
                    st.write(f"{label.upper()}: {top8_per_pos[i][:6]} (tanpa refinement)")

        except Exception as e:
            st.error(f"Gagal prediksi: {e}")
