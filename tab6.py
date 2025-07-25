# tab6.py
import streamlit as st
import pandas as pd
from datetime import datetime
import os
from tab6_fungsi import (
    DIGIT_LABELS,
    build_lstm4d_model,
    train_lstm4d,
    prepare_lstm4d_data,
    predict_lstm4d_topk_per_digit,
    generate_all_4d_combinations_with_probs,
    filter_and_rank_by_reference,
    save_prediction_to_txt
)

def tab6(df, lokasi="lokasi_default"):
    st.title("🎯 Prediksi Langsung 4D (Top-8 Match Ref, Rank by Bobot)")

    window_size = st.slider("Window Size", 5, 30, 10, 1, key="tab6_ws")
    epochs = st.slider("Epochs", 1, 100, 20, 1, key="tab6_epochs")
    batch_size = st.slider("Batch Size", 8, 128, 32, 8, key="tab6_batch")

    st.markdown("### 📘 Referensi Prediksi per Posisi")
    referensi_digit = {}
    cols = st.columns(4)
    for i, pos in enumerate(DIGIT_LABELS):
        with cols[i]:
            textarea = st.text_area(f"{pos.upper()}", height=150, key=f"ref_{pos}")
            lines = [int(line.strip()) for line in textarea.strip().splitlines() if line.strip().isdigit()]
            referensi_digit[pos] = lines

    if st.button("🚀 Jalankan Prediksi", key="tab6_run"):
        with st.spinner("Melatih model dan memprediksi..."):
            try:
                model = build_lstm4d_model(window_size)
                X, y = prepare_lstm4d_data(df, window_size)
                model = train_lstm4d(model, X, y, epochs=epochs, batch_size=batch_size)

                topk_per_digit, full_probs = predict_lstm4d_topk_per_digit(model, df, window_size, top_k=8)

                st.markdown("### 🔢 Top-8 Prediksi per Posisi")
                for i, label in enumerate(DIGIT_LABELS):
                    st.write(f"{label.upper()}: `{topk_per_digit[i]}`")

                all_4d = generate_all_4d_combinations_with_probs(topk_per_digit, full_probs)
                filtered_sorted_4d = filter_and_rank_by_reference(all_4d, referensi_digit)

                st.markdown("### 🧠 Final Prediksi 4D (Disaring dan Diurutkan Bobot)")
                if filtered_sorted_4d:
                    st.success(f"{len(filtered_sorted_4d)} kombinasi cocok ditemukan:")
                    preview = "\n".join([f"{item[0]} | Skor: {item[1]:.4f}" for item in filtered_sorted_4d[:30]])
                    st.code(preview + ("\n..." if len(filtered_sorted_4d) > 30 else ""))
                    save_prediction_to_txt([x[0] for x in filtered_sorted_4d], lokasi)
                else:
                    st.warning("Tidak ada kombinasi 4D yang cocok dengan referensi posisi.")
                    save_prediction_to_txt([x[0] for x in all_4d], lokasi, note="(Tanpa kecocokan referensi posisi)")

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
