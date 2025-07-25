import streamlit as st
import pandas as pd
from tab6_fungsi import (
    DIGIT_LABELS,
    build_lstm4d_model,
    train_lstm4d,
    prepare_lstm4d_data,
    predict_lstm4d_top6_per_digit
)

def tab6(df, lokasi):
    st.markdown("## 🎯 Prediksi 4D Langsung (Hasil Per Posisi)")

    col1, col2 = st.columns(2)
    ws = col1.slider("Window Size", 3, 20, 10, key="ws6")
    epoch = col2.number_input("Epoch", 1, 100, 10, key="ep6")
    batch = col1.number_input("Batch Size", 1, 128, 32, key="batch6")

    if len(df) < ws + 1:
        st.warning("Data tidak cukup untuk prediksi.")
        return

    if st.button("🔮 Jalankan Prediksi 4D", key="run_tab6"):
        try:
            X, y = prepare_lstm4d_data(df, window_size=ws)
            model = build_lstm4d_model(ws)
            model = train_lstm4d(model, X, y, epoch, batch)
            top6, full_probs = predict_lstm4d_top6_per_digit(model, df, ws)

            if top6 is None:
                st.error("Data tidak cukup untuk prediksi.")
                return

            st.markdown("### ✅ Hasil Prediksi Top-6 per Digit")
            for i, label in enumerate(DIGIT_LABELS):
                top_digits = top6[i]
                st.write(f"**{label.upper()}**: `{top_digits}`")

            st.markdown("### 📊 Probabilitas Lengkap per Digit")
            for i, label in enumerate(DIGIT_LABELS):
                prob_row = full_probs[i]
                prob_dict = {str(j): round(p, 3) for j, p in enumerate(prob_row)}
                st.write(f"{label.upper()}:")
                st.json(prob_dict)

        except Exception as e:
            st.error(f"Gagal prediksi: {e}")
