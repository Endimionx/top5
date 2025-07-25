# tab6.py
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
    st.markdown("## 🔮 Prediksi 4D Langsung dengan Model AI")
    st.info("Model ini memprediksi seluruh angka 4D sekaligus, namun hasil tetap ditampilkan per digit.")

    # Set parameter pelatihan
    ws = st.number_input("Window Size (WS)", 5, 50, 10, key="tab6_ws")
    epoch = st.number_input("Epoch", 1, 100, 10, key="tab6_epoch")
    batch = st.number_input("Batch Size", 1, 128, 32, key="tab6_batch")

    if st.button("🚀 Latih dan Prediksi", use_container_width=True, key="tab6_btn_train_predict"):
        with st.spinner("Melatih model dan melakukan prediksi..."):
            try:
                # Siapkan data
                X, y = prepare_lstm4d_data(df, window_size=ws)
                if len(X) == 0:
                    st.error("Data tidak cukup untuk pelatihan.")
                    return

                # Latih model
                model = build_lstm4d_model(window_size=ws)
                model = train_lstm4d(model, X, y, epochs=epoch, batch_size=batch)

                # Prediksi
                top6_digits, full_probs = predict_lstm4d_top6_per_digit(model, df, window_size=ws)

                if top6_digits is None:
                    st.error("Data tidak cukup untuk prediksi.")
                    return

                st.success("✅ Prediksi selesai!")

                # Tampilkan hasil per posisi digit
                st.markdown("### 🎯 Top-6 Digit Terprediksi per Posisi")
                for i, label in enumerate(DIGIT_LABELS):
                    top6 = top6_digits[i]
                    st.write(f"**{label.upper()}**: `{top6}`")

                st.markdown("---")
                st.markdown("### 🔍 Probabilitas Lengkap per Digit")
                for i, label in enumerate(DIGIT_LABELS):
                    probs = full_probs[i]
                    df_probs = pd.DataFrame({
                        'Digit': list(range(10)),
                        'Probabilitas': [round(p, 4) for p in probs]
                    }).sort_values("Probabilitas", ascending=False).reset_index(drop=True)
                    st.write(f"**{label.upper()}**:")
                    st.table(df_probs)

            except Exception as e:
                st.error(f"Gagal prediksi: {e}")
