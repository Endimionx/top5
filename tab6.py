import streamlit as st
import pandas as pd
import numpy as np
import os
from tab6_fungsi import (
    DIGIT_LABELS,
    build_lstm4d_model,
    train_lstm4d,
    prepare_lstm4d_data,
    predict_lstm4d_top6_per_digit
)

def tab6(df, lokasi):
    st.markdown("## 🔢 Prediksi Langsung 4D (Hasil Per Posisi)")
    ws = st.slider("Window Size (WS)", 5, 20, 10, key="tab6_ws")
    test_size = st.slider("Ukuran Data Test", 10, 100, 20, key="tab6_testsize")
    epochs = st.slider("Epochs", 5, 100, 20, key="tab6_epochs")
    seed = st.number_input("Seed", 0, 9999, 42, key="tab6_seed")

    if st.button("🚀 Prediksi 4D", use_container_width=True):
        try:
            # Persiapan data
            X, y = prepare_lstm4d_data(df, window_size=ws)
            X_train, y_train = X[:-test_size], y[:-test_size]

            # Training
            st.write("⏳ Melatih model...")
            model = build_lstm4d_model(input_shape=X.shape[1:])
            model = train_lstm4d(model, X_train, y_train, epochs=epochs)

            # Prediksi terhadap window terakhir
            X_last = X[-1].reshape(1, *X.shape[1:])
            top6_dict = predict_lstm4d_top6_per_digit(model, X_last)

            # Tampilkan hasil
            st.subheader("🎯 Hasil Prediksi 4D per Posisi")
            cols = st.columns(4)
            for i, pos in enumerate(DIGIT_LABELS):
                cols[i].markdown(f"**{pos.upper()}**: `{top6_dict[pos]}`")

            # Logging prediksi
            with open("log_tab6.txt", "a") as f:
                f.write(f"Lokasi: {lokasi} | WS={ws}, Epochs={epochs}, Seed={seed}\n")
                for pos in DIGIT_LABELS:
                    f.write(f"{pos.upper()}: {top6_dict[pos]}\n")
                f.write("-" * 40 + "\n")

        except Exception as e:
            st.error(f"❌ Gagal prediksi: {e}")

    # Tampilkan log
    st.markdown("---")
    col1, col2 = st.columns(2)
    if col1.button("📄 Lihat Log Tab6", use_container_width=True):
        if os.path.exists("log_tab6.txt"):
            with open("log_tab6.txt", "r") as f:
                st.code(f.read())
        else:
            st.info("Belum ada log.")
    if col2.button("🧹 Hapus Log", use_container_width=True):
        if os.path.exists("log_tab6.txt"):
            os.remove("log_tab6.txt")
            st.success("Log dihapus.")
        else:
            st.info("Tidak ada log.")
