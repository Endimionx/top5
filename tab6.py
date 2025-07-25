# tab6.py

import streamlit as st
from tab6_fungsi import (
    DIGIT_LABELS,
    build_lstm4d_model,
    train_lstm4d,
    predict_lstm4d_top8,
    parse_manual_input,
    extract_digit_patterns_flat_per_digit,
    refine_top8_with_patterns,
    save_prediction_log
)

def tab6(df, lokasi):
    st.header("🔮 Prediksi 4D (LSTM + 49 Referensi Manual Prediksi Tepat)")
    st.caption("Gunakan 49 baris referensi 8-digit (prediksi tepat) untuk mempengaruhi prediksi AI")

    col1, col2 = st.columns(2)
    with col1:
        ws = st.number_input("Window Size", 5, 20, 10, key="tab6_ws")
        epochs = st.number_input("Epochs", 1, 100, 15, key="tab6_epochs")
        batch_size = st.number_input("Batch Size", 4, 128, 16, key="tab6_batch")

    st.subheader("🧠 Referensi Prediksi Manual (49 baris x 8 digit)")
    manual_input = st.text_area(
        "Masukkan 50 baris (gunakan hanya 49 untuk referensi):",
        height=300,
        key="tab6_textarea"
    )

    if st.button("🚀 Prediksi Sekarang", key="tab6_run_button"):
        digits_50 = parse_manual_input(manual_input)
        if not digits_50:
            st.error("Format input salah. Pastikan minimal 49 baris dan tiap baris 8 digit.")
            return

        referensi = digits_50[:49]  # hanya 49 baris pertama
        st.success("✅ Referensi valid. Melatih model dan melakukan prediksi...")

        # Train dan prediksi
        model = train_lstm4d(df, window_size=ws, epochs=epochs, batch_size=batch_size)
        top8, probs = predict_lstm4d_top8(model, df, window_size=ws)
        if top8 is None:
            st.error("Data terlalu sedikit untuk melakukan prediksi.")
            return

        # Ambil pola referensi dari 49 baris manual
        pola_refs = extract_digit_patterns_flat_per_digit(referensi)
        refined = refine_top8_with_patterns(top8, pola_refs)

        st.subheader("📊 Hasil Prediksi Final (Top-6 per Posisi):")
        result_dict = {DIGIT_LABELS[i]: refined[i] for i in range(4)}
        st.table(result_dict)

        # Simpan log otomatis
        filepath = save_prediction_log(result_dict, lokasi)
        st.info(f"📁 Hasil disimpan di: `{filepath}`")
