# tab6.py

import streamlit as st
from tab6_fungsi import (
    DIGIT_LABELS,
    build_lstm4d_model,
    train_lstm4d,
    predict_lstm4d_top8,
    parse_manual_input,
    extract_digit_patterns_from_manual_ref,
    refine_top8_with_patterns,
    save_prediction_log
)

def tab6(df, selected_lokasi):
    st.header("📘 Prediksi 4D LSTM + Referensi Pola Manual")

    st.markdown("### ✍️ Masukkan Data Referensi Manual (8 digit, 50 baris, per posisi)")
    
    col1, col2, col3, col4 = st.columns(4)
    ribuan_input = col1.text_area("Ribuan", height=300, key="ribuan_manual")
    ratusan_input = col2.text_area("Ratusan", height=300, key="ratusan_manual")
    puluhan_input = col3.text_area("Puluhan", height=300, key="puluhan_manual")
    satuan_input = col4.text_area("Satuan", height=300, key="satuan_manual")

    st.markdown("### 🔧 Parameter Model")
    window_size = st.number_input("Window Size", min_value=5, max_value=20, value=10, step=1, key="ws_tab6")
    epochs = st.slider("Epochs", 1, 50, 15, key="epochs_tab6")
    batch_size = st.slider("Batch Size", 1, 64, 16, key="batch_tab6")

    if st.button("🚀 Jalankan Prediksi", key="run_tab6"):
        try:
            parsed_ribuan = parse_manual_input(ribuan_input)
            parsed_ratusan = parse_manual_input(ratusan_input)
            parsed_puluhan = parse_manual_input(puluhan_input)
            parsed_satuan = parse_manual_input(satuan_input)

            if None in (parsed_ribuan, parsed_ratusan, parsed_puluhan, parsed_satuan):
                st.error("❌ Semua posisi harus berisi 50 baris valid (8 digit per baris).")
                return

            # Gabungkan ke format referensi: list of 50 baris, tiap baris 8 digit
            referensi_gabungan = [
                [parsed_ribuan[i][0], parsed_ratusan[i][1], parsed_puluhan[i][2], parsed_satuan[i][3]]
                for i in range(50)
            ]

            pola_refs, pred_besok = extract_digit_patterns_from_manual_ref(referensi_gabungan)

            # Latih model dari data utama
            model = train_lstm4d(df, window_size=window_size, epochs=epochs, batch_size=batch_size)

            top8, full_probs = predict_lstm4d_top8(model, df, window_size=window_size)

            if top8 is None:
                st.warning("❗ Jumlah data tidak cukup untuk window size.")
                return

            refined = refine_top8_with_patterns(top8, pola_refs, pred_besok)

            st.success("✅ Prediksi berhasil dibuat:")
            for i, label in enumerate(DIGIT_LABELS):
                st.write(f"**{label.upper()}**: {refined[i]}")

            log_dict = {label: refined[i] for i, label in enumerate(DIGIT_LABELS)}
            file_path = save_prediction_log(log_dict, selected_lokasi)
            st.info(f"📁 Hasil disimpan di: `{file_path}`")

        except Exception as e:
            st.error(f"❌ Gagal prediksi: {e}")
