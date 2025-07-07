import streamlit as st

def tampilkan_user_manual():
    if "show_manual" not in st.session_state:
        st.session_state.show_manual = False

    col_manual = st.columns([1, 3, 1])
    with col_manual[1]:
        if st.button("📘 Tampilkan Panduan Pengguna"):
            st.session_state.show_manual = not st.session_state.show_manual

    if st.session_state.show_manual:
    with st.container():
        st.markdown('''
        <style>
        .manual-popup {
            position: fixed;
            top: 8%;
            left: 5%;
            width: 90%;
            height: 85%;
            background-color: #222222f0;
            color: white;
            border-radius: 15px;
            padding: 20px;
            z-index: 9999;
            overflow-y: auto;
            box-shadow: 0 0 20px rgba(0,0,0,0.7);
        }
        .manual-popup h2 {
            margin-top: 0;
        }
        </style>
        ''', unsafe_allow_html=True)

        st.markdown('<div class="manual-popup">', unsafe_allow_html=True)
        st.markdown("""
        ### 📖 PANDUAN PENGGUNA APLIKASI
        ---
        #### 1. Pengaturan Dasar
        - 🌍 **Pilih Pasaran**: Wilayah pasaran (misal: SINGAPORE, SYDNEY)
        - 📅 **Pilih Hari**: Hari data diambil (harian s/d 5 hari lalu)
        - 📊 **Data Uji Akurasi**: Jumlah data historis untuk evaluasi
        - 🧠 **Metode Prediksi**:
            - Markov: Statistik sederhana
            - Order-2: Pola 2 langkah
            - Gabungan: Kombinasi Markov
            - LSTM AI: BiLSTM + Attention + Positional Encoding
            - Ensemble: Gabungan AI + Markov

        #### 2. Cari Putaran Otomatis
        - 🔍 Toggle: Otomatis cari jumlah putaran terbaik
        - Manual: Tetap bisa gunakan slider jika toggle dimatikan
        - ✅ Info: Putaran terbaik langsung dipakai API

        #### 3. Pengambilan Data
        - API akan mengambil data sesuai hari & jumlah putaran
        - 📥 Lihat Data: Tampilkan angka 4D yang berhasil diambil

        #### 4. Manajemen Model AI (LSTM)
        - 📂 Cek Model: Status model digit 0-3
        - 📚 Latih Model: Latih model per digit
        - 🗑 Hapus: Hapus model lama jika ingin retrain
        - Fine-tune: Model lama bisa diperbarui otomatis

        #### 5. Prediksi & Kombinasi 4D
        - 🔮 Prediksi Top-6 per digit
        - 💡 Kombinasi 4D berdasarkan confidence
        - ⚡️ Confidence Score bisa diatur threshold dan power-nya

        #### 6. Evaluasi Akurasi
        - 📈 Grafik Akurasi: Per prediksi
        - 🔥 Heatmap Akurasi per digit

        #### 7. Tips Presisi
        - Gunakan Cari Otomatis untuk kestabilan prediksi
        - Latih model jika pasaran belum pernah digunakan
        - Atur confidence minimum & power untuk filter prediksi

        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.button("❌ Tutup Panduan", on_click=lambda: st.session_state.update({"show_manual": False}))
