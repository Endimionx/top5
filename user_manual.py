import streamlit as st

def tampilkan_user_manual():
    if "show_manual" not in st.session_state:
        st.session_state.show_manual = False

    col_manual = st.columns([1, 3, 1])
    with col_manual[1]:
        if st.button("📘 Tampilkan Panduan Pengguna"):
            st.session_state.show_manual = not st.session_state.show_manual

    if st.session_state.show_manual:
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
        .manual-popup button {
            background-color: #e74c3c;
            border: none;
            padding: 10px 20px;
            border-radius: 10px;
            color: white;
            cursor: pointer;
        }
        </style>

        <div class="manual-popup">
        <h2>📖 PANDUAN PENGGUNA APLIKASI</h2>
        <hr>

        <h3>1. Pengaturan Dasar</h3>
        <ul>
            <li><b>🌍 Pilih Pasaran:</b> Tentukan wilayah pasaran (misalnya: SINGAPORE, SYDNEY, dll)</li>
            <li><b>📅 Pilih Hari:</b> Data berdasarkan hari (harian, kemarin, hingga 5 hari lalu)</li>
            <li><b>📊 Data Uji Akurasi:</b> Jumlah data historis yang digunakan untuk menghitung akurasi prediksi</li>
            <li><b>🧠 Metode Prediksi:</b> 
                <ul>
                    <li>Markov: Model statistik sederhana</li>
                    <li>Markov Order-2: Pola 2 langkah</li>
                    <li>Markov Gabungan: Gabungan model Markov</li>
                    <li>LSTM AI: Model AI canggih (BiLSTM + Attention + PE)</li>
                    <li>Ensemble AI + Markov: Gabungan hasil prediksi LSTM & Markov</li>
                </ul>
            </li>
        </ul>

        <h3>2. Cari Putaran Otomatis</h3>
        <ul>
            <li><b>🔍 Toggle:</b> Aktifkan untuk mencari jumlah putaran terbaik secara otomatis berdasarkan akurasi historis</li>
            <li><b>Slider Manual:</b> Tetap tersedia jika ingin menentukan jumlah putaran secara manual</li>
            <li><b>✅ Info:</b> Putaran terbaik akan otomatis digunakan untuk mengambil data dari API</li>
        </ul>

        <h3>3. Pengambilan Data</h3>
        <ul>
            <li>✅ Data diambil dari API berdasarkan jumlah putaran dan hari</li>
            <li>📥 Lihat Data: Menampilkan seluruh angka 4D hasil pengambilan data</li>
        </ul>

        <h3>4. Manajemen Model AI (LSTM)</h3>
        <ul>
            <li><b>📂 Cek Model:</b> Menampilkan status model untuk setiap digit (0-3)</li>
            <li><b>📚 Latih Model:</b> Melatih model per digit jika belum tersedia</li>
            <li><b>🗑 Hapus:</b> Hapus model digit tertentu jika ingin retrain</li>
            <li><b>Auto Fine-tune:</b> Sistem otomatis fine-tune jika model lama tersedia</li>
        </ul>

        <h3>5. Prediksi</h3>
        <ul>
            <li><b>🔮 Tombol Prediksi:</b> Menjalankan model dan menampilkan hasil prediksi Top-6 per digit</li>
            <li><b>💡 Kombinasi 4D:</b> Disediakan jika metode AI/Ensemble digunakan</li>
            <li><b>⚡️ Confidence Score:</b> Muncul untuk setiap kombinasi hasil AI (bisa pakai power dan minimum threshold)</li>
        </ul>

        <h3>6. Evaluasi Akurasi</h3>
        <ul>
            <li><b>📈 Grafik Akurasi:</b> Menampilkan akurasi per prediksi terhadap data uji</li>
            <li><b>🔥 Heatmap:</b> Menampilkan akurasi tiap digit secara visual</li>
        </ul>

        <h3>7. Tips Presisi</h3>
        <ul>
            <li>Gunakan Cari Putaran Otomatis jika ingin hasil prediksi lebih stabil</li>
            <li>Latih model dulu jika pertama kali menggunakan pasaran tertentu</li>
            <li>Confidence Score bisa diatur untuk menyaring kombinasi paling kuat</li>
        </ul>

        <div style='text-align:center; margin-top:20px'>
            <button onclick="window.location.reload()">❌ Tutup Panduan</button>
        </div>
        </div>
        ''', unsafe_allow_html=True)
