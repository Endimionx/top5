# Toggle cari putaran otomatis
cari_otomatis = st.toggle("🔍 Cari Putaran Otomatis", value=False)

# Slider manual hanya muncul jika toggle mati
putaran = 100
if not cari_otomatis:
    putaran = st.slider("🔁 Jumlah Putaran", 20, 1000, 100, step=10)

# Fungsi analisis putaran otomatis
def analisis_putaran_terbaik(df_all, lokasi, metode, jumlah_uji=10):
    best_score, best_n, akurasi_list = 0, 0, []
    hasil_all = {}
    for n in range(30, min(len(df_all), 200)):
        subset = df_all.tail(n).reset_index(drop=True)
        acc_total, acc_benar = 0, 0
        for i in range(min(jumlah_uji, len(subset) - 30)):
            train_df = subset.iloc[:-(jumlah_uji - i)]
            if len(train_df) < 30:
                continue
            try:
                pred = (
                    top6_markov(train_df)[0] if metode == "Markov" else
                    top6_markov_order2(train_df) if metode == "Markov Order-2" else
                    top6_markov_hybrid(train_df) if metode == "Markov Gabungan" else
                    top6_lstm(train_df, lokasi=lokasi) if metode == "LSTM AI" else
                    top6_ensemble(train_df, lokasi=lokasi)
                )
                actual = f"{int(subset.iloc[-(jumlah_uji - i)]['angka']):04d}"
                acc = sum(int(actual[j]) in pred[j] for j in range(4))
                acc_benar += acc
                acc_total += 4
            except:
                continue
        akurasi = acc_benar / acc_total * 100 if acc_total else 0
        akurasi_list.append(akurasi)
        hasil_all[n] = akurasi
        if akurasi > best_score:
            best_score = akurasi
            best_n = n
    return best_n, best_score, hasil_all
