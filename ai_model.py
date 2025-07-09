def evaluate_lstm_accuracy_all_digits(df, lokasi, model_type="lstm", window_size=7):
    print(f"[DEBUG] Mulai evaluasi akurasi untuk lokasi: {lokasi}, model_type: {model_type}")
    
    X, y_all = preprocess_data(df, window_size=window_size)
    print(f"[DEBUG] X shape: {X.shape}")
    
    if X.shape[0] == 0 or any(y.shape[0] == 0 for y in y_all):
        print("[DEBUG] ❌ Data hasil preprocessing kosong.")
        return None, None

    acc_top1_list, acc_top6_list = [], []

    for i in range(4):
        path = f"saved_models/{lokasi.lower().replace(' ', '_')}_digit{i}_{model_type}.h5"
        print(f"[DEBUG] Cek file model: {path}")
        if not os.path.exists(path):
            print(f"[DEBUG] ❌ File model tidak ditemukan: {path}")
            return None, None
        try:
            model = load_model(path, compile=False, custom_objects={"PositionalEncoding": PositionalEncoding})
            print(f"[DEBUG] Model input shape: {model.input_shape}, X shape: {X.shape}")
            
            if model.input_shape[1] != X.shape[1]:
                print(f"[DEBUG] ❌ Input shape mismatch untuk digit {i}")
                return None, None
            
            y_true = y_all[i]
            print(f"[DEBUG] y_true shape (digit {i}): {y_true.shape}")
            
            if y_true.shape[0] != X.shape[0]:
                print(f"[DEBUG] ❌ Jumlah sample X dan y tidak cocok untuk digit {i}")
                return None, None

            try:
                acc_top1 = model.evaluate(X, y_true, verbose=0)[1]
                acc_top6 = evaluate_top6_accuracy(model, X, y_true)
                acc_top1_list.append(acc_top1)
                acc_top6_list.append(acc_top6)
                print(f"[DEBUG] Digit {i}: acc_top1={acc_top1:.4f}, acc_top6={acc_top6:.4f}")
            except Exception as e:
                print(f"[DEBUG] ❌ Gagal evaluate model digit {i}: {e}")
                return None, None

        except Exception as e:
            print(f"[DEBUG] ❌ ERROR loading model digit {i}: {e}")
            return None, None

    return acc_top1_list, acc_top6_list
