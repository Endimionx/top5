import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, Concatenate, Permute, Multiply, Lambda, RepeatVector
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
import tensorflow.keras.backend as K

# ==================== Data Prep ====================
def prepare_lstm_data(df):
    data = df['angka'].dropna().apply(lambda x: [int(d) for d in f"{int(x):04d}"]).tolist()
    data = np.array(data)

    if len(data) < 11:
        return None, None

    X, y = [], []
    for i in range(len(data) - 10):
        X.append(data[i:i+10])
        y.append(data[i+10])
    X, y = np.array(X), np.array(y)

    if len(X) == 0 or len(y) == 0:
        return None, None

    y_encoded = [to_categorical(y[:, i], num_classes=10) for i in range(4)]
    return X, y_encoded

def model_exists(lokasi):
    filename = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"
    return os.path.exists(filename)

# ==================== Attention Layer ====================
def attention_block(inputs):
    attention = Dense(inputs.shape[-1], activation='tanh')(inputs)
    attention = Dense(1, activation='softmax')(attention)
    attention = Lambda(lambda x: K.squeeze(x, -1))(attention)
    attention = RepeatVector(inputs.shape[-1])(attention)
    attention = Permute([2, 1])(attention)
    output = Multiply()([inputs, attention])
    return output

# ==================== Training ====================
def train_and_save_lstm(df, lokasi):
    os.makedirs("saved_models", exist_ok=True)
    filename = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"
    history_file = f"saved_models/history_{lokasi.lower().replace(' ', '_')}.csv"

    X, y = prepare_lstm_data(df)
    if X is None or y is None:
        return

    inputs = Input(shape=(10, 4))
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = attention_block(x)
    x = LSTM(32)(x)

    outputs = [Dense(10, activation='softmax', name=f'output_{i}')(x) for i in range(4)]
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='loss', patience=3, factor=0.5, verbose=1),
        CSVLogger(history_file, append=True)
    ]

    model.fit(X, y, epochs=50, batch_size=16, verbose=0, callbacks=callbacks)
    model.save(filename)

# ==================== Prediction ====================
def top6_lstm(df, lokasi=None, return_confidence=False):
    filename = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"
    if not os.path.exists(filename):
        return None

    data = df['angka'].dropna().apply(lambda x: [int(d) for d in f"{int(x):04d}"]).tolist()
    data = np.array(data)
    if len(data) < 10:
        return None

    model = load_model(filename, compile=False)
    input_seq = np.array(data[-10:]).reshape(1, 10, 4)
    preds = model.predict(input_seq, verbose=0)

    top6 = []
    confidences = []
    for i in range(4):
        prob = preds[i][0]
        idx = np.argsort(-prob)[:6]
        top6.append(list(idx))
        confidences.append([round(prob[j], 3) for j in idx])

    if return_confidence:
        return top6, confidences
    return top6

def anti_top6_lstm(df, lokasi=None):
    filename = f"saved_models/lstm_{lokasi.lower().replace(' ', '_')}.h5"
    if not os.path.exists(filename):
        return None

    data = df['angka'].dropna().apply(lambda x: [int(d) for d in f"{int(x):04d}"]).tolist()
    data = np.array(data)
    if len(data) < 10:
        return None

    model = load_model(filename, compile=False)
    input_seq = np.array(data[-10:]).reshape(1, 10, 4)
    preds = model.predict(input_seq, verbose=0)

    anti_top6 = []
    for i in range(4):
        prob = preds[i][0]
        idx = np.argsort(prob)[:6]
        anti_top6.append(list(idx))
    return anti_top6

def low6_lstm(df, lokasi=None):
    return anti_top6_lstm(df, lokasi)

# ==================== Evaluate Akurasi per Digit ====================
def evaluate_per_digit_accuracy(df, lokasi, jumlah_uji=10):
    if len(df) < 10 + jumlah_uji:
        return [0, 0, 0, 0]

    total_digit = [0, 0, 0, 0]
    benar_digit = [0, 0, 0, 0]

    for i in range(jumlah_uji):
        train_df = df.iloc[:-(jumlah_uji - i)]
        test_row = df.iloc[-(jumlah_uji - i)]
        pred = top6_lstm(train_df, lokasi=lokasi)
        if pred is None:
            continue
        actual = [int(d) for d in f"{int(test_row['angka']):04d}"]
        for j in range(4):
            total_digit[j] += 1
            if actual[j] in pred[j]:
                benar_digit[j] += 1

    akurasi_per_digit = [
        round(100 * benar_digit[i] / total_digit[i], 2) if total_digit[i] > 0 else 0
        for i in range(4)
    ]
    return akurasi_per_digit

# ==================== Visualisasi Confidence ====================
def plot_confidence_bar(confidences, digit_label):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    plt.figure(figsize=(6, 3))
    digits = list(range(10))
    plt.bar(digits, confidences, color=sns.color_palette("Blues", 10))
    plt.title(f"Confidence Output - {digit_label}")
    plt.xlabel("Digit")
    plt.ylabel("Probabilitas")
    plt.xticks(digits)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

# Optional: Call `plot_confidence_bar(pred[0], "Ribuan")` from Streamlit if needed
