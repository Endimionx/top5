import numpy as np
from collections import Counter
from itertools import product


def top6_markov(df):
    transisi = [{} for _ in range(4)]
    for angka in df["angka"]:
        for i in range(4):
            if i < len(angka) - 1:
                curr_digit = int(angka[i])
                next_digit = int(angka[i + 1])
                if curr_digit not in transisi[i]:
                    transisi[i][curr_digit] = Counter()
                transisi[i][curr_digit][next_digit] += 1

    hasil = []
    for i in range(4):
        counter = Counter()
        for trans in transisi[i].values():
            counter += trans
        hasil.append([digit for digit, _ in counter.most_common(6)])
    return hasil, transisi


def top6_markov_order2(df):
    transisi = [{} for _ in range(4)]
    for angka in df["angka"]:
        for i in range(4):
            if i < len(angka) - 2:
                key = (int(angka[i]), int(angka[i + 1]))
                next_digit = int(angka[i + 2])
                if key not in transisi[i]:
                    transisi[i][key] = Counter()
                transisi[i][key][next_digit] += 1

    hasil = []
    for i in range(4):
        counter = Counter()
        for trans in transisi[i].values():
            counter += trans
        hasil.append([digit for digit, _ in counter.most_common(6)])
    return hasil


def top6_markov_hybrid(df, digit_weights=None):
    from collections import Counter

    if digit_weights is None:
        digit_weights = [1.0, 1.0, 1.0, 1.0]

    hasil = []

    for i in range(4):
        counter = Counter()

        # Ambil Markov order-1
        pred1, _ = top6_markov(df)
        for k in pred1[i]:
            counter[k] += 1 * digit_weights[i]

        # Ambil Markov order-2
        pred2 = top6_markov_order2(df)
        for k in pred2[i]:
            counter[k] += 1 * digit_weights[i]

        # Jika counter kosong, isi default [0-9]
        if not counter:
            counter = Counter({k: 1.0 for k in range(10)})

        # Ambil top 6 dari counter
        top6 = [k for k, v in counter.most_common(6)]
        hasil.append(top6)

    return hasil


def kombinasi_4d_markov_hybrid(df, top_n=10, digit_weights=None, mode="average"):
    if digit_weights is None:
        digit_weights = [1.0] * 4
    elif isinstance(digit_weights, dict):
        digit_weights = [
            digit_weights.get("ribuan", 1.0),
            digit_weights.get("ratusan", 1.0),
            digit_weights.get("puluhan", 1.0),
            digit_weights.get("satuan", 1.0),
        ]

    pred = top6_markov_hybrid(df, digit_weights=digit_weights)
    kombinasi = list(product(*pred))

    skor_dict = {}
    for komb in kombinasi:
        score = 1.0
        for i in range(4):
            weight = digit_weights[i]
            score *= (1.0 / (pred[i].index(komb[i]) + 1)) * weight if komb[i] in pred[i] else 0.01
        if mode == "average":
            score = score ** (1/4)
        skor_dict["".join(map(str, komb))] = score

    sorted_komb = sorted(skor_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return sorted_komb
