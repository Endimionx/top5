from collections import defaultdict, Counter
import numpy as np

def top6_markov(df):
    angka_list = [int(a) for a in df["angka"] if a.isdigit()]
    transisi = [f"{x:04d}" for x in angka_list]
    model = [defaultdict(Counter) for _ in range(4)]
    for i in range(1, len(transisi)):
        for j in range(4):
            prev = transisi[i-1][j]
            curr = transisi[i][j]
            model[j][prev][curr] += 1

    hasil = []
    for j in range(4):
        last = transisi[-1][j]
        counter = model[j][last]
        top6 = [int(k) for k, _ in counter.most_common(6)] if counter else list(range(6))
        hasil.append(top6)
    return hasil, model

def top6_markov_order2(df):
    angka_list = [int(a) for a in df["angka"] if a.isdigit()]
    transisi = [f"{x:04d}" for x in angka_list]
    model = [defaultdict(Counter) for _ in range(4)]
    for i in range(2, len(transisi)):
        for j in range(4):
            prev = transisi[i-2][j] + transisi[i-1][j]
            curr = transisi[i][j]
            model[j][prev][curr] += 1

    hasil = []
    for j in range(4):
        prev = transisi[-2][j] + transisi[-1][j]
        counter = model[j][prev]
        top6 = [int(k) for k, _ in counter.most_common(6)] if counter else list(range(6))
        hasil.append(top6)
    return hasil

def top6_markov_hybrid(df, digit_weights=None):
    basic, model1 = top6_markov(df)
    order2 = top6_markov_order2(df)
    hasil = []
    for i in range(4):
        total = basic[i] + order2[i]
        counter = {x: total.count(x) for x in set(total)}
        if digit_weights:
            for k in counter:
                counter[k] *= digit_weights[i]
        top = sorted(counter, key=lambda x: (-counter[x], x))[:6]
        hasil.append(top)
    return hasil

def kombinasi_4d_markov_hybrid(df, top_n=10, digit_weights=None, mode="average"):
    pred = top6_markov_hybrid(df, digit_weights=digit_weights)
    prob = [{} for _ in range(4)]
    for i in range(4):
        for rank, val in enumerate(pred[i]):
            prob[i][val] = (6 - rank) / 6.0  # Skor tertinggi = 1.0

    kombinasi = []
    for a in pred[0]:
        for b in pred[1]:
            for c in pred[2]:
                for d in pred[3]:
                    angka = f"{a}{b}{c}{d}"
                    if mode == "average":
                        score = (prob[0][a] + prob[1][b] + prob[2][c] + prob[3][d]) / 4
                    else:
                        score = prob[0][a] * prob[1][b] * prob[2][c] * prob[3][d]
                    kombinasi.append((angka, score))
    kombinasi = sorted(kombinasi, key=lambda x: x[1], reverse=True)[:top_n]
    return kombinasi
