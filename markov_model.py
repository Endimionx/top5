
import random
from collections import defaultdict
import pandas as pd

def build_transition_matrix(data):
    matrix = [defaultdict(lambda: defaultdict(int)) for _ in range(3)]
    for number in data:
        digits = f"{int(number):04d}"
        for i in range(3):
            matrix[i][digits[i]][digits[i+1]] += 1
    return matrix

def top5_markov(df):
    data = df["angka"].astype(str).tolist()
    matrix = build_transition_matrix(data)
    prediksi = [str(random.randint(0, 9))]

    for i in range(3):
        prev = prediksi[-1]
        dist = matrix[i][prev]
        sorted_digits = sorted(dist.items(), key=lambda x: -x[1])
        top5 = [int(k) for k, v in sorted_digits[:5]]
        while len(top5) < 5:
            top5.append(random.randint(0, 9))
        prediksi.append(str(top5[0]))  # choose highest for sequence

    # kumpulkan top5 alternatif tiap posisi dari transisi
    hasil = []
    first_digits = [int(d) for d in df["angka"].astype(str).str[0]]
    top_first = sorted({i: first_digits.count(i) for i in set(first_digits)}.items(), key=lambda x: -x[1])
    top5_pos1 = [k for k, _ in top_first[:5]]
    while len(top5_pos1) < 5:
        top5_pos1.append(random.randint(0, 9))

    hasil.append(top5_pos1)
    for i in range(3):
        prev = prediksi[i]
        dist = matrix[i][prev]
        sorted_digits = sorted(dist.items(), key=lambda x: -x[1])
        top5 = [int(k) for k, v in sorted_digits[:5]]
        while len(top5) < 5:
            top5.append(random.randint(0, 9))
        hasil.append(top5)

    return hasil
