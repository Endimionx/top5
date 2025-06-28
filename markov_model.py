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
        prediksi.append(str(top5[0]))

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

def build_transition_matrix_order2(data):
    matrix = [{} for _ in range(2)]
    for number in data:
        digits = f"{int(number):04d}"
        key1 = digits[0] + digits[1]
        key2 = digits[1] + digits[2]
        if key1 not in matrix[0]:
            matrix[0][key1] = defaultdict(int)
        if key2 not in matrix[1]:
            matrix[1][key2] = defaultdict(int)
        matrix[0][key1][digits[2]] += 1
        matrix[1][key2][digits[3]] += 1
    return matrix

def top5_markov_order2(df):
    data = df["angka"].astype(str).tolist()
    matrix = build_transition_matrix_order2(data)

    pairs = df["angka"].astype(str).apply(lambda x: x[:2])
    top_pairs = pairs.value_counts().head(5).index.tolist()

    if not top_pairs:
        return [[random.randint(0, 9)] * 5 for _ in range(4)]

    d1, d2 = top_pairs[0][0], top_pairs[0][1]
    top5_d1 = list(set([int(p[0]) for p in top_pairs]))
    top5_d2 = list(set([int(p[1]) for p in top_pairs]))
    while len(top5_d1) < 5:
        top5_d1.append(random.randint(0, 9))
    while len(top5_d2) < 5:
        top5_d2.append(random.randint(0, 9))

    hasil = [top5_d1, top5_d2]

    key1 = d1 + d2
    dist3 = matrix[0].get(key1, {})
    sorted3 = sorted(dist3.items(), key=lambda x: -x[1])
    top5_d3 = [int(k) for k, _ in sorted3[:5]]
    while len(top5_d3) < 5:
        top5_d3.append(random.randint(0, 9))
    hasil.append(top5_d3)

    key2 = d2 + str(top5_d3[0])
    dist4 = matrix[1].get(key2, {})
    sorted4 = sorted(dist4.items(), key=lambda x: -x[1])
    top5_d4 = [int(k) for k, _ in sorted4[:5]]
    while len(top5_d4) < 5:
        top5_d4.append(random.randint(0, 9))
    hasil.append(top5_d4)

    return hasil

def top5_markov_hybrid(df):
    hasil1 = top5_markov(df)
    hasil2 = top5_markov_order2(df)

    hasil = []
    for i in range(4):
        gabung = hasil1[i] + hasil2[i]
        freq = {x: gabung.count(x) for x in set(gabung)}
        top5 = sorted(freq.items(), key=lambda x: -x[1])
        top5 = [k for k, _ in top5[:5]]
        while len(top5) < 5:
            top5.append(random.randint(0, 9))
        hasil.append(top5)

    return hasil
