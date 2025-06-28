
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

    first_digits = [int(d) for d in df["angka"].astype(str).str[0]]
    top1 = sorted({i: first_digits.count(i) for i in set(first_digits)}.items(), key=lambda x: -x[1])
    top5_d1 = [k for k, _ in top1[:5]]
    while len(top5_d1) < 5:
        top5_d1.append(random.randint(0, 9))

    second_digits = [int(d[1]) for d in df["angka"].astype(str)]
    top2 = sorted({i: second_digits.count(i) for i in set(second_digits)}.items(), key=lambda x: -x[1])
    top5_d2 = [k for k, _ in top2[:5]]
    while len(top5_d2) < 5:
        top5_d2.append(random.randint(0, 9))

    d1 = str(top5_d1[0])
    d2 = str(top5_d2[0])
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
