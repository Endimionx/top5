import random
from collections import defaultdict, Counter
import pandas as pd

# MARKOV DASAR
def build_transition_matrix(data):
    matrix = [defaultdict(lambda: defaultdict(int)) for _ in range(3)]
    for number in data:
        digits = f"{int(number):04d}"
        for i in range(3):
            matrix[i][digits[i]][digits[i+1]] += 1
    return matrix

def top6_markov(df, return_confidence=False):
    data = df["angka"].astype(str).tolist()
    matrix = build_transition_matrix(data)
    prediksi = [str(random.randint(0, 9))]  # digit pertama random untuk prediksi ke-2

    for i in range(3):
        prev = prediksi[-1]
        dist = matrix[i][prev]
        sorted_digits = sorted(dist.items(), key=lambda x: -x[1])
        top6 = [int(k) for k, _ in sorted_digits[:6]]
        while len(top6) < 6:
            top6.append(random.randint(0, 9))
        prediksi.append(str(top6[0]))

    hasil = []
    confidences = []

    # Posisi 1 (ribuan) pakai frekuensi langsung
    first_digits = [int(d[0]) for d in data]
    counter = Counter(first_digits)
    sorted_freq = sorted(counter.items(), key=lambda x: -x[1])
    top6_pos1 = [k for k, _ in sorted_freq[:6]]
    while len(top6_pos1) < 6:
        top6_pos1.append(random.randint(0, 9))
    hasil.append(top6_pos1)
    total_first = sum(counter.values())
    conf_first = [counter.get(d, 0)/total_first for d in range(10)]
    confidences.append(conf_first)

    # Posisi 2-4
    for i in range(3):
        prev = prediksi[i]
        dist = matrix[i][prev]
        total = sum(dist.values()) if dist else 1
        sorted_digits = sorted(dist.items(), key=lambda x: -x[1])
        top6 = [int(k) for k, _ in sorted_digits[:6]]
        while len(top6) < 6:
            top6.append(random.randint(0, 9))
        hasil.append(top6)

        conf = [dist.get(str(d), 0)/total for d in range(10)]
        confidences.append(conf)

    return (hasil, confidences) if return_confidence else hasil

# MARKOV ORDER-2
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

def top6_markov_order2(df, return_confidence=False):
    data = df["angka"].astype(str).tolist()
    matrix = build_transition_matrix_order2(data)

    pairs = df["angka"].astype(str).apply(lambda x: x[:2])
    top_pairs = pairs.value_counts().head(6).index.tolist()

    if not top_pairs:
        return [[random.randint(0, 9)] * 6 for _ in range(4)]

    d1, d2 = top_pairs[0][0], top_pairs[0][1]
    top6_d1 = list(set([int(p[0]) for p in top_pairs]))
    top6_d2 = list(set([int(p[1]) for p in top_pairs]))
    while len(top6_d1) < 6:
        top6_d1.append(random.randint(0, 9))
    while len(top6_d2) < 6:
        top6_d2.append(random.randint(0, 9))

    hasil = [top6_d1, top6_d2]
    confidences = []

    counter1 = Counter([int(p[0]) for p in pairs])
    counter2 = Counter([int(p[1]) for p in pairs])
    total1 = sum(counter1.values())
    total2 = sum(counter2.values())
    conf1 = [counter1.get(d, 0)/total1 for d in range(10)]
    conf2 = [counter2.get(d, 0)/total2 for d in range(10)]
    confidences.append(conf1)
    confidences.append(conf2)

    key1 = d1 + d2
    dist3 = matrix[0].get(key1, {})
    sorted3 = sorted(dist3.items(), key=lambda x: -x[1])
    top6_d3 = [int(k) for k, _ in sorted3[:6]]
    while len(top6_d3) < 6:
        top6_d3.append(random.randint(0, 9))
    hasil.append(top6_d3)
    total3 = sum(dist3.values()) if dist3 else 1
    conf3 = [dist3.get(str(d), 0)/total3 for d in range(10)]
    confidences.append(conf3)

    key2 = d2 + str(top6_d3[0])
    dist4 = matrix[1].get(key2, {})
    sorted4 = sorted(dist4.items(), key=lambda x: -x[1])
    top6_d4 = [int(k) for k, _ in sorted4[:6]]
    while len(top6_d4) < 6:
        top6_d4.append(random.randint(0, 9))
    hasil.append(top6_d4)
    total4 = sum(dist4.values()) if dist4 else 1
    conf4 = [dist4.get(str(d), 0)/total4 for d in range(10)]
    confidences.append(conf4)

    return (hasil, confidences) if return_confidence else hasil

# HYBRID
def top6_markov_hybrid(df, return_confidence=False):
    hasil1, conf1 = top6_markov(df, return_confidence=True)
    hasil2, conf2 = top6_markov_order2(df, return_confidence=True)

    hasil = []
    confidences = []

    for i in range(4):
        combined = hasil1[i] + hasil2[i]
        freq = Counter(combined)
        top6 = [k for k, _ in freq.most_common(6)]
        while len(top6) < 6:
            top6.append(random.randint(0, 9))
        hasil.append(top6)

        avg_conf = [(conf1[i][d] + conf2[i][d]) / 2 for d in range(10)]
        confidences.append(avg_conf)

    return (hasil, confidences) if return_confidence else hasil
