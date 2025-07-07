import random
from collections import defaultdict, Counter
import pandas as pd

# MARKOV ORDER-1 (Probabilitas Transisi)
def build_transition_matrix(data):
    matrix = [defaultdict(lambda: defaultdict(int)) for _ in range(3)]
    for number in data:
        digits = f"{int(number):04d}"
        for i in range(3):
            matrix[i][digits[i]][digits[i+1]] += 1
    return matrix

def top6_markov(df):
    data = df["angka"].astype(str).tolist()
    matrix = build_transition_matrix(data)

    freq_ribuan = Counter([int(x[0]) for x in data])
    hasil = []

    # Ribuan
    top6_ribuan = [k for k, _ in freq_ribuan.most_common(6)]
    while len(top6_ribuan) < 6:
        top6_ribuan.append(random.randint(0, 9))
    hasil.append(top6_ribuan)

    # Posisi 2-4: Probabilitas transisi
    for i in range(3):
        total_counts = Counter()
        for prev_digit, next_digits in matrix[i].items():
            total_counts.update(next_digits)
        top6 = [int(k) for k, _ in total_counts.most_common(6)]
        while len(top6) < 6:
            top6.append(random.randint(0, 9))
        hasil.append(top6)

    return hasil, {
        "frekuensi_ribuan": dict(freq_ribuan),
        "transisi": [{k: dict(v) for k, v in m.items()} for m in matrix],
    }

# MARKOV ORDER-2 (Transisi 2-digit → 1-digit)
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

def top6_markov_order2(df):
    data = df["angka"].astype(str).tolist()
    matrix = build_transition_matrix_order2(data)

    pairs = [x[:2] for x in data]
    top_pairs = Counter(pairs).most_common(6)
    if not top_pairs:
        return [[random.randint(0, 9) for _ in range(6)] for _ in range(4)]

    d1, d2 = top_pairs[0][0][0], top_pairs[0][0][1]

    top6_d1 = list(set(int(p[0][0]) for p in top_pairs))
    top6_d2 = list(set(int(p[0][1]) for p in top_pairs))
    while len(top6_d1) < 6:
        top6_d1.append(random.randint(0, 9))
    while len(top6_d2) < 6:
        top6_d2.append(random.randint(0, 9))

    hasil = [top6_d1, top6_d2]

    key1 = d1 + d2
    dist3 = matrix[0].get(key1, {})
    top6_d3 = sorted(dist3.items(), key=lambda x: -x[1])
    top6_d3 = [int(k) for k, _ in top6_d3[:6]]
    while len(top6_d3) < 6:
        top6_d3.append(random.randint(0, 9))
    hasil.append(top6_d3)

    key2 = d2 + str(top6_d3[0]) if top6_d3 else d2 + str(random.randint(0, 9))
    dist4 = matrix[1].get(key2, {})
    top6_d4 = sorted(dist4.items(), key=lambda x: -x[1])
    top6_d4 = [int(k) for k, _ in top6_d4[:6]]
    while len(top6_d4) < 6:
        top6_d4.append(random.randint(0, 9))
    hasil.append(top6_d4)

    return hasil

# HYBRID MARKOV: Gabungan Order-1 & Order-2
def top6_markov_hybrid(df):
    hasil1, _ = top6_markov(df)
    hasil2 = top6_markov_order2(df)

    hasil = []
    for i in range(4):
        gabung = hasil1[i] + hasil2[i]
        freq = Counter(gabung)
        top6 = [k for k, _ in freq.most_common(6)]
        while len(top6) < 6:
            top6.append(random.randint(0, 9))
        hasil.append(top6)

    return hasil

def kombinasi_4d_markov_hybrid(df, top_n=10, mode="average", min_conf=0.0001):
    from collections import defaultdict, Counter

    data = df["angka"].astype(str).tolist()
    if len(data) < 30:
        return []

    def smooth(counter, k=1):
        return Counter({str(i): counter.get(str(i), 0) + k for i in range(10)})

    def normalize(counter):
        total = sum(counter.values())
        return {k: v / total for k, v in counter.items()} if total else {k: 0.1 for k in counter}

    def prob_order1(i, prev, curr):
        dist = smooth(matrix_order1[i].get(prev, {}))
        norm = normalize(dist)
        return norm.get(curr, 0.001)

    def prob_order2(i, key, curr):
        dist = smooth(matrix_order2[i].get(key, {}))
        norm = normalize(dist)
        return norm.get(curr, 0.001)

    # --- Matrix Setup ---
    freq_ribuan = smooth(Counter(x[0] for x in data))
    norm_ribuan = normalize(freq_ribuan)
    matrix_order1 = build_transition_matrix(data)
    matrix_order2 = build_transition_matrix_order2(data)

    hasil = []
    for d1 in range(10):
        for d2 in range(10):
            for d3 in range(10):
                for d4 in range(10):
                    s1, s2, s3, s4 = str(d1), str(d2), str(d3), str(d4)

                    pr = norm_ribuan.get(s1, 0.001)
                    p1 = prob_order1(0, s1, s2)
                    p2 = prob_order1(1, s2, s3)
                    p3 = prob_order1(2, s3, s4)
                    p4 = prob_order2(0, s1 + s2, s3)
                    p5 = prob_order2(1, s2 + s3, s4)

                    # Combine probabilitas:
                    if mode == "average":
                        score = pr * 0.3 + (p1 + p2 + p3) / 3 * 0.35 + (p4 + p5) / 2 * 0.35
                    elif mode == "product":
                        score = (pr**0.3) * ((p1 * p2 * p3)**(0.35 / 3)) * ((p4 * p5)**(0.35 / 2))
                    else:
                        score = pr * 0.3 + (p1 + p2 + p3) / 3 * 0.35 + (p4 + p5) / 2 * 0.35

                    if score >= min_conf:
                        hasil.append((f"{d1}{d2}{d3}{d4}", score))

    hasil.sort(key=lambda x: -x[1])
    return hasil[:top_n]
