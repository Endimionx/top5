import random
from collections import defaultdict, Counter
import pandas as pd

# MARKOV ORDER-1
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
    transisi = [{k: dict(v) for k, v in matrix[i].items()} for i in range(3)]
    kombinasi = Counter(data).most_common(10)

    hasil = [[], [], [], []]

    # Ribuan (digit ke-0)
    top6_ribuan = [k for k, _ in freq_ribuan.most_common(6)]
    while len(top6_ribuan) < 6:
        top6_ribuan.append(random.randint(0, 9))
    hasil[0] = top6_ribuan

    # Ratusan (digit ke-1), dari transisi ribuan -> ratusan = matrix[0]
    kandidat_ratusan = []
    for prev in matrix[0]:
        kandidat_ratusan.extend(matrix[0][prev].keys())
    top6_ratusan = [int(k) for k, _ in Counter(kandidat_ratusan).most_common(6)]
    while len(top6_ratusan) < 6:
        top6_ratusan.append(random.randint(0, 9))
    hasil[1] = top6_ratusan

    # Puluhan (digit ke-2), dari transisi ratusan -> puluhan = matrix[1]
    kandidat_puluhan = []
    for prev in matrix[1]:
        kandidat_puluhan.extend(matrix[1][prev].keys())
    top6_puluhan = [int(k) for k, _ in Counter(kandidat_puluhan).most_common(6)]
    while len(top6_puluhan) < 6:
        top6_puluhan.append(random.randint(0, 9))
    hasil[2] = top6_puluhan

    # Satuan (digit ke-3), dari transisi puluhan -> satuan = matrix[2]
    kandidat_satuan = []
    for prev in matrix[2]:
        kandidat_satuan.extend(matrix[2][prev].keys())
    top6_satuan = [int(k) for k, _ in Counter(kandidat_satuan).most_common(6)]
    while len(top6_satuan) < 6:
        top6_satuan.append(random.randint(0, 9))
    hasil[3] = top6_satuan

    info = {
        "frekuensi_ribuan": dict(freq_ribuan),
        "transisi": transisi,
        "kombinasi_populer": kombinasi
    }

    return hasil, info

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

def top6_markov_order2(df):
    data = df["angka"].astype(str).tolist()
    matrix = build_transition_matrix_order2(data)

    pairs = [x[:2] for x in data]
    top_pairs = Counter(pairs).most_common(6)
    d1, d2 = top_pairs[0][0][0], top_pairs[0][0][1]

    top6_ribuan = list(set([int(p[0][0]) for p in top_pairs]))
    top6_ratusan = list(set([int(p[0][1]) for p in top_pairs]))
    while len(top6_ribuan) < 6:
        top6_ribuan.append(random.randint(0, 9))
    while len(top6_ratusan) < 6:
        top6_ratusan.append(random.randint(0, 9))

    hasil = [top6_ribuan, top6_ratusan]

    key1 = d1 + d2
    dist3 = matrix[0].get(key1, {})
    top6_puluhan = sorted(dist3.items(), key=lambda x: -x[1])
    top6_puluhan = [int(k) for k, _ in top6_puluhan[:6]]
    while len(top6_puluhan) < 6:
        top6_puluhan.append(random.randint(0, 9))
    hasil.append(top6_puluhan)

    key2 = d2 + str(top6_puluhan[0])
    dist4 = matrix[1].get(key2, {})
    top6_satuan = sorted(dist4.items(), key=lambda x: -x[1])
    top6_satuan = [int(k) for k, _ in top6_satuan[:6]]
    while len(top6_satuan) < 6:
        top6_satuan.append(random.randint(0, 9))
    hasil.append(top6_satuan)

    return hasil

# HYBRID
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
