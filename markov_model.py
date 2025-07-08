import pandas as pd
from collections import defaultdict, Counter


def top6_markov(df):
    data = [f"{int(x):04d}" for x in df["angka"].values if str(x).isdigit() and len(str(x)) == 4]
    if len(data) < 2:
        return [[0]*4]*4, {}

    transitions = [defaultdict(Counter) for _ in range(4)]
    for a, b in zip(data[:-1], data[1:]):
        for i in range(4):
            transitions[i][a[i]][b[i]] += 1

    result = []
    for i in range(4):
        digit_preds = []
        for prev_digit, counts in transitions[i].items():
            most_common = [int(d) for d, _ in counts.most_common(6)]
            digit_preds.extend(most_common)
        digit_preds = [d for d, _ in Counter(digit_preds).most_common(6)]
        result.append(digit_preds[:6] + [0] * (6 - len(digit_preds)))
    return result, transitions


def top6_markov_order2(df):
    data = [f"{int(x):04d}" for x in df["angka"].values if str(x).isdigit() and len(str(x)) == 4]
    if len(data) < 3:
        return [[0]*4]*4

    transitions = [defaultdict(lambda: defaultdict(Counter)) for _ in range(4)]
    for i in range(len(data) - 2):
        for j in range(4):
            d1, d2, d3 = data[i][j], data[i+1][j], data[i+2][j]
            transitions[j][d1][d2][d3] += 1

    result = []
    for i in range(4):
        pred_digits = Counter()
        for d1 in transitions[i]:
            for d2 in transitions[i][d1]:
                pred_digits += transitions[i][d1][d2]
        result.append([int(k) for k, _ in pred_digits.most_common(6)])
    return result


def top6_markov_hybrid(df, digit_weights=None):
    if digit_weights is None:
        digit_weights = [1.0, 1.0, 1.0, 1.0]
    r1, _ = top6_markov(df)
    r2 = top6_markov_order2(df)
    result = []
    for i in range(4):
        combined = r1[i] + r2[i]
        freq = Counter(combined)
        for k in freq:
            freq[k] *= digit_weights[i]
        top = [d for d, _ in freq.most_common(6)]
        result.append(top)
    return result


def kombinasi_4d_markov_hybrid(df, top_n=10, mode="average", digit_weights=None):
    if digit_weights is None:
        digit_weights = {"ribuan": 1.0, "ratusan": 1.0, "puluhan": 1.0, "satuan": 1.0}
    r = top6_markov_hybrid(df, digit_weights=[digit_weights["ribuan"], digit_weights["ratusan"],
                                              digit_weights["puluhan"], digit_weights["satuan"]])
    from itertools import product
    kombs = list(product(*r))
    result = []
    for k in kombs:
        score = sum([digit_weights[d]*1 for d in ["ribuan", "ratusan", "puluhan", "satuan"]]) if mode == "average" else \
                product([digit_weights[d]*1 for d in ["ribuan", "ratusan", "puluhan", "satuan"]])
        result.append(("".join(map(str, k)), score))
    return sorted(result, key=lambda x: -x[1])[:top_n]
