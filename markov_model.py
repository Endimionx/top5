def kombinasi_4d_markov_hybrid(df, top_n=10, mode="average", min_conf=0.0001, digit_weight_input=None):
    from itertools import product
    result = top6_markov_hybrid(df)
    digit_conf_list = []

    for digit in result:
        conf = {}
        total = sum(digit.values())
        for k, v in digit.items():
            conf[int(k)] = v / total if total > 0 else 0.0
        digit_conf_list.append(conf)

    if digit_weight_input is None:
        digit_weight_input = [1.0, 1.0, 1.0, 1.0]

    kombinasi_scores = []
    for komb in product(*[digit_conf_list[i].keys() for i in range(4)]):
        confs = [digit_conf_list[i].get(komb[i], 0.0) for i in range(4)]

        # Terapkan bobot digit
        weighted = [confs[i] ** digit_weight_input[i] for i in range(4)]

        if mode == "average":
            score = sum(weighted) / 4
        elif mode == "product":
            score = np.prod(weighted)
        else:
            score = sum(weighted) / 4  # default ke average

        if score >= min_conf:
            kombinasi_scores.append(("".join(map(str, komb)), score))

    kombinasi_scores.sort(key=lambda x: x[1], reverse=True)
    return kombinasi_scores[:top_n]
