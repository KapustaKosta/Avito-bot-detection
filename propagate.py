"""
Распространяет score LightGBM по графу общих объявлений и пишет out/propagation_results.csv
с метрикой для сетки коэффициентов смешивания и числа итераций.
Запускается командой python propagate.py --out out
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

import modeling as M
from seed import SEED

# коэффициент lam задаёт, насколько оценка куки подтягивается к оценкам соседей.
# Он выбирается по хронологическому сплиту, кросс-валидация служит подтверждением
LAMBDAS = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6]
ITERATIONS = [1, 3]


def build_graph(pairs, cookie_ids):
    """Граф кука и кука в порядке cookie_ids.

    Возвращает матрицу весов, нормированную по строкам (умножение на вектор даёт взвешенное
    среднее по соседям), маску узлов, у которых есть соседи, и исходную матрицу весов."""
    # узлы это все куки, включая test. Ребро соединяет две куки с общими объявлениями,
    # вес ребра равен мере Жаккара множеств просмотренных объявлений (посчитана в features.py)
    pos = pd.Series(np.arange(len(cookie_ids)), index=cookie_ids)      # идентификатор куки в номер строки
    p = pairs[pairs["c"].isin(cookie_ids) & pairs["o"].isin(cookie_ids)]
    raw = sparse.csr_matrix((p["jac"].values, (pos[p["c"]].values, pos[p["o"]].values)), shape=(len(cookie_ids),) * 2)
    row_sum = np.asarray(raw.sum(axis=1)).ravel()   # сумма весов рёбер каждой куки
    has_neighbors = row_sum > 0
    # деление каждой строки на её сумму; у изолированных узлов строка остаётся нулевой, деления на ноль нет
    normalized = sparse.diags(np.where(has_neighbors, 1 / np.where(has_neighbors, row_sum, 1), 0)) @ raw
    return normalized.tocsr(), has_neighbors, raw


def propagate(score, labeled, y, graph, lam, iterations):
    """Итерации распространения. Размеченные узлы на каждой итерации возвращаются к своей метке."""
    W, has_neighbors, _ = graph
    p = np.where(labeled, y, score).astype(float)   # старт. У размеченных кук метка, у остальных score модели
    for _ in range(iterations):
        neighbor_mean = W @ p                       # взвешенное среднее текущих оценок по соседям
        # правило p_i = (1 - lam) * score_i + lam * среднее по соседям. Кука без соседей сохраняет свой score
        updated = np.where(has_neighbors, (1 - lam) * score + lam * neighbor_mean, score)
        p = np.where(labeled, y, updated)           # метки закреплены, они передаются соседям, но сами не меняются
    return p


def base_scores(df, pairs, cols, fit_mask, seed):
    """LightGBM, обученный на fit_mask, даёт score всем остальным кукам, включая test."""
    d = M.split_dependent_features(df, pairs, fit_mask, seed)     # признаки с метками только по меткам fit_mask
    fit_idx, rest_idx = df.index[fit_mask], df.index[~fit_mask]
    s, _, _ = M.fit_lightgbm(d.loc[fit_idx, cols], d.loc[fit_idx, "target"].astype(int), d.loc[rest_idx, cols], seed=seed)
    score = np.full(len(df), np.nan)                # у обучающих кук score не нужен, там будет стоять метка
    score[rest_idx] = s
    return score


def evaluate(df, pairs, cols, graph, fit_mask, eval_idx, seed):
    """Метрика на eval_idx для всех сочетаний lam и числа итераций."""
    y = df["target"].fillna(0).values               # у test метки нет, ноль там никогда не читается как метка
    score = np.where(fit_mask.values, y, base_scores(df, pairs, cols, fit_mask, seed))
    result = {}
    for lam in LAMBDAS:
        for it in ITERATIONS:
            if lam == 0 and it > 1:
                continue        # при lam = 0 итерации ничего не меняют
            p = propagate(score, fit_mask.values, y, graph, lam, it)
            result[(lam, it)] = M.precision_at_recall(y[eval_idx].astype(int), p[eval_idx])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="out")
    args = parser.parse_args()
    out = Path(args.out)
    df, pairs = M.load_features(out)
    cols = M.select_features(df, out / "hypothesis_results.csv")
    graph = build_graph(pairs, pd.Index(df["cookie_id"]))
    print(f"граф: {len(df)} узлов, изолированных {int((~graph[1]).sum())}")

    # хронологический сплит, по нему выбирается лучшее сочетание lam и итераций
    fit_mask, eval_idx = M.chronological_split(df)
    chrono = evaluate(df, pairs, cols, graph, fit_mask, eval_idx, SEED)
    # кросс-валидация, у каждого фолда свой seed для базовой модели
    cv = {key: [] for key in chrono}
    for k, (mask, held) in enumerate(M.cv_folds(df)):
        for key, value in evaluate(df, pairs, cols, graph, mask, held, SEED + k).items():
            cv[key].append(value)
    res = pd.DataFrame([{"lambda": lam, "iters": it, "chrono_p_at_r70": chrono[(lam, it)],
                         "cv_folds_mean": float(np.mean(cv[(lam, it)])), "cv_folds_std": float(np.std(cv[(lam, it)]))}
                        for (lam, it) in chrono]).round(4)
    res.to_csv(out / "propagation_results.csv", index=False)
    print(res.to_string(index=False))
    best = res.iloc[res["chrono_p_at_r70"].idxmax()]
    print(f"лучшее по хронологии: lambda={best['lambda']}, итераций {int(best['iters'])}")


if __name__ == "__main__":
    main()
