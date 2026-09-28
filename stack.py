"""
Обучает модель второго уровня на score LightGBM и графовых признаках, пишет out/stack_results.csv
и с флагом --write файл предсказаний out/submission_stack.csv.
Запускается командой python stack.py --out out --write
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

import modeling as M
import propagate as P
from seed import SEED

INNER_FOLDS = 5
# вместо одного коэффициента распространения (как в propagate.py) второму уровню даются
# сразу несколько вариантов, и он сам решает, когда доверять графу, а когда собственному score куки
LAMBDAS = [0.1, 0.3]
ITERATIONS = [1, 3]
# малый LightGBM. Входов у второго уровня немного, поэтому деревья мелкие и листья крупные
LEVEL2_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=7, min_child_samples=50, n_estimators=200,
                     feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, n_jobs=4)


def logit(p):
    """Перевод вероятности в логит, модели второго уровня удобнее работать с неограниченной шкалой."""
    p = np.clip(p, 1e-6, 1 - 1e-6)     # защита от log(0) при вероятностях ровно 0 или 1
    return np.log(p / (1 - p))


def graph_features(score, labeled, y, graph):
    """Входы уровня 2 для всех узлов при закреплённых метках labeled.

    W это матрица весов, нормированная по строкам, поэтому W @ v даёт взвешенное среднее v по соседям.
    Через неё считаются доля размеченных соседей, средняя метка размеченных соседей, средний score
    неразмеченных соседей и среднее по всем соседям."""
    W, has_neighbors, raw = graph
    p_known = np.where(labeled, y, score)       # у размеченных кук метка, у остальных score
    lab = labeled.astype(float)
    labeled_weight = W @ lab                    # какая доля веса соседей приходится на размеченных
    unlabeled_weight = W @ (1 - lab)            # и на неразмеченных
    f = pd.DataFrame({"base": logit(score),
                      "deg": raw.getnnz(axis=1).astype(float),                 # число соседей
                      "wsum": np.asarray(raw.sum(axis=1)).ravel(),             # сумма Жаккаров
                      "max_w": raw.max(axis=1).toarray().ravel(),              # самый похожий сосед
                      "labeled_share": labeled_weight,
                      # средняя метка среди размеченных соседей. Деление только там, где такие соседи есть, иначе NaN
                      "nb_label_mean": np.divide(W @ (lab * y), labeled_weight, out=np.full(len(y), np.nan), where=labeled_weight > 0),
                      # средний score среди неразмеченных соседей
                      "nb_score_mean": np.divide(W @ ((1 - lab) * score), unlabeled_weight, out=np.full(len(y), np.nan), where=unlabeled_weight > 0),
                      "nb_all_mean": np.where(has_neighbors, W @ p_known, np.nan)})
    # четыре варианта распространения из propagate.py как отдельные входы
    for lam in LAMBDAS:
        for it in ITERATIONS:
            f[f"prop_l{lam}_t{it}"] = logit(P.propagate(p_known, labeled, y, graph, lam, it))
    return f


def level1(df, pairs, cols, graph, fit_mask, seed):
    """Входы уровня 2 для всех строк.

    Обучающая часть получает их из вложенных фолдов, все остальные строки (валидация и test)
    от модели, обученной на всей обучающей части с закреплёнными метками."""
    y = df["target"].fillna(0).values
    fit_idx, rest_idx = df.index[fit_mask], df.index[~fit_mask]
    feats = None
    y_fit = df.loc[fit_idx, "target"].astype(int)
    # вложенные фолды. Каждый фолд обучающей части оценивается моделью, которая его не видела,
    # признаки с метками пересчитываются без его меток, а распространение по графу считает его
    # неразмеченным. Без этого второй уровень увидел бы ответы тех кук, на которых его потом оценивают
    for k, (fit, held) in enumerate(StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed).split(fit_idx, y_fit)):
        inner = pd.Series(False, index=df.index)    # обучающая часть вложенного фолда как маска по всему df
        inner.loc[fit_idx[fit]] = True
        score = np.where(inner.values, y, P.base_scores(df, pairs, cols, inner, seed + 100 + k))
        gf = graph_features(score, inner.values, y, graph)
        feats = pd.DataFrame(index=df.index, columns=gf.columns, dtype=float) if feats is None else feats
        feats.loc[fit_idx[held]] = gf.loc[fit_idx[held]].values     # записываются только строки отложенного фолда
    # валидация и test получают входы от модели на всей обучающей части
    score = np.where(fit_mask.values, y, P.base_scores(df, pairs, cols, fit_mask, seed))
    gf = graph_features(score, fit_mask.values, y, graph)
    feats.loc[rest_idx] = gf.loc[rest_idx].values
    return feats


def level2(feats, y, fit_idx, eval_idx, seed):
    """Score уровня 2 на eval_idx. Возвращает базу, распространение с lam 0,1, логистическую регрессию и малый LightGBM."""
    X_fit, X_eval = feats.loc[fit_idx], feats.loc[eval_idx]
    median = X_fit.median()
    # две точки отсчёта без обучения второго уровня, обратный перевод логита в вероятность
    out = {"base": 1 / (1 + np.exp(-X_eval["base"].values)),
           "prop_l0.1_t1": 1 / (1 + np.exp(-X_eval["prop_l0.1_t1"].values))}
    # логистическая регрессия не принимает NaN, поэтому пропуски заполняются медианой обучающей части
    logreg = LogisticRegression(C=1.0, max_iter=3000).fit(X_fit.fillna(median), y[fit_idx])
    out["stack_logreg"] = logreg.predict_proba(X_eval.fillna(median))[:, 1]
    # LightGBM работает с NaN сам и различает "соседей нет" и "соседи есть"
    gbm = lgb.LGBMClassifier(random_state=seed, **LEVEL2_PARAMS).fit(X_fit, y[fit_idx])
    out["stack_lgbm"] = gbm.predict_proba(X_eval)[:, 1]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="out")
    parser.add_argument("--write", action="store_true", help="записать файл предсказаний для test")
    args = parser.parse_args()
    out = Path(args.out)
    df, pairs = M.load_features(out)
    cols = M.select_features(df, out / "hypothesis_results.csv")
    graph = P.build_graph(pairs, pd.Index(df["cookie_id"]))
    y = df["target"].fillna(0).astype(int).values
    train_idx = df.index[df["is_test"] == 0]

    # хронологический сплит
    fit_mask, eval_idx = M.chronological_split(df)
    scores = level2(level1(df, pairs, cols, graph, fit_mask, SEED), y, df.index[fit_mask], eval_idx, SEED)
    chrono = {name: M.precision_at_recall(y[eval_idx], s) for name, s in scores.items()}
    print("неделя 0, затем неделя 1:", {k: round(v, 4) for k, v in chrono.items()}, flush=True)

    # кросс-валидация, у каждого фолда свои сиды вложенных фолдов и второго уровня
    cv = {name: [] for name in chrono}
    for k, (mask, held) in enumerate(M.cv_folds(df)):
        scores = level2(level1(df, pairs, cols, graph, mask, SEED + 10 * (k + 1)), y, df.index[mask], held, SEED + k)
        for name, s in scores.items():
            cv[name].append(M.precision_at_recall(y[held], s))
        print(f"  fold {k}:", {n: round(v[-1], 4) for n, v in cv.items()}, flush=True)
    res = pd.DataFrame([{"model": name, "chrono_p_at_r70": chrono[name], "cv_folds_mean": float(np.mean(cv[name])),
                         "cv_folds_std": float(np.std(cv[name])),
                         # в скольких фолдах из пяти вариант обошёл базовый LightGBM
                         "cv_wins_vs_base": int(sum(a > b for a, b in zip(cv[name], cv["base"])))} for name in chrono]).round(4)
    res.to_csv(out / "stack_results.csv", index=False)
    print(res.to_string(index=False))

    if args.write:
        # финальная модель обучается на всём train, предсказания для test усредняются по сидам
        fit_mask = df["is_test"] == 0
        test_idx = df.index[~fit_mask]
        predictions = [level2(level1(df, pairs, cols, graph, fit_mask, seed), y, train_idx, test_idx, seed)["stack_lgbm"]
                       for seed in M.SEEDS]
        sub = pd.DataFrame({"cookie_id": df.loc[test_idx, "cookie_id"].values, "score": np.mean(predictions, axis=0)})
        sub.to_csv(out / "submission_stack.csv", index=False)
        print(f"записан {out / 'submission_stack.csv'}, {len(sub)} кук")


if __name__ == "__main__":
    main()
