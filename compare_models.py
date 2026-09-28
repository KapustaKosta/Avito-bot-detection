"""
Сравнивает логистическую регрессию, случайный лес, LightGBM, XGBoost и CatBoost на подтверждённых
признаках и пишет out/model_comparison.csv с метрикой на хронологическом сплите и кросс-валидации.
Запускается командой python compare_models.py --out out
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import lightgbm as lgb

import modeling as M
from seed import SEED

warnings.filterwarnings("ignore")

# каждая модель создаётся заново для каждого сплита, поэтому в словаре лежат фабрики, а не объекты.
# Все модели получают одни и те же признаки, отличается только подготовка пропусков
MODELS = {
    "logreg": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)),
    "random_forest": lambda: make_pipeline(SimpleImputer(strategy="median"),
                                           RandomForestClassifier(500, min_samples_leaf=5, n_jobs=4, random_state=SEED)),
    "lightgbm": lambda: lgb.LGBMClassifier(n_estimators=M.N_TREES, random_state=SEED, **M.LGB_PARAMS),
    "xgboost": lambda: XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, min_child_weight=5, subsample=0.8,
                                     colsample_bytree=0.8, reg_lambda=5.0, tree_method="hist", random_state=SEED, n_jobs=4, verbosity=0),
    "catboost": lambda: CatBoostClassifier(iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=5.0,
                                           random_seed=SEED, verbose=0, thread_count=4),
}
# логистическая регрессия и случайный лес не умеют работать с пропусками. Для них пропуски
# заполняются медианой, а сам факт пропуска (например отсутствие курсора) сохраняется отдельным флагом
NEEDS_IMPUTATION = {"logreg", "random_forest"}
# смеси, то есть среднее рангов предсказаний нескольких моделей. Модели разного устройства ошибаются
# на разных куках, и усреднение часто даёт больше, чем любая из них по отдельности
BLENDS = {"lgb+cat+xgb": ["lightgbm", "catboost", "xgboost"],
          "lgb+cat+xgb+rf": ["lightgbm", "catboost", "xgboost", "random_forest"],
          "lgb+rf": ["lightgbm", "random_forest"],
          "cat+rf": ["catboost", "random_forest"]}


def prepare(X_train, X_test, name):
    """Убирает константные колонки; моделям без поддержки NaN добавляет флаги пропусков."""
    # константная колонка ничего не даёт модели, а у StandardScaler даёт нулевое отклонение
    keep = [c for c in X_train.columns if X_train[c].nunique(dropna=True) >= 2]
    X_train, X_test = X_train[keep], X_test[keep]
    if name in NEEDS_IMPUTATION:
        with_na = [c for c in keep if X_train[c].isna().any()]
        # флаг на каждую колонку с пропусками, сами пропуски позже заполнит SimpleImputer в пайплайне
        X_train = X_train.assign(**{c + "_na": X_train[c].isna().astype(int) for c in with_na})
        X_test = X_test.assign(**{c + "_na": X_test[c].isna().astype(int) for c in with_na})
    return X_train, X_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="out")
    args = parser.parse_args()
    out = Path(args.out)
    df, pairs = M.load_features(out)
    cols = M.select_features(df, out / "hypothesis_results.csv")
    train_idx = df.index[df["is_test"] == 0]
    y_train = df.loc[train_idx, "target"].astype(int).values

    # сплиты и признаки с метками считаются один раз, чтобы все модели получили одинаковые данные
    fit_mask, eval_idx = M.chronological_split(df)
    chrono = (M.split_dependent_features(df, pairs, fit_mask), fit_mask, eval_idx)
    folds = [(M.split_dependent_features(df, pairs, mask, SEED + k), mask, held) for k, (mask, held) in enumerate(M.cv_folds(df))]

    scores_chrono, scores_oof, rows = {}, {}, []
    y_chrono = df.loc[eval_idx, "target"].astype(int).values
    for name, make in MODELS.items():
        started = time.time()
        # хронологический сплит, обучение на первой неделе и оценка на второй
        d, mask, held = chrono
        X_tr, X_te = prepare(d.loc[mask, cols], d.loc[held, cols], name)
        scores_chrono[name] = make().fit(X_tr, d.loc[mask, "target"].astype(int)).predict_proba(X_te)[:, 1]
        # кросс-валидация. Каждая кука трейна получает скор от модели, которая её не видела,
        # из этих скоров собирается один вектор по всему train (out of fold)
        oof = pd.Series(np.nan, index=train_idx)
        fold_scores = []
        for d, mask, held in folds:
            X_tr, X_te = prepare(d.loc[mask, cols], d.loc[held, cols], name)
            s = make().fit(X_tr, d.loc[mask, "target"].astype(int)).predict_proba(X_te)[:, 1]
            oof.loc[held] = s
            fold_scores.append(M.precision_at_recall(d.loc[held, "target"].astype(int).values, s))   # метрика внутри фолда
        scores_oof[name] = oof.values
        rows.append({"model": name, "chrono_p_at_r70": M.precision_at_recall(y_chrono, scores_chrono[name]),
                     "cv_oof_p_at_r70": M.precision_at_recall(y_train, oof.values),         # метрика по всему вектору oof
                     "cv_folds_mean": float(np.mean(fold_scores)), "cv_folds_std": float(np.std(fold_scores)),
                     "cv_oof_pr_auc": M.metrics(y_train, oof.values)["pr_auc"], "seconds": time.time() - started})
        print(f"{name:14s} chrono={rows[-1]['chrono_p_at_r70']:.4f} cv={rows[-1]['cv_oof_p_at_r70']:.4f}", flush=True)
    for name, members in BLENDS.items():
        # смесь считается по уже готовым предсказаниям, поэтому обучения не требует и времени не занимает
        blend_chrono = np.mean([M.rank01(scores_chrono[m]) for m in members], axis=0)
        blend_oof = np.mean([M.rank01(scores_oof[m]) for m in members], axis=0)
        rows.append({"model": "blend:" + name, "chrono_p_at_r70": M.precision_at_recall(y_chrono, blend_chrono),
                     "cv_oof_p_at_r70": M.precision_at_recall(y_train, blend_oof), "cv_folds_mean": np.nan, "cv_folds_std": np.nan,
                     "cv_oof_pr_auc": M.metrics(y_train, blend_oof)["pr_auc"], "seconds": 0.0})
    res = pd.DataFrame(rows).round(4)
    res.to_csv(out / "model_comparison.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
