"""
Общие функции для скриптов с моделями. Метрика соревнования, хронологический сплит и
кросс-валидация, отбор подтверждённых признаков, пересчёт признаков с метками внутри сплита,
обучение LightGBM с едиными параметрами и ранги для смешивания предсказаний.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

from seed import SEED

# сиды для усреднения финальной модели. При SEED = 11 это 11, 22, 33, 44, 55
SEEDS = [SEED + 11 * i for i in range(5)]

# признаки, которые нужны не для ловли ботов, а чтобы модель правильно читала пропуски.
# Без доли платформ пропуск в курсорных признаках у телефона и у HTTP-парсера выглядит одинаково
CONTEXT_FEATURES = ["share_platform_web", "share_platform_android", "share_platform_ios", "n_web_events", "few_events"]

# признак с меткой, который считается по совпадению ключа. Слева столбец признака, справа столбец ключа
LABEL_RATE_KEYS = {"ua_hash_bot_rate_oof": "ua_norm_mode"}
NEIGHBOR_RATE = "neighbor_bot_rate_oof"
MIN_KEY_COUNT = 3       # меньше трёх кук с тем же ключом дают слишком шумную долю, ставится пропуск
K_NEIGHBORS = 20        # сколько ближайших по Жаккару соседей учитывается в доле ботов
INNER_FOLDS = 5         # вложенные фолды, по которым обучающая часть получает признаки с метками

LGB_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=40,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                  verbose=-1, n_jobs=4)
N_TREES = 281           # выбрано ранней остановкой на хронологическом сплите


def precision_at_recall(y, score, min_recall=0.7):
    """Максимальная precision среди порогов, при которых recall не ниже min_recall.

    Куки сортируются по убыванию score. Для каждого возможного порога считается, сколько ботов
    попало выше порога (tp) и сколько кук всего выше порога (n). Порог может стоять только между
    группами с разным score, поэтому берутся позиции, где score меняется."""
    # mergesort устойчив, поэтому куки с равным score сохраняют исходный порядок и результат воспроизводим
    order = np.argsort(-score, kind="mergesort")
    y, score = np.asarray(y)[order], np.asarray(score)[order]
    tp = np.cumsum(y)                       # ботов среди первых i кук
    n = np.arange(1, len(y) + 1)            # всего кук среди первых i
    # куки с одинаковым score образуют одну группу. Порог либо берёт всю группу, либо не берёт никого,
    # так же считает проверяющая система. Поэтому порог допустим только после последней куки группы
    last_in_group = np.r_[score[1:] != score[:-1], True]
    tp, n = tp[last_in_group], n[last_in_group]
    recall = tp / max(y.sum(), 1)
    precision = tp / n
    ok = recall >= min_recall               # допустимые пороги
    return float(precision[ok].max()) if ok.any() else 0.0


def metrics(y, score):
    """Основная метрика и две вспомогательные для диагностики."""
    return {"p_at_r70": round(precision_at_recall(y, score), 4),
            "pr_auc": round(float(average_precision_score(y, score)), 4),
            "roc_auc": round(float(roc_auc_score(y, score)), 4)}


def load_features(out_dir):
    """Таблица признаков и пары соседей, которые записал features.py."""
    df = pd.read_csv(f"{out_dir}/features.csv")
    df["target"] = df["target"].astype(float)      # у test метки нет, там NaN
    pairs = pd.read_csv(f"{out_dir}/neighbor_pairs.csv")
    return df, pairs


def select_features(df, results_path):
    """Признаки, подтверждённые в hypotheses.py, плюс контекст платформы."""
    results = pd.read_csv(results_path)
    confirmed = results.loc[results["confirmed"].astype(bool), "feature"].unique().tolist()
    cols = []
    for c in confirmed + CONTEXT_FEATURES:
        # один признак может подтвердиться в нескольких гипотезах, поэтому проверяется повтор;
        # строковые колонки (например ua_norm_mode) модели не подаются, только числа и флаги
        if c in df.columns and c not in cols and df[c].dtype.kind in "fib":
            cols.append(c)
    return cols


def chronological_split(df):
    """Маска обучающих строк (первая неделя train) и индексы оценочных строк (вторая неделя)."""
    # обучение на первой неделе, оценка на второй. Так же соотносятся train и test,
    # поэтому эта схема ближе всего к условиям соревнования
    train = df["is_test"] == 0
    first_week = df["week_id"] == df.loc[train, "week_id"].min()
    return train & first_week, df.index[train & ~first_week]


def cv_folds(df, seed=SEED, n_splits=5):
    """Список пар (маска обучающих строк, индексы оценочных строк) для кросс-валидации по train."""
    # стратификация сохраняет долю ботов в каждом фолде; test в фолды не входит, но остаётся в df,
    # чтобы графовые модели видели весь граф
    train_idx = df.index[df["is_test"] == 0]
    y = df.loc[train_idx, "target"].astype(int)
    folds = []
    for fit, held in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(train_idx, y):
        mask = pd.Series(False, index=df.index)         # маска по всему df, включая test
        mask.loc[train_idx[fit]] = True
        folds.append((mask, train_idx[held]))
    return folds


def _rate_by_key(df, key, target_idx, source_idx):
    """Доля ботов среди кук source с тем же значением key, для каждой куки из target."""
    stats = df.loc[source_idx].groupby(key)["target"].agg(["sum", "count"])   # ботов и кук на каждое значение ключа
    keys = df.loc[target_idx, key]
    s, c = keys.map(stats["sum"]), keys.map(stats["count"])
    # редкий ключ даёт пропуск, а не долю по одной или двум кукам
    return pd.Series(np.where(c >= MIN_KEY_COUNT, s / c, np.nan), index=target_idx)


def _rate_by_neighbors(df, pairs, target_idx, source_idx):
    """Доля ботов среди K ближайших по Жаккару соседей из source, для каждой куки из target.

    В pairs куки записаны своими идентификаторами, а df индексирован позициями строк, поэтому
    сначала идёт переход к идентификаторам, а в конце обратно к позициям."""
    labels = pd.Series(df.loc[source_idx, "target"].values, index=df.loc[source_idx, "cookie_id"].values)
    target_ids = df.loc[target_idx, "cookie_id"]
    # оставляем только пары, где кука из target, а сосед размечен и входит в source
    p = pairs[pairs["c"].isin(target_ids) & pairs["o"].isin(labels.index)]
    # у каждой куки берутся K соседей с наибольшим Жаккаром, по ним усредняется метка
    p = p.assign(t=p["o"].map(labels)).sort_values(["c", "jac"], ascending=[True, False]).groupby("c").head(K_NEIGHBORS)
    rate = p.groupby("c")["t"].mean()
    return pd.Series(target_ids.map(rate).values, index=target_idx)     # кука без размеченных соседей получает NaN


def split_dependent_features(df, pairs, fit_mask, seed=SEED):
    """Копия df, где признаки с метками и udist пересчитаны только по меткам fit_mask.

    Строки делятся на блоки (кому считаем, по чьим меткам считаем). Обучающая часть разбивается
    на вложенные фолды, каждому фолду доли считаются по остальным фолдам. Все строки вне обучающей
    части (валидация и test) получают доли по всей обучающей части."""
    # два признака используют целевую переменную, доля ботов по User-Agent и доля ботов среди соседей.
    # Если посчитать их один раз по всем меткам, кука увидит собственную метку через соседей,
    # и модель научится читать ответ. Поэтому в каждом сплите они считаются заново
    out = df.copy()
    fit_idx, rest_idx = df.index[fit_mask], df.index[~fit_mask]
    y_fit = df.loc[fit_idx, "target"].astype(int)
    # блок (кому, по кому). Для вложенного фолда доли считаются по остальным фолдам обучающей части
    blocks = [(fit_idx[held], fit_idx[fit])
              for fit, held in StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed).split(fit_idx, y_fit)]
    if len(rest_idx):
        blocks.append((rest_idx, fit_idx))          # валидация и test по всей обучающей части
    for col in list(LABEL_RATE_KEYS) + [NEIGHBOR_RATE]:
        out[col] = np.nan                           # старые значения из features.csv затираются
    for target_idx, source_idx in blocks:
        for col, key in LABEL_RATE_KEYS.items():
            out.loc[target_idx, col] = _rate_by_key(df, key, target_idx, source_idx)
        out.loc[target_idx, NEIGHBOR_RATE] = _rate_by_neighbors(df, pairs, target_idx, source_idx)
    # расстояние от медианы людей тоже зависит от того, кто попал в обучение.
    # Медиана берётся только по людям обучающей части
    humans = df.loc[fit_idx][df.loc[fit_idx, "target"] == 0]
    for col in [c for c in df.columns if c.startswith("udist_")]:
        raw = col[len("udist_"):]                   # имя исходного признака без префикса
        out[col] = (df[raw] - humans[raw].median()).abs()
    return out


def fit_lightgbm(X_train, y_train, X_test, n_trees=N_TREES, seed=SEED, X_val=None, y_val=None):
    """Обучает LightGBM и возвращает score на X_test, число деревьев и важности признаков.

    Если передана валидационная выборка, число деревьев выбирает ранняя остановка по log-loss.
    Обучение прекращается, когда ошибка на валидации не улучшается сто деревьев подряд."""
    model = lgb.LGBMClassifier(n_estimators=n_trees, random_state=seed, **LGB_PARAMS)
    if X_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        best = int(model.best_iteration_ or n_trees)    # лучшая итерация, либо все деревья, если остановки не было
    else:
        model.fit(X_train, y_train)
        best = n_trees
    # важность по суммарному выигрышу в функции потерь, нужна только для отчёта в ноутбуке
    importance = pd.Series(model.booster_.feature_importance("gain"), index=X_train.columns)
    return model.predict_proba(X_test)[:, 1], best, importance


def rank01(score):
    """Ранги от 0 до 1. Нужны, чтобы смешивать предсказания моделей с разным масштабом score."""
    # вероятности бустинга и логиты сети напрямую не складываются, а ранги у всех в одной шкале
    return pd.Series(score).rank().values / len(score)
