"""
Обучает гибридную модель, GRU по последовательности событий куки плюс табличные признаки,
и пишет out/seq_model_results.csv со сравнением с LightGBM и стеком.
Запускается командой python seq_model.py --data data --out out --seeds 11,22,33
"""
import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import features as F
import modeling as M
import propagate as P
import stack as S
from seed import SEED

SEQ_LEN = 64            # средняя кука содержит 20 событий, длиннее 64 только 3% кук
N_CATEGORIES = 60       # самые частые категории получают свой эмбеддинг, остальные общий
CLASSES = ["serp", "item", "photo", "seller", "contact", "fav", "login", "captcha", "other"]
MAX_EPOCHS, PATIENCE, BATCH, LR = 25, 5, 256, 2e-3
HIDDEN, DROPOUT = 48, 0.3


def prepare_sequences(data_dir, cookie_ids):
    """Тензоры последовательностей для всех кук в порядке cookie_ids.

    Куки короче SEQ_LEN дополняются нулями, которые сеть игнорирует через маску. У кук длиннее
    берутся последние SEQ_LEN событий. Индекс 0 во всех категориальных полях зарезервирован под паддинг."""
    # бустинг видит куку как набор сводок, порядок событий в них потерян. Эта модель читает события
    # в том порядке, в каком они происходили, поэтому ей нужен сам журнал, а не features.csv
    events = pd.read_csv(data_dir / "events.csv.gz")
    meta = pd.concat([pd.read_csv(data_dir / "train.csv").assign(is_test=0),
                      pd.read_csv(data_dir / "test.csv").assign(is_test=1)], ignore_index=True)
    ev = F.preprocess(events, meta)                 # та же подготовка, что и у признаков, включая обрезку по окну
    g = ev.groupby("cookie_id", sort=False)
    # категориальные поля события переводятся в целые индексы для эмбеддингов
    ev["cls_id"] = ev["cls"].map({c: i for i, c in enumerate(CLASSES)}).astype(int)
    ev["plat_id"] = ev["platform_norm"].map({"web": 0, "android": 1, "ios": 2}).fillna(3).astype(int)
    top = ev["item_category"].value_counts().index[:N_CATEGORIES]
    ev["cat_id"] = ev["item_category"].map({c: i + 1 for i, c in enumerate(top)}).fillna(0).astype(int)    # 0 это редкая или пустая категория
    # семь чисел на событие. Лог паузы до предыдущего события, сдвиг курсора по осям,
    # флаги наличия курсора и страницы выдачи
    numeric = pd.DataFrame({
        "log_dt": np.log1p(ev["dt_prev"].fillna(0)),
        "dx": (ev["pointer_x"] - g["pointer_x"].shift(1)).fillna(0) / 1000,     # в тысячах пикселей, чтобы масштаб был около единицы
        "dy": (ev["pointer_y"] - g["pointer_y"].shift(1)).fillna(0) / 1000,
        "pointer_missing_web": (ev["is_web"] & ~ev["pointer_valid"]).astype(float),
        "has_pointer": ev["pointer_valid"].astype(float),
        "log_page": np.log1p(ev["search_page"].fillna(0)),
        "has_page": ev["search_page"].notna().astype(float)})
    rank = g.cumcount()                             # номер события внутри куки
    length = g["cls_id"].transform("size")          # число событий куки
    keep = rank >= length - SEQ_LEN                          # последние SEQ_LEN событий каждой куки
    position = (rank - (length - SEQ_LEN).clip(lower=0))[keep].values      # позиция события в тензоре, от 0
    row = pd.Series(np.arange(len(cookie_ids)), index=cookie_ids)[ev.loc[keep, "cookie_id"]].values   # строка куки в тензоре
    n = len(cookie_ids)
    seq = {"cls": np.zeros((n, SEQ_LEN), np.int64), "plat": np.zeros((n, SEQ_LEN), np.int64),
           "cat": np.zeros((n, SEQ_LEN), np.int64), "num": np.zeros((n, SEQ_LEN, numeric.shape[1]), np.float32),
           "mask": np.zeros((n, SEQ_LEN), np.float32)}
    # индексы сдвигаются на единицу, чтобы ноль остался за паддингом
    seq["cls"][row, position] = ev.loc[keep, "cls_id"].values + 1
    seq["plat"][row, position] = ev.loc[keep, "plat_id"].values + 1
    seq["cat"][row, position] = ev.loc[keep, "cat_id"].values + 1
    seq["num"][row, position] = numeric[keep].values.astype(np.float32)
    seq["mask"][row, position] = 1.0                # единица там, где стоит настоящее событие
    return {k: torch.from_numpy(v) for k, v in seq.items()}


def tabular(d, cols, fit_idx):
    """Стандартизованные табличные признаки с индикаторами пропусков, как в gnn_model."""
    X = d[cols].astype(float)
    mean, std = X.loc[fit_idx].mean(), X.loc[fit_idx].std().replace(0, 1).fillna(1)     # статистики только по обучающей части
    Z = ((X - mean) / std).clip(-6, 6)
    missing = X.isna().astype(float).loc[:, X.isna().any()]
    return torch.from_numpy(np.hstack([Z.fillna(0).values, missing.values]).astype(np.float32))


class Hybrid(nn.Module):
    """Две ветки, GRU по событиям и линейный слой по табличным признакам, соединённые в одной голове."""

    def __init__(self, n_numeric, n_tabular, variant):
        super().__init__()
        self.variant = variant
        # эмбеддинги категориальных полей события, размер словаря на два больше из-за паддинга и сдвига
        self.emb_cls = nn.Embedding(len(CLASSES) + 2, 12, padding_idx=0)
        self.emb_plat = nn.Embedding(5, 4, padding_idx=0)
        self.emb_cat = nn.Embedding(N_CATEGORIES + 2, 8, padding_idx=0)
        self.numeric = nn.Linear(n_numeric, 16)
        self.gru = nn.GRU(12 + 4 + 8 + 16, HIDDEN, batch_first=True)    # вход GRU это конкатенация всех частей события
        # табличная ветка получает те же подтверждённые признаки, что и бустинг. Без неё модель не видела бы
        # графовых признаков и статистик по всему окну, с ней можно измерить вклад именно порядка событий
        self.tabular = nn.Sequential(nn.Linear(n_tabular, 64), nn.ReLU(), nn.Dropout(DROPOUT))
        # ширина входа головы зависит от того, какие ветки включены. От GRU идут среднее и последнее состояние
        width = {"hybrid": 2 * HIDDEN + 64, "seq": 2 * HIDDEN, "tab": 64}[variant]
        self.head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(width, 48), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(48, 1))

    def forward(self, cls, plat, cat, num, mask, tab):
        parts = []
        if self.variant != "tab":
            x = torch.cat([self.emb_cls(cls), self.emb_plat(plat), self.emb_cat(cat), torch.relu(self.numeric(num))], -1)
            h, _ = self.gru(x)                      # состояние после каждого события
            # среднее по настоящим событиям (маска убирает паддинг) и состояние после последнего события
            m = mask.unsqueeze(-1)
            mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
            last = h[torch.arange(len(h)), (mask.sum(1).long() - 1).clamp(min=0)]
            parts += [mean, last]
        if self.variant != "seq":
            parts.append(self.tabular(tab))
        return self.head(torch.cat(parts, -1)).squeeze(-1)


def batch(seq, tab, idx):
    """Срез всех тензоров по индексам кук, в том порядке, который ждёт forward."""
    return [seq["cls"][idx], seq["plat"][idx], seq["cat"][idx], seq["num"][idx], seq["mask"][idx], tab[idx]]


def predict(model, seq, tab, idx):
    """Логиты модели для кук idx, кусками по 1024, чтобы не держать всё в памяти."""
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(*batch(seq, tab, idx[i:i + 1024])).numpy() for i in range(0, len(idx), 1024)])


def fit(seq, tab, y, train_idx, variant, seed, n_epochs=None, val_idx=None):
    """Обучение. Без n_epochs число эпох выбирает ранняя остановка по log-loss на val_idx."""
    torch.manual_seed(seed)
    model = Hybrid(seq["num"].shape[-1], tab.shape[1], variant)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    y_train = torch.from_numpy(y[train_idx].astype(np.float32))
    best_loss, best_epoch, best_state, waited = np.inf, 0, None, 0
    for epoch in range(1, (n_epochs or MAX_EPOCHS) + 1):
        model.train()
        order = torch.randperm(len(train_idx))      # новое перемешивание кук каждую эпоху
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            optimizer.zero_grad()
            loss = loss_fn(model(*batch(seq, tab, train_idx[b])), y_train[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)      # защита от взрыва градиента в GRU
            optimizer.step()
        if n_epochs is None:
            # ранняя остановка по ошибке на валидации
            logits = torch.from_numpy(predict(model, seq, tab, val_idx))
            val_loss = float(loss_fn(logits, torch.from_numpy(y[val_idx].astype(np.float32))))
            if val_loss < best_loss - 1e-4:
                best_loss, best_epoch, waited = val_loss, epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}    # копия лучших весов
            elif (waited := waited + 1) >= PATIENCE:
                break                               # PATIENCE эпох без улучшения
    if best_state is not None:
        model.load_state_dict(best_state)           # возвращаемся к лучшей эпохе
        return model, best_epoch
    return model, n_epochs


def run_split(df, pairs, seq, cols, graph, fit_mask, eval_idx, variants, seeds, n_epochs, seed):
    """Логиты сети по вариантам, score LightGBM и стека на eval_idx, плюс число эпох ранней остановки."""
    d = M.split_dependent_features(df, pairs, fit_mask, seed)     # признаки с метками только по меткам fit_mask
    fit_idx, eval_idx = df.index[fit_mask].values, np.asarray(eval_idx)
    y = d["target"].fillna(0).astype(int).values
    tab = tabular(d, cols, fit_idx)
    scores, epochs = {}, {}
    for variant in variants:
        logits = []
        for s in seeds:
            # если число эпох для варианта уже известно, валидация не нужна и ранняя остановка выключена
            model, ep = fit(seq, tab, y, fit_idx, variant, seed + s, n_epochs.get(variant), eval_idx if variant not in n_epochs else None)
            logits.append(predict(model, seq, tab, eval_idx))
            epochs.setdefault(variant, []).append(ep)
        scores[variant] = np.mean(logits, axis=0)   # усреднение логитов по сидам
    # точки отсчёта на тех же данных, базовый LightGBM и стек
    scores["lgb_base"], _, _ = M.fit_lightgbm(d.loc[fit_idx, cols], y[fit_idx], d.loc[eval_idx, cols], seed=seed)
    with contextlib.redirect_stdout(io.StringIO()):     # стек печатает свой прогресс, здесь он не нужен
        scores["stack_lgbm"] = S.level2(S.level1(df, pairs, cols, graph, fit_mask, seed), y, fit_idx, eval_idx, seed)["stack_lgbm"]
    if "hybrid" in scores:
        # смеси по рангам, логиты сети и вероятности бустинга напрямую не складываются
        scores["blend_hybrid_lgb"] = (M.rank01(scores["hybrid"]) + M.rank01(scores["lgb_base"])) / 2
        scores["blend_hybrid_stack"] = (M.rank01(scores["hybrid"]) + M.rank01(scores["stack_lgbm"])) / 2
    return scores, epochs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="out")
    parser.add_argument("--variants", default="hybrid,seq,tab")
    parser.add_argument("--seeds", default="11,22,33")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    variants = args.variants.split(",")
    out = Path(args.out)
    df, pairs = M.load_features(out)
    cols = M.select_features(df, out / "hypothesis_results.csv")
    graph = P.build_graph(pairs, pd.Index(df["cookie_id"]))        # граф нужен только стеку, с которым идёт сравнение
    seq = prepare_sequences(Path(args.data), pd.Index(df["cookie_id"]))
    y = df["target"].fillna(0).astype(int).values
    print(f"обрезано до {SEQ_LEN} событий: {(df['n_events'] > SEQ_LEN).mean():.1%} кук")

    # хронологический сплит. Здесь ранняя остановка выбирает число эпох
    fit_mask, eval_idx = M.chronological_split(df)
    scores, epochs = run_split(df, pairs, seq, cols, graph, fit_mask, eval_idx, variants, seeds, {}, SEED)
    # для кросс-валидации число эпох фиксируется, среднее по сидам с запасом 10 %, но не меньше трёх
    n_epochs = {v: max(3, int(round(np.mean(epochs[v]) * 1.1))) for v in variants}
    chrono = {name: M.precision_at_recall(y[eval_idx], s) for name, s in scores.items()}
    print("эпохи по ранней остановке:", epochs, "для кросс-валидации:", n_epochs)
    print("неделя 0, затем неделя 1:", {k: round(v, 4) for k, v in chrono.items()}, flush=True)

    # кросс-валидация с фиксированным числом эпох
    cv = {name: [] for name in chrono}
    for k, (mask, held) in enumerate(M.cv_folds(df)):
        scores, _ = run_split(df, pairs, seq, cols, graph, mask, held, variants, seeds, n_epochs, SEED + 10 * (k + 1))
        for name, s in scores.items():
            cv[name].append(M.precision_at_recall(y[held], s))
        print(f"  fold {k}:", {n: round(v[-1], 4) for n, v in cv.items()}, flush=True)
    res = pd.DataFrame([{"model": name, "chrono_p_at_r70": chrono[name], "cv_folds_mean": float(np.mean(cv[name])),
                         "cv_folds_std": float(np.std(cv[name])),
                         # в скольких фолдах из пяти вариант обошёл базовый LightGBM и стек
                         "cv_wins_vs_lgb": int(sum(a > b for a, b in zip(cv[name], cv["lgb_base"]))),
                         "cv_wins_vs_stack": int(sum(a > b for a, b in zip(cv[name], cv["stack_lgbm"])))} for name in chrono]).round(4)
    res.to_csv(out / "seq_model_results.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
