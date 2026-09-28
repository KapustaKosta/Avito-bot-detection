"""
Обучает графовую нейросеть GraphSAGE на графе кука и кука по общим объявлениям, пишет
out/gnn_results.csv и с флагом --write файлы предсказаний out/submission_gnn.csv (только сеть)
и out/submission_gnn_stack.csv (смесь рангов сети и стека, финальный вариант решения).
Запускается командой python gnn_model.py --out out --write --device cuda --seeds 11,22,33
"""
import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import modeling as M
import propagate as P
import stack as S
from seed import SEED

MAX_EPOCHS, PATIENCE, LR, WEIGHT_DECAY = 300, 30, 5e-3, 1e-4
HIDDEN, DROPOUT, EDGE_DROPOUT = 64, 0.3, 0.2
# gnn два слоя (узел видит соседей своих соседей), gnn1 один слой,
# mlp та же сеть без графовых слоёв, она показывает вклад графа
LAYERS = {"gnn": 2, "gnn1": 1, "mlp": 0}
DEVICE = torch.device("cpu")


def node_features(d, cols, fit_idx):
    """Стандартизованные признаки узлов с индикаторами пропусков; сами пропуски заменены нулём."""
    # узловые признаки это те же подтверждённые табличные признаки, что и у бустинга
    X = d[cols].astype(float)
    # среднее и отклонение только по обучающей части, нулевое отклонение заменяется единицей
    mean, std = X.loc[fit_idx].mean(), X.loc[fit_idx].std().replace(0, 1).fillna(1)
    Z = ((X - mean) / std).clip(-6, 6)              # обрезка выбросов, иначе они ломают обучение сети
    missing = X.isna().astype(float).loc[:, X.isna().any()]     # индикатор пропуска на каждую колонку с пропусками
    return torch.from_numpy(np.hstack([Z.fillna(0).values, missing.values]).astype(np.float32))


def edges(graph):
    """Рёбра графа в виде трёх тензоров (откуда, куда, вес); веса нормированы по строке."""
    W = graph[0].tocoo()                            # координатный формат, по одной записи на ребро
    return (torch.from_numpy(W.row.astype(np.int64)), torch.from_numpy(W.col.astype(np.int64)),
            torch.from_numpy(W.data.astype(np.float32)))


class SageLayer(nn.Module):
    """Один слой GraphSAGE. Сумма линейного преобразования узла и линейного преобразования среднего по соседям."""

    def __init__(self, dim):
        super().__init__()
        self.self_linear = nn.Linear(dim, dim)
        self.neighbor_linear = nn.Linear(dim, dim, bias=False)

    def forward(self, h, src, dst, w):
        # index_add_ складывает в строку src взвешенные векторы её соседей dst;
        # веса нормированы по строке, поэтому получается взвешенное среднее с весами Жаккара
        neighbor_mean = torch.zeros_like(h).index_add_(0, src, h[dst] * w.unsqueeze(-1))
        return self.self_linear(h) + self.neighbor_linear(neighbor_mean)


class GraphSage(nn.Module):
    """Входной слой, n_layers слоёв GraphSAGE с остаточной связью и LayerNorm, линейная голова с одним логитом."""

    def __init__(self, n_inputs, n_layers):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(n_inputs, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT))
        self.layers = nn.ModuleList(SageLayer(HIDDEN) for _ in range(n_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(HIDDEN) for _ in range(n_layers))
        self.dropout = nn.Dropout(DROPOUT)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, x, src, dst, w):
        h = self.input(x)
        for layer, norm in zip(self.layers, self.norms):
            # остаточная связь h + ..., чтобы узел не растворялся в соседях; при n_layers = 0 цикл пуст
            h = h + self.dropout(torch.relu(norm(layer(h, src, dst, w))))
        return self.head(h).squeeze(-1)


def fit(x, graph_edges, y, train_idx, n_layers, seed, n_epochs=None, val_idx=None):
    """Обучение сети. Без n_epochs число эпох выбирает ранняя остановка по log-loss на val_idx.

    Весь граф помещается в память, поэтому каждая эпоха это один шаг по всем узлам сразу."""
    torch.manual_seed(seed)
    src, dst, w = (t.to(DEVICE) for t in graph_edges)
    x = x.to(DEVICE)
    model = GraphSage(x.shape[1], n_layers).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    y_train = torch.from_numpy(y[train_idx].astype(np.float32)).to(DEVICE)
    train_t = torch.from_numpy(np.asarray(train_idx)).to(DEVICE)
    best_loss, best_epoch, best_state, waited = np.inf, 0, None, 0
    for epoch in range(1, (n_epochs or MAX_EPOCHS) + 1):
        model.train()
        optimizer.zero_grad()
        # выбрасывание рёбер. Часть рёбер случайно убирается, оставшиеся веса масштабируются, чтобы сохранить
        # среднее; так сеть не заучивает конкретных соседей
        keep = torch.rand(len(w), device=DEVICE) >= EDGE_DROPOUT
        # обучение трансдуктивное. В графе присутствуют все куки, включая test, но потери считаются
        # только на размеченных узлах обучающей части
        loss = loss_fn(model(x, src[keep], dst[keep], w[keep] / (1 - EDGE_DROPOUT))[train_t], y_train)
        loss.backward()
        optimizer.step()
        if n_epochs is None:
            # ранняя остановка. Ошибка на валидации считается по полному графу без выбрасывания рёбер
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(x, src, dst, w)[torch.from_numpy(np.asarray(val_idx)).to(DEVICE)],
                                         torch.from_numpy(y[val_idx].astype(np.float32)).to(DEVICE)))
            if val_loss < best_loss - 1e-4:
                best_loss, best_epoch, waited = val_loss, epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}    # копия лучших весов
            elif (waited := waited + 1) >= PATIENCE:
                break                               # PATIENCE эпох без улучшения
    if best_state is not None:
        model.load_state_dict(best_state)       # возвращаемся к лучшей эпохе
        return model, best_epoch
    return model, n_epochs


def predict(model, x, graph_edges, idx):
    """Логиты сети для узлов idx по полному графу."""
    model.eval()
    with torch.no_grad():
        logits = model(x.to(DEVICE), *(t.to(DEVICE) for t in graph_edges))
    return logits[torch.from_numpy(np.asarray(idx)).to(DEVICE)].cpu().numpy()


def run_split(df, pairs, cols, graph, fit_mask, eval_idx, variants, seeds, n_epochs, seed):
    """Логиты сети по вариантам, score LightGBM и стека на eval_idx, плюс число эпох ранней остановки."""
    d = M.split_dependent_features(df, pairs, fit_mask, seed)     # признаки с метками только по меткам fit_mask
    fit_idx, eval_idx = df.index[fit_mask].values, np.asarray(eval_idx)
    y = d["target"].fillna(0).astype(int).values
    x = node_features(d, cols, fit_idx)
    graph_edges = edges(graph)
    scores, epochs = {}, {}
    for variant in variants:
        logits = []
        for s in seeds:
            # если число эпох для варианта уже известно, валидация не нужна и ранняя остановка выключена
            model, ep = fit(x, graph_edges, y, fit_idx, LAYERS[variant], seed + s, n_epochs.get(variant),
                            eval_idx if variant not in n_epochs else None)
            logits.append(predict(model, x, graph_edges, eval_idx))
            epochs.setdefault(variant, []).append(ep)
        scores[variant] = np.mean(logits, axis=0)   # усреднение логитов по сидам
    # точки отсчёта на тех же данных, базовый LightGBM и стек
    scores["lgb_base"], _, _ = M.fit_lightgbm(d.loc[fit_idx, cols], y[fit_idx], d.loc[eval_idx, cols], seed=seed)
    with contextlib.redirect_stdout(io.StringIO()):     # стек печатает свой прогресс, здесь он не нужен
        scores["stack_lgbm"] = S.level2(S.level1(df, pairs, cols, graph, fit_mask, seed), y, fit_idx, eval_idx, seed)["stack_lgbm"]
    if "gnn" in scores:
        # смеси по рангам, логиты сети и вероятности бустинга напрямую не складываются
        scores["blend_gnn_lgb"] = (M.rank01(scores["gnn"]) + M.rank01(scores["lgb_base"])) / 2
        scores["blend_gnn_stack"] = (M.rank01(scores["gnn"]) + M.rank01(scores["stack_lgbm"])) / 2
    return scores, epochs


def main():
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="out")
    parser.add_argument("--variants", default="gnn,gnn1,mlp")
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--device", default="cpu", help="cpu или cuda")
    parser.add_argument("--write", action="store_true", help="записать файлы предсказаний для test")
    args = parser.parse_args()
    # cuda берётся только если она действительно доступна, иначе тихий откат на cpu
    DEVICE = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]
    variants = args.variants.split(",")
    print(f"устройство {DEVICE}, сиды {seeds}")

    out = Path(args.out)
    df, pairs = M.load_features(out)
    cols = M.select_features(df, out / "hypothesis_results.csv")
    graph = P.build_graph(pairs, pd.Index(df["cookie_id"]))
    y = df["target"].fillna(0).astype(int).values

    # хронологический сплит. Здесь ранняя остановка выбирает число эпох
    fit_mask, eval_idx = M.chronological_split(df)
    scores, epochs = run_split(df, pairs, cols, graph, fit_mask, eval_idx, variants, seeds, {}, SEED)
    # для кросс-валидации число эпох фиксируется, среднее по сидам с запасом 10 %, но не меньше пяти
    n_epochs = {v: max(5, int(round(np.mean(epochs[v]) * 1.1))) for v in variants}
    chrono = {name: M.precision_at_recall(y[eval_idx], s) for name, s in scores.items()}
    print("эпохи по ранней остановке:", epochs, "для кросс-валидации:", n_epochs)
    print("неделя 0, затем неделя 1:", {k: round(v, 4) for k, v in chrono.items()}, flush=True)

    # кросс-валидация с фиксированным числом эпох
    cv = {name: [] for name in chrono}
    for k, (mask, held) in enumerate(M.cv_folds(df)):
        scores, _ = run_split(df, pairs, cols, graph, mask, held, variants, seeds, n_epochs, SEED + 10 * (k + 1))
        for name, s in scores.items():
            cv[name].append(M.precision_at_recall(y[held], s))
        print(f"  fold {k}:", {n: round(v[-1], 4) for n, v in cv.items()}, flush=True)
    res = pd.DataFrame([{"model": name, "chrono_p_at_r70": chrono[name], "cv_folds_mean": float(np.mean(cv[name])),
                         "cv_folds_std": float(np.std(cv[name])),
                         # в скольких фолдах из пяти вариант обошёл базовый LightGBM и стек
                         "cv_wins_vs_lgb": int(sum(a > b for a, b in zip(cv[name], cv["lgb_base"]))),
                         "cv_wins_vs_stack": int(sum(a > b for a, b in zip(cv[name], cv["stack_lgbm"])))} for name in chrono]).round(4)
    res.to_csv(out / "gnn_results.csv", index=False)
    print(res.to_string(index=False))

    if args.write and "gnn" in variants:
        # финальная модель. Обучение на всём train, для test к сидам добавляются ещё два
        fit_mask = df["is_test"] == 0
        test_idx = df.index[~fit_mask]
        scores, _ = run_split(df, pairs, cols, graph, fit_mask, test_idx, ["gnn"], seeds + [SEED + 33, SEED + 44], n_epochs, SEED)
        ids = df.loc[test_idx, "cookie_id"].values
        # у сети логиты переводятся в вероятность, у смеси score уже лежит в отрезке от 0 до 1
        pd.DataFrame({"cookie_id": ids, "score": 1 / (1 + np.exp(-scores["gnn"]))}).to_csv(out / "submission_gnn.csv", index=False)
        pd.DataFrame({"cookie_id": ids, "score": scores["blend_gnn_stack"]}).to_csv(out / "submission_gnn_stack.csv", index=False)
        print(f"записаны {out / 'submission_gnn.csv'} и {out / 'submission_gnn_stack.csv'}")


if __name__ == "__main__":
    main()
