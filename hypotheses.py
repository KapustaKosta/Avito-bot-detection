"""
Проверяет гипотезы о признаках ботов в две стадии, скрининг на первой неделе train и подтверждение
на второй, и пишет out/hypothesis_results.csv со строкой на каждую пару (гипотеза, признак).
Запускается командой python hypotheses.py --features out/features.csv --out out
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from seed import SEED

# стадия 1, скрининг. Все пары проверяются вместе, p-значения корректируются по Бенджамини и Хохбергу,
# чтобы среди сотни проверок не набрать случайных находок
Q_SCREEN = 0.01
# стадия 2, подтверждение. Только для пар, прошедших скрининг, на неделе, которая в отборе не участвовала
ALPHA_CONFIRM = 0.01
Q_CONFIRM = 0.05
# пороги эффекта. При 11 тысячах кук значимым становится почти всё, критерий отвергает нулевую
# гипотезу уже при AUC 0,52, а такая разница модели бесполезна
AUC_MIN = 0.55                  # AUC в ожидаемую сторону, бутстрэп-интервал должен не накрывать 0,5
FLAG_MIN_SUPPORT = 30           # у флага не меньше 30 кук с поднятым флагом
FLAG_MIN_PRECISION = 0.25       # доля ботов среди них втрое выше базовой доли ботов
FLAG_MIN_WILSON_LOW = 0.16      # нижняя граница интервала Уилсона для этой доли
CRAMER_MIN = 0.10               # V Крамера для категориального признака
N_BOOTSTRAP = 300

# реестр проверок в виде (гипотеза, признак, тип теста, направление, короткое описание).
# Типы тестов. cont это непрерывный признак, критерий Манна и Уитни, эффект равен AUC.
# flag это бинарный признак, точный критерий Фишера, эффект равен доле ботов среди кук с флагом.
# ks это форма распределения, двухвыборочный критерий Колмогорова и Смирнова, эффект равен расстоянию.
# cat это категориальный признак, хи-квадрат, эффект равен V Крамера.
# Направление higher или lower означает односторонний тест в сторону "у ботов выше" или "ниже",
# two означает двусторонний тест
HYPOTHESES = [
    ("A1", "ua_nonbrowser", "flag", "higher", "User-Agent HTTP-библиотеки"),
    ("A2", "browser_lag_mode", "cont", "higher", "отставание версии браузера"),
    ("A2", "browser_lag_ge_6", "flag", "higher", "отставание не меньше 6 версий"),
    ("A4", "is_firefox", "flag", "two", "Firefox"),
    ("A5", "n_distinct_ua", "cont", "higher", "несколько User-Agent в куке"),
    ("A5", "ua_switch_count", "cont", "higher", "смены User-Agent"),
    ("A7", "ua_headless", "flag", "higher", "HeadlessChrome"),
    ("A8", "app_version_rank", "cont", "higher", "редкая версия приложения"),
    ("A9", "is_web_mode", "flag", "higher", "платформа web"),
    ("A10", "ua_cookie_count", "cont", "higher", "число кук с тем же User-Agent"),
    ("A10", "ua_hash_bot_rate_oof", "cont", "higher", "доля ботов по User-Agent"),
    ("B1", "iei_median", "cont", "lower", "медианный интервал"),
    ("B2", "iei_cv", "cont", "lower", "коэффициент вариации интервалов"),
    ("B2", "iei_iqr_over_median", "cont", "lower", "IQR к медиане"),
    ("B2", "iei_log_std", "cont", "lower", "разброс log-интервалов"),
    ("B3", "iei_mode_share", "cont", "higher", "доля интервалов у моды"),
    ("B4", "iei_round_share", "cont", "higher", "круглые периоды"),
    ("B5", "long_pause_share", "cont", "lower", "доля пауз дольше 5 минут"),
    ("B5", "max_gap_ratio", "cont", "lower", "максимальная пауза"),
    ("B6", "iei_acf1", "cont", "higher", "автокорреляция интервалов"),
    ("B7", "night_share", "cont", "higher", "ночная доля"),
    ("B8", "udist_hour_entropy", "cont", "higher", "расстояние энтропии часов от людей"),
    ("B8", "hour_entropy", "ks", "two", "форма энтропии часов"),
    ("B9", "max_events_per_min", "cont", "higher", "всплеск в минуту"),
    ("B10", "udist_n_events", "cont", "higher", "расстояние числа событий от людей"),
    ("B10", "n_events", "ks", "two", "форма числа событий"),
    ("B11", "udist_active_span_ratio", "cont", "higher", "расстояние активного промежутка"),
    ("B12", "udist_n_sessions", "cont", "higher", "расстояние числа сессий"),
    ("B13", "iei_share_2_15s", "cont", "higher", "доля интервалов от 2 до 15 секунд"),
    ("B14", "sec_aligned_share", "cont", "higher", "секунды 0 и 30"),
    ("B14", "minute_phase_concentration", "cont", "higher", "концентрация по секунде минуты"),
    ("B17", "share_iei_eq_0", "cont", "higher", "нулевые интервалы"),
    ("B17", "iei_min", "cont", "lower", "минимальный интервал"),
    ("B18", "first_event_at_creation", "flag", "higher", "первое событие в момент создания куки"),
    ("B18", "first_event_minus_created_s", "cont", "lower", "от создания куки до первого события"),
    ("C1", "created_in_window", "flag", "higher", "кука создана в окне"),
    ("C2", "cookie_age_h", "cont", "lower", "возраст куки"),
    ("C3", "cookie_age_lt_12h", "flag", "higher", "возраст меньше 12 часов"),
    ("C3", "cookie_age_h", "ks", "two", "форма возраста"),
    ("C4", "old_cookie_burst", "flag", "higher", "старая кука со всплеском"),
    ("C4", "age_x_events", "cont", "two", "возраст умножить на события"),
    ("C5", "window_weekday", "cat", "two", "день недели окна"),
    ("C6", "first_event_hour", "ks", "two", "час первого события"),
    ("D1", "serp_share", "cont", "higher", "доля выдачи"),
    ("D1", "item_share", "cont", "lower", "доля карточек"),
    ("D2", "udist_items_per_serp", "cont", "higher", "расстояние карточек на выдачу"),
    ("D2", "items_per_serp", "ks", "two", "форма карточек на выдачу"),
    ("D3", "photo_share", "cont", "lower", "доля фото"),
    ("D3", "photos_per_item_std", "cont", "lower", "разброс свайпов на карточку"),
    ("D4", "seller_per_item", "ks", "two", "продавец на карточку"),
    ("D5", "contact_intensity", "cont", "higher", "контакты к фото и продавцу"),
    ("D6", "contact_without_item_share", "cont", "higher", "контакт без карточки"),
    ("D7", "fav_per_item", "cont", "two", "избранное на карточку"),
    ("D8", "login_gt_1", "flag", "higher", "больше одного входа"),
    ("D8", "n_login", "cont", "two", "число входов"),
    ("D10", "event_type_entropy", "cont", "lower", "энтропия типов событий"),
    ("D11", "bigram_serp_serp", "cont", "higher", "выдача за выдачей"),
    ("D11", "bigram_item_item", "cont", "higher", "карточка за карточкой"),
    ("D11", "bigram_item_photo", "cont", "lower", "фото после карточки"),
    ("D11", "bigram_photo_photo", "cont", "lower", "фото за фото"),
    ("D12", "dup_share", "cont", "higher", "повторы за 5 секунд"),
    ("D13", "first_is_item_no_search", "flag", "higher", "первое событие карточка без поиска"),
    ("D13", "first_item_random_id", "flag", "higher", "первая карточка одноразовая"),
    ("D14", "max_search_page", "cont", "higher", "максимальная страница выдачи"),
    ("D14", "page_ge_10_any", "flag", "higher", "есть страница не меньше 10"),
    ("D15", "page_step_eq_1_share", "cont", "higher", "шаг страниц равен 1"),
    ("D15", "page_run_max", "cont", "higher", "длина цепочки страниц"),
    ("D16", "max_query_repeats", "cont", "higher", "повторы запроса"),
    ("D16", "query_repeat_cv", "cont", "lower", "разброс периода повторов"),
    ("D18", "udist_n_distinct_queries", "cont", "higher", "расстояние числа запросов"),
    ("D19", "query_overlap_mean", "cont", "lower", "пересечение соседних запросов"),
    ("E1", "udist_n_categories", "cont", "higher", "расстояние числа категорий"),
    ("E1", "n_categories", "ks", "two", "форма числа категорий"),
    ("E2", "n_locations", "cont", "higher", "число локаций"),
    ("E3", "private_seller_share", "cont", "two", "доля частных продавцов"),
    ("E4", "n_items", "cont", "higher", "число объявлений"),
    ("E4", "items_per_min", "cont", "higher", "объявлений в минуту"),
    ("E5", "item_revisit_share", "cont", "lower", "повторные просмотры"),
    ("E6", "item_id_monotone_share", "cont", "higher", "монотонность item_id"),
    ("E7", "items_shared_cookies_mean", "cont", "higher", "общие объявления с другими куками"),
    ("E7", "neighbor_jaccard_max", "cont", "higher", "максимальный Жаккар с соседом"),
    ("E7", "neighbor_bot_rate_oof", "cont", "higher", "доля ботов среди соседей"),
    ("E8", "cat_loc_grid_ratio", "cont", "lower", "решётка категория и локация"),
    ("F1", "pointer_missing_share_web", "cont", "higher", "веб без курсора"),
    ("F2", "pointer_zero_share", "cont", "higher", "точка (0, 0)"),
    ("F2", "pointer_top1_share", "cont", "higher", "одна фиксированная точка"),
    ("F3", "pointer_unique_ratio", "cont", "lower", "уникальные позиции"),
    ("F4", "pointer_repeat_share", "cont", "higher", "повтор координат"),
    ("F5", "udist_jump_dist_median", "cont", "higher", "расстояние прыжков от людей"),
    ("F5", "jump_zero_share", "cont", "higher", "нулевые прыжки"),
    ("F5", "jump_gt_500_share", "cont", "higher", "прыжки больше 500 px"),
    ("F6", "pointer_bbox_area_log", "cont", "lower", "площадь охвата"),
    ("F6", "pointer_std_x", "cont", "lower", "разброс по x"),
    ("F7", "viewport_default_flag", "flag", "higher", "стандартный размер окна"),
    ("F7", "max_x", "ks", "two", "форма максимума x"),
    ("F10", "pointer_valid_share_var", "cont", "higher", "смесь событий с курсором и без"),
    ("F11", "collinear_triplet_share", "cont", "higher", "коллинеарные тройки"),
    ("F12", "pointer_by_class_eta2_x", "cont", "lower", "зависимость координат от типа события"),
    ("F13", "coord_round10_share", "cont", "higher", "координаты кратны 10"),
    ("F13", "coord_half_share", "cont", "higher", "координаты с половиной"),
]


def auc_from_u(u, n_bot, n_human):
    """Статистика Манна и Уитни, делённая на произведение размеров выборок, равна AUC."""
    return u / (n_bot * n_human)


def bootstrap_auc_interval(x_bot, x_human, rng):
    """Бутстрэп-интервал AUC. Выборки ботов и людей пересобираются с возвращением N_BOOTSTRAP раз."""
    aucs = []
    for _ in range(N_BOOTSTRAP):
        b = rng.choice(x_bot, len(x_bot))           # выборка с возвращением того же размера
        h = rng.choice(x_human, len(x_human))
        u = stats.mannwhitneyu(b, h, alternative="two-sided", method="asymptotic").statistic
        aucs.append(auc_from_u(u, len(b), len(h)))
    return np.percentile(aucs, [2.5, 97.5])         # границы 95 % интервала


def test_pair(part, feature, kind, direction, rng):
    """Тест одной пары на одной части данных. Возвращает p, размер эффекта и флаг, что эффект достаточен."""
    y = part["target"].astype(int)
    x = part[feature]
    if x.notna().sum() == 0 or (kind != "cat" and x.dropna().nunique() <= 1):
        return {"p": np.nan, "effect": np.nan, "effect_ok": False}     # признак пуст или константен
    if kind == "cont":
        bot, human = x[y == 1].dropna().values, x[y == 0].dropna().values
        # односторонняя альтернатива в ожидаемую сторону, двусторонняя для направления two
        alt = {"higher": "greater", "lower": "less", "two": "two-sided"}[direction]
        u = stats.mannwhitneyu(bot, human, alternative=alt, method="asymptotic")
        auc = auc_from_u(u.statistic, len(bot), len(human))     # вероятность, что случайный бот выше случайного человека
        lo, hi = bootstrap_auc_interval(bot, human, rng)
        # для направления lower хороший AUC меньше 0,5, поэтому порог зеркальный
        ok = {"higher": auc >= AUC_MIN and lo > 0.5,
              "lower": auc <= 1 - AUC_MIN and hi < 0.5,
              "two": abs(auc - 0.5) >= AUC_MIN - 0.5 and (lo > 0.5 or hi < 0.5)}[direction]
        return {"p": float(u.pvalue), "effect": auc, "effect_ok": bool(ok)}
    if kind == "flag":
        # таблица два на два. Строки это флаг есть или нет, столбцы это бот или человек
        flag = x.fillna(0).astype(int)
        a, b = int(((flag == 1) & (y == 1)).sum()), int(((flag == 1) & (y == 0)).sum())
        c, d = int(((flag == 0) & (y == 1)).sum()), int(((flag == 0) & (y == 0)).sum())
        support = a + b                             # сколько кук подняли флаг
        alt = {"higher": "greater", "lower": "less", "two": "two-sided"}[direction]
        p = stats.fisher_exact([[a, b], [c, d]], alternative=alt)[1] if support else np.nan
        precision = a / support if support else np.nan          # доля ботов среди кук с флагом
        wilson_low = proportion_confint(a, support, method="wilson")[0] if support else np.nan
        base = (a + c) / len(y)                     # базовая доля ботов в этой части данных
        # для lower флаг должен, наоборот, собирать людей, то есть долю ботов втрое ниже базовой
        ok = support >= FLAG_MIN_SUPPORT and (
            precision <= base / 3 if direction == "lower"
            else precision >= FLAG_MIN_PRECISION and wilson_low >= FLAG_MIN_WILSON_LOW)
        return {"p": p, "effect": precision, "effect_ok": bool(ok)}
    if kind == "ks":
        # сравнивается форма распределения, направление здесь не задаётся
        bot, human = x[y == 1].dropna().values, x[y == 0].dropna().values
        r = stats.ks_2samp(bot, human)
        return {"p": float(r.pvalue), "effect": float(r.statistic), "effect_ok": bool(r.statistic >= 0.10)}
    if kind == "cat":
        table = pd.crosstab(x, y)                   # категории по строкам, метка по столбцам
        stat, p, _, _ = stats.chi2_contingency(table.values, correction=False)
        # V Крамера нормирует хи-квадрат на размер выборки, у таблицы с двумя столбцами min(shape) - 1 = 1
        cramer = np.sqrt(stat / (table.values.sum() * (min(table.shape) - 1)))
        return {"p": float(p), "effect": float(cramer), "effect_ok": bool(cramer >= CRAMER_MIN)}
    raise ValueError(kind)


def run(features_path):
    """Обе стадии проверки. Возвращает таблицу с p, эффектами и вердиктом по каждой паре."""
    df = pd.read_csv(features_path)
    train = df[df["is_test"] == 0]
    weeks = sorted(train["week_id"].unique())
    # если отобрать лучшие признаки и оценить их эффект на тех же данных, оценка окажется завышенной.
    # Поэтому скрининг идёт на первой неделе, а подтверждение на второй, которая в отборе не участвовала.
    # Заодно это проверяет устойчивость во времени, test тоже начинается позже train
    screen, confirm = train[train["week_id"] == weeks[0]], train[train["week_id"] == weeks[1]]
    rng = np.random.default_rng(SEED)               # один генератор на весь прогон, бутстрэп воспроизводим

    # стадия 1, все пары
    rows = []
    for hyp, feature, kind, direction, comment in HYPOTHESES:
        r1 = test_pair(screen, feature, kind, direction, rng)
        rows.append({"hyp": hyp, "feature": feature, "kind": kind, "direction": direction, "comment": comment,
                     "s1_p": r1["p"], "s1_effect": r1["effect"], "s1_effect_ok": r1["effect_ok"]})
    res = pd.DataFrame(rows)
    res["s1_p_adj"] = np.nan
    tested = res["s1_p"].notna()                    # пары с пустым признаком в коррекцию не входят
    res.loc[tested, "s1_p_adj"] = multipletests(res.loc[tested, "s1_p"], method="fdr_bh")[1]
    res["s1_pass"] = (res["s1_p_adj"] <= Q_SCREEN) & res["s1_effect_ok"]       # и значимость, и размер эффекта

    # стадия 2, только для прошедших скрининг, с тем же направлением
    res["s2_p"], res["s2_effect"], res["s2_effect_ok"] = np.nan, np.nan, False
    for i in res.index[res["s1_pass"]]:
        r2 = test_pair(confirm, res.at[i, "feature"], res.at[i, "kind"], res.at[i, "direction"], rng)
        res.loc[i, ["s2_p", "s2_effect", "s2_effect_ok"]] = r2["p"], r2["effect"], r2["effect_ok"]
    res["s2_p_adj"] = np.nan
    tested = res["s2_p"].notna()
    if tested.any():
        res.loc[tested, "s2_p_adj"] = multipletests(res.loc[tested, "s2_p"], method="fdr_bh")[1]
    # на второй стадии требуется и сырое p ниже alpha, и скорректированное ниже q
    res["s2_pass"] = (res["s2_p"] <= ALPHA_CONFIRM) & (res["s2_p_adj"] <= Q_CONFIRM) & res["s2_effect_ok"]
    res["confirmed"] = res["s1_pass"] & res["s2_pass"]
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="out/features.csv")
    parser.add_argument("--out", default="out")
    args = parser.parse_args()
    res = run(args.features)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    res.to_csv(Path(args.out) / "hypothesis_results.csv", index=False)
    print(f"пар проверено: {len(res)}, прошли скрининг: {int(res['s1_pass'].sum())}, подтверждены: {int(res['confirmed'].sum())}")
    print("подтверждённые признаки:", sorted(res.loc[res["confirmed"], "feature"].unique()))


if __name__ == "__main__":
    main()
