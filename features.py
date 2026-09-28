"""
Считает признаки по кукам из журнала событий и пишет их в папку out.
Запускается командой python features.py --data data --out out
Записывает features.csv (строка на куку train и test, признаки и метка), neighbor_pairs.csv
(пары кук с общими объявлениями и мера Жаккара) и feature_groups.json (какие колонки к какой
группе относятся). Мотивация признаков описана в ноутбуке solution.ipynb, раздел 3.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy

from seed import SEED

# коды событий из данных и их классы
EVENT_CLASS = {100: "serp", 200: "item", 210: "photo", 220: "seller",
               300: "contact", 301: "contact", 303: "contact",
               400: "fav", 500: "login", 900: "captcha"}
CLASSES = ["serp", "item", "photo", "seller", "contact", "fav", "login", "captcha"]
# переходы между соседними событиями, доли которых становятся признаками
BIGRAMS = [("serp", "serp"), ("serp", "item"), ("item", "item"), ("item", "photo"), ("photo", "photo"),
           ("item", "seller"), ("item", "contact"), ("item", "fav"), ("serp", "contact")]
SESSION_GAP = 1800          # разрыв больше 30 минут начинает новую сессию
DUPLICATE_WINDOW = 5        # то же событие в течение 5 секунд считается повтором (ретрай скрипта)
LONG_PAUSE = 300            # пауза длиннее 5 минут считается длинной
ROUND_PERIODS = np.array([30, 60, 120, 180, 300, 600, 900, 1800, 3600], dtype=float)   # типичные периоды опроса
DEFAULT_VIEWPORTS = [(1280, 720), (1920, 1080), (1200, 600), (1280, 800), (1440, 900), (1366, 768)]  # размеры окна из кода парсеров
HTTP_LIBRARY = re.compile(r"python-requests|python-urllib3|curl/|go-http-client|node-fetch|scrapy|aiohttp|httpx|wget|java/", re.I)
MOSCOW_OFFSET_HOURS = 3     # время событий записано в UTC, часы суток считаются по Москве


# ----------------------------------------------------------------------------- подготовка событий
def parse_user_agent(ua):
    """Класс, семейство браузера, версия и признаки платформы из строки User-Agent.

    Классы такие. browser это обычный браузер, app это приложение Авито, headless это HeadlessChrome,
    nonbrowser это HTTP-библиотека вроде python-requests, empty это пустая строка."""
    s = ua.fillna("").astype(str).str.strip()
    out = pd.DataFrame(index=ua.index)
    is_app = s.str.startswith("Avito/")
    is_headless = s.str.contains("HeadlessChrome|PhantomJS")
    is_browser = s.str.startswith("Mozilla/") & ~is_headless
    is_library = s.str.contains(HTTP_LIBRARY) | (s == "")
    # np.select берёт первое подходящее условие, поэтому порядок задаёт приоритет
    out["ua_class"] = np.select([s == "", is_headless, is_app, is_library, is_browser],
                                ["empty", "headless", "app", "nonbrowser", "browser"], default="other")
    out["browser_family"] = np.select(
        [is_headless, is_app, s.str.contains("YaBrowser"), s.str.contains("Firefox/"),
         s.str.contains("Chrome/") & s.str.contains("Mobile"), s.str.contains("Chrome/"),
         s.str.contains("iPhone OS") & s.str.contains("Safari"), is_library],
        ["headless_chrome", "avito_app", "yabrowser", "firefox", "chrome_mobile", "chrome", "safari_ios", "http_lib"],
        default="other")
    # старшая версия браузера или приложения, у строк без версии получается NaN
    out["browser_major"] = pd.to_numeric(s.str.extract(r"(?:HeadlessChrome|Chrome|Firefox|Version|Avito)/(\d+)")[0], errors="coerce")
    out["android_major"] = pd.to_numeric(s.str.extract(r"Android (\d+)")[0], errors="coerce")
    out["app_version"] = s.where(is_app).str.extract(r"Avito/(\d+)")[0]
    out["ua_mobile"] = s.str.contains("Mobile|Android|iPhone") | is_app
    out["ua_headless"] = is_headless.astype(int)
    return out


def preprocess(events, meta):
    """События внутри окна наблюдения с нормализованными полями и разобранным User-Agent."""
    ev = events.merge(meta[["cookie_id", "window_start_ts", "window_end_ts", "is_test"]], on="cookie_id")
    ev["ts"] = pd.to_datetime(ev["event_ts"])
    ev["ws"] = pd.to_datetime(ev["window_start_ts"])
    ev["we"] = pd.to_datetime(ev["window_end_ts"])
    # в train часть событий записана после конца окна, в test таких событий нет. Любой признак на них
    # поднял бы кросс-валидацию и ничего не дал бы на test, поэтому они отбрасываются
    ev = ev[(ev["ts"] >= ev["ws"]) & (ev["ts"] < ev["we"])].copy()
    ev["cls"] = ev["eid"].map(EVENT_CLASS).fillna("other")
    # платформа записана в произвольном регистре, desktop и web это один веб, iphone это ios
    ev["platform_norm"] = ev["platform"].str.lower().str.strip().replace({"desktop": "web", "iphone": "ios"})
    ev["is_web"] = ev["platform_norm"] == "web"
    ev = pd.concat([ev, parse_user_agent(ev["user_agent"])], axis=1)
    ev["ua_norm"] = ev["user_agent"].fillna("").str.strip().str.lower()     # одна строка в разном регистре считается одной
    # User-Agent не соответствует платформе. Мобильный User-Agent на вебе, десктопный в приложении и наоборот
    ev["ua_platform_mismatch"] = ((ev["ua_mobile"] & ev["is_web"])
                                  | (~ev["ua_mobile"] & ev["platform_norm"].isin(["ios", "android"]) & (ev["ua_class"] != "app"))
                                  | ((ev["ua_class"] == "app") & ev["is_web"])).astype(int)
    ev["pointer_valid"] = ev["pointer_x"].notna() & ev["pointer_y"].notna()
    ev["hour"] = (ev["ts"] + pd.Timedelta(hours=MOSCOW_OFFSET_HOURS)).dt.hour
    ev["sec"] = ev["ts"].dt.second
    ev["week_id"] = ((ev["ws"] - ev["ws"].min()).dt.days // 7).astype(int)     # неделя окна от начала данных
    # события каждой куки по времени, при равном времени по коду события, чтобы порядок был воспроизводим
    ev = ev.sort_values(["cookie_id", "ts", "eid"]).reset_index(drop=True)
    g = ev.groupby("cookie_id", sort=False)
    ev["dt_prev"] = g["ts"].diff().dt.total_seconds()          # пауза до предыдущего события куки
    # первое событие куки всегда открывает сессию (пауза бесконечна), дальше новая сессия после долгой паузы
    ev["session_id"] = (ev["dt_prev"].fillna(np.inf) > SESSION_GAP).groupby(ev["cookie_id"]).cumsum()
    # повтор. То же событие с тем же объявлением, запросом и страницей в течение DUPLICATE_WINDOW секунд
    key = (ev["cls"] + "|" + ev["item_id"].fillna(-1).astype(int).astype(str) + "|"
           + ev["search_query"].fillna("").str.lower().str.strip() + "|" + ev["search_page"].fillna(-1).astype(int).astype(str))
    same_key_gap = ev.groupby(["cookie_id", key], sort=False)["ts"].diff().dt.total_seconds()
    ev["is_dup"] = (same_key_gap <= DUPLICATE_WINDOW).fillna(False).astype(int)
    # отставание версии браузера от самой свежей версии того же семейства на той же неделе;
    # абсолютная версия быстро устаревает, отставание переносится между неделями
    newest = ev.groupby(["week_id", "browser_family"])["browser_major"].transform("max")
    ev["browser_lag"] = newest - ev["browser_major"]
    return ev


# ----------------------------------------------------------------------------- признаки одной куки
def autocorrelation(x, lag):
    """Выборочная автокорреляция ряда с заданным лагом."""
    x = x - x.mean()
    denom = (x * x).sum()
    return float((x[:-lag] * x[lag:]).sum() / denom) if denom > 0 else np.nan     # у константного ряда автокорреляция не определена


def norm_entropy(counts, k):
    """Энтропия распределения, поделённая на максимум log(k), чтобы лежать в отрезке от 0 до 1."""
    return entropy(np.asarray(counts, dtype=float)) / np.log(k) if k > 1 and np.sum(counts) > 0 else np.nan


def rhythm_features(ts, hours, secs, session_id):
    """Ритм. Интервалы между событиями, паузы, всплески, время суток, сессии.

    ts это времена событий в секундах, dt это интервалы между соседними событиями."""
    n = len(ts)
    f = {"n_events": n, "few_events": int(n < 3)}
    dt = np.diff(ts).astype(float)
    span = float(ts[-1] - ts[0]) if n > 1 else 0.0      # активный промежуток куки
    f["active_span_s"] = span
    if len(dt) >= 2:                                    # статистики интервалов имеют смысл от двух интервалов
        med, mean = np.median(dt), dt.mean()
        q25, q75 = np.percentile(dt, [25, 75])
        f["iei_median"], f["iei_min"] = med, dt.min()
        f["iei_cv"] = dt.std() / mean if mean > 0 else np.nan               # коэффициент вариации, у робота маленький
        f["iei_iqr_over_median"] = (q75 - q25) / med if med > 0 else np.nan
        f["iei_log_std"] = np.log1p(dt).std()
        # доля интервалов около моды. Строится гистограмма log-интервалов, берётся самый частый столбец,
        # его центр переводится обратно в секунды и считается, сколько интервалов попало в плюс-минус 10 %
        hist, edges = np.histogram(np.log1p(dt), bins=max(5, min(30, len(dt))))
        mode = np.expm1((edges[hist.argmax()] + edges[hist.argmax() + 1]) / 2)
        f["iei_mode_share"] = float(((dt >= mode * 0.9) & (dt <= mode * 1.1)).mean())
        # доля интервалов, отличающихся от одного из типичных периодов опроса не больше чем на 5 %
        near_round = np.abs(dt[:, None] - ROUND_PERIODS[None, :]) / ROUND_PERIODS[None, :] <= 0.05
        f["iei_round_share"] = float(near_round.any(axis=1).mean())
        f["long_pause_share"] = float((dt > LONG_PAUSE).mean())
        f["iei_acf1"] = autocorrelation(dt, 1) if len(dt) >= 10 else np.nan   # на коротком ряде автокорреляция слишком шумная
        f["iei_share_2_15s"] = float(((dt >= 2) & (dt <= 15)).mean())
        f["share_iei_eq_0"] = float((dt == 0).mean())
    if span > 0:
        f["max_gap_ratio"] = dt.max() / span            # какая часть промежутка приходится на одну паузу
    if n >= 2:
        # для каждого события считается, сколько событий уложилось в следующие 60 секунд
        ends = np.searchsorted(ts, ts + 60, side="left")
        f["max_events_per_min"] = int((ends - np.arange(n)).max())
    else:
        f["max_events_per_min"] = n
    hour_counts = np.bincount(hours, minlength=24)
    f["night_share"] = float(((hours >= 2) & (hours < 6)).mean())
    f["hour_entropy"] = norm_entropy(hour_counts, 24)   # насколько равномерно события размазаны по часам
    f["first_event_hour"] = int(hours[0])
    f["n_sessions"] = int(len(np.unique(session_id)))
    # планировщики вроде cron запускаются на границе минуты, поэтому смотрим на секунды события
    f["sec_aligned_share"] = float(np.isin(secs, [0, 30]).mean())
    f["minute_phase_concentration"] = float(np.bincount(secs, minlength=60).max() / n)     # доля самой частой секунды
    return f


def composition_features(g, cls, ts):
    """Состав событий, переходы между ними, пагинация и поисковые запросы."""
    n = len(cls)
    f = {}
    counts = pd.Series(cls).value_counts()
    for c in CLASSES:
        f[f"n_{c}"] = int(counts.get(c, 0))
        f[f"{c}_share"] = counts.get(c, 0) / n
    f["event_type_entropy"] = norm_entropy(counts.values, len(CLASSES))
    # отношения между классами. Знаменатель не меньше единицы, чтобы не делить на ноль
    f["items_per_serp"] = f["n_item"] / max(f["n_serp"], 1)
    f["seller_per_item"] = f["n_seller"] / max(f["n_item"], 1)
    f["fav_per_item"] = f["n_fav"] / max(f["n_item"], 1)
    f["contact_intensity"] = f["n_contact"] / (f["n_photo"] + f["n_seller"] + 1)
    f["login_gt_1"] = int(f["n_login"] > 1)
    f["dup_share"] = float(g["is_dup"].mean())
    items = g["item_id"].values
    if f["n_item"] > 0:
        # число листаний фото после каждого просмотра объявления. События группируются по
        # порядковому номеру последнего просмотра объявления, блок 0 это события до первой карточки
        item_block = np.cumsum(cls == "item")
        photos = pd.Series(cls == "photo").groupby(item_block).sum()
        photos = photos[photos.index > 0]
        f["photos_per_item_std"] = float(photos.std(ddof=0)) if len(photos) else 0.0
    contact_idx = np.where(cls == "contact")[0]
    if len(contact_idx):
        # контакт без просмотра того же объявления среди пяти предыдущих событий
        without = [not ((cls[max(0, i - 5):i] == "item") & (items[max(0, i - 5):i] == items[i])).any() for i in contact_idx]
        f["contact_without_item_share"] = float(np.mean(without))
    if n >= 5:
        # доли переходов между соседними событиями, на очень коротких куках они слишком шумные
        prev, cur = cls[:-1], cls[1:]
        for a, b in BIGRAMS:
            f[f"bigram_{a}_{b}"] = float(((prev == a) & (cur == b)).mean())
    f["first_is_item_no_search"] = int(cls[0] == "item" and "serp" not in cls[:3])
    if cls[0] == "item":
        # первое событие это просмотр объявления, которое больше не встречается. Так парсер получает cookie
        f["first_item_random_id"] = int((items[1:] == items[0]).sum() == 0)
    serp = g[g["cls"] == "serp"]
    if len(serp):
        pages = serp["search_page"].values
        # запросы приводятся к одному виду, чтобы "Iphone  12" и "iphone 12" считались одним запросом
        queries = serp["search_query"].fillna("").str.lower().str.strip().str.replace(r"\s+", " ", regex=True).values
        f["max_search_page"] = float(np.nanmax(pages))
        f["page_ge_10_any"] = int((pages >= 10).any())
        # шаг между соседними просмотрами выдачи с тем же запросом. У краулера всегда плюс один
        same_query = queries[1:] == queries[:-1]
        if same_query.any():
            step = pages[1:][same_query] - pages[:-1][same_query]
            f["page_step_eq_1_share"] = float((step == 1).mean())
        # самая длинная цепочка страниц подряд по одному запросу
        run = best = 1
        for i in range(1, len(pages)):
            run = run + 1 if (same_query[i - 1] and pages[i] == pages[i - 1] + 1) else 1
            best = max(best, run)
        f["page_run_max"] = best
        f["n_distinct_queries"] = int(len(set(q for q in queries if q)))
        # самая частая пара (запрос, страница) и разброс интервалов между её повторами
        repeats = pd.Series(list(zip(queries, pages))).value_counts()
        f["max_query_repeats"] = int(repeats.iloc[0])
        top_mask = np.array([qp == repeats.index[0] for qp in zip(queries, pages)])
        top_ts = serp["ts"].values.astype("datetime64[s]").astype(np.int64)[top_mask]
        if len(top_ts) >= 4:                            # разброс периода имеет смысл от трёх интервалов
            rd = np.diff(top_ts).astype(float)
            f["query_repeat_cv"] = float(rd.std() / rd.mean()) if rd.mean() > 0 else np.nan
        # пересечение слов соседних различных запросов. Человек уточняет запрос, скрипт идёт по списку
        unique_queries = [q for q in pd.Series(queries).drop_duplicates() if q]
        if len(unique_queries) >= 2:
            tokens = [set(t for t in re.split(r"[^\w]+", q) if len(t) >= 2) for q in unique_queries]   # слова от двух букв
            jac = [len(a & b) / len(a | b) if (a | b) else 0.0 for a, b in zip(tokens[:-1], tokens[1:])]
            f["query_overlap_mean"] = float(np.mean(jac))
    return f


def diversity_features(g, cls, span):
    """Категории, локации, продавцы и объявления."""
    f = {}
    cat, loc = g["item_category"].dropna(), g["item_location"].dropna()
    f["n_categories"], f["n_locations"] = int(cat.nunique()), int(loc.nunique())
    # отношение произведения числа категорий и локаций к числу их сочетаний. Единица означает полную
    # решётку, то есть обход всех категорий во всех городах
    pairs = g[["item_category", "item_location"]].dropna().drop_duplicates()
    f["cat_loc_grid_ratio"] = f["n_categories"] * f["n_locations"] / len(pairs) if len(pairs) else np.nan
    seller = g.loc[g["cls"] == "item", "seller_type"]
    f["private_seller_share"] = float((seller == "private").mean()) if seller.notna().any() else np.nan
    items = g["item_id"].dropna()
    f["n_items"] = int(items.nunique())
    f["items_per_min"] = f["n_items"] / (span / 60) if span > 0 else np.nan
    f["item_revisit_share"] = 1 - f["n_items"] / len(items) if len(items) else np.nan    # доля повторных обращений к объявлению
    item_seq = g.loc[g["cls"] == "item", "item_id"].values
    if len(item_seq) >= 3:
        # доля шагов в одну сторону по номеру объявления. Обход по возрастанию идентификаторов даёт единицу
        d = np.diff(item_seq)
        f["item_id_monotone_share"] = float(max((d < 0).mean(), (d > 0).mean()))
    return f


def pointer_features(g):
    """Курсор. Считается только по веб-событиям, у мобильных платформ курсора нет."""
    f = {}
    web = g[g["is_web"]]
    f["n_web_events"] = len(web)
    if not len(web):
        return f                                        # у мобильной куки курсорных признаков нет вообще
    valid = web["pointer_valid"].values
    f["pointer_missing_share_web"] = 1 - float(valid.mean())
    f["pointer_valid_share_var"] = float(valid.var())   # смесь событий с курсором и без даёт большую дисперсию
    p = web.loc[web["pointer_valid"], ["pointer_x", "pointer_y", "session_id", "cls"]]
    if not len(p):
        return f
    x, y = p["pointer_x"].values, p["pointer_y"].values
    points = pd.Series(list(zip(x, y)))
    counts = points.value_counts()
    f["pointer_zero_share"] = float(((x == 0) & (y == 0)).mean())
    f["pointer_top1_share"] = float(counts.iloc[0] / len(points))      # доля самой частой точки
    f["pointer_unique_ratio"] = len(counts) / len(points)
    f["pointer_repeat_share"] = float(points.duplicated().mean())
    f["pointer_std_x"] = float(x.std())
    f["pointer_bbox_area_log"] = float(np.log1p((x.max() - x.min()) * (y.max() - y.min())))   # площадь охваченного прямоугольника
    f["max_x"] = float(x.max())
    # максимумы координат совпадают с одним из стандартных размеров окна с точностью 2 %
    f["viewport_default_flag"] = int(any(abs(x.max() - w) <= 0.02 * w and abs(y.max() - h) <= 0.02 * h for w, h in DEFAULT_VIEWPORTS))
    f["coord_round10_share"] = float(((x % 10 == 0) & (y % 10 == 0)).mean())
    f["coord_half_share"] = float(((x % 1 == 0.5) | (y % 1 == 0.5)).mean())
    # прыжки курсора между соседними событиями одной сессии. У клика "телепортом" они нулевые или огромные
    same_session = p["session_id"].values[1:] == p["session_id"].values[:-1]
    if same_session.any():
        jump = np.hypot(np.diff(x), np.diff(y))[same_session]
        f["jump_dist_median"] = float(np.median(jump))
        f["jump_zero_share"] = float((jump == 0).mean())
        f["jump_gt_500_share"] = float((jump > 500).mean())
    if len(x) >= 3:
        # три подряд точки на одной прямой. Площадь треугольника через векторное произведение,
        # поделённая на произведение длин сторон, близка к нулю
        x0, y0, x1, y1, x2, y2 = x[:-2], y[:-2], x[1:-1], y[1:-1], x[2:], y[2:]
        area = np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
        base = np.hypot(x1 - x0, y1 - y0) * np.hypot(x2 - x1, y2 - y1)
        if (base > 0).any():
            f["collinear_triplet_share"] = float((area[base > 0] / base[base > 0] < 0.02).mean())
    if p["cls"].nunique() >= 3 and len(p) >= 15:
        # доля дисперсии координаты x, объяснённая типом события (eta квадрат из дисперсионного анализа).
        # У человека разные действия происходят в разных зонах экрана, у скрипта нет
        grand = p["pointer_x"].mean()
        between = sum(len(s) * (s.mean() - grand) ** 2 for _, s in p.groupby("cls")["pointer_x"])    # межгрупповая сумма квадратов
        total = ((p["pointer_x"] - grand) ** 2).sum()                                                 # общая сумма квадратов
        f["pointer_by_class_eta2_x"] = float(between / total) if total > 0 else np.nan
    return f


def user_agent_features(g):
    """User-Agent и платформа на уровне куки. Большинство значений берётся у самой частой строки."""
    n = len(g)
    f = {"n_distinct_ua": int(g["ua_norm"].nunique()),
         "ua_switch_count": int((g["ua_norm"].values[1:] != g["ua_norm"].values[:-1]).sum())}   # сколько раз строка менялась подряд
    top_ua = g["ua_norm"].value_counts().index[0]
    row = g[g["ua_norm"] == top_ua].iloc[0]             # любое событие с самой частой строкой, разобранные поля у них одинаковы
    f["ua_norm_mode"] = top_ua
    f["browser_lag_mode"] = row["browser_lag"]
    f["app_version_mode"] = row["app_version"]
    f["is_firefox"] = int(row["browser_family"] == "firefox")
    # флаги "хотя бы раз за окно"
    f["ua_nonbrowser"] = int(g["ua_class"].isin(["nonbrowser", "empty"]).any())
    f["ua_headless"] = int(g["ua_headless"].any())
    f["ua_platform_mismatch"] = int(g["ua_platform_mismatch"].any())
    f["android_le_7"] = int((g["android_major"] <= 7).any())
    f["browser_lag_ge_6"] = int((g["browser_lag"] >= 6).any())
    platforms = g["platform_norm"].value_counts()
    f["is_web_mode"] = int(platforms.index[0] == "web")
    for p in ("web", "android", "ios"):
        f[f"share_platform_{p}"] = platforms.get(p, 0) / n
    return f


def cookie_features(g):
    """Все признаки одной куки. g содержит её события внутри окна в хронологическом порядке."""
    cls = g["cls"].values
    ts = g["ts"].values.astype("datetime64[s]").astype(np.int64)     # время в целых секундах
    f = rhythm_features(ts, g["hour"].values, g["sec"].values, g["session_id"].values)
    f.update(composition_features(g, cls, ts))
    f.update(diversity_features(g, cls, f["active_span_s"]))
    f.update(pointer_features(g))
    f.update(user_agent_features(g))
    return f


# ----------------------------------------------------------------------------- признаки между куками
def cross_cookie_features(ev, feats):
    """Сколько кук делят тот же User-Agent, сколько кук смотрели те же объявления, пары соседей.

    Пары соседей строятся самосоединением таблицы (кука, объявление) по объявлению. Для каждой пары
    считается мера Жаккара, число общих объявлений, делённое на число объявлений в объединении.
    Пары записываются в обе стороны, чтобы по любой куке можно было быстро найти её соседей."""
    out = pd.DataFrame(index=feats.index)
    out["ua_cookie_count"] = feats["ua_norm_mode"].map(feats["ua_norm_mode"].value_counts())     # сколько кук с той же строкой
    views = ev.loc[ev["item_id"].notna(), ["cookie_id", "item_id"]].drop_duplicates()           # кто какое объявление видел
    others_per_item = views.groupby("item_id")["cookie_id"].nunique() - 1                        # сколько других кук видели объявление
    shared = views.assign(others=views["item_id"].map(others_per_item)).groupby("cookie_id")["others"].agg(["mean", "max"])
    out["items_shared_cookies_mean"] = shared["mean"]
    out["items_shared_cookies_max"] = shared["max"]
    pairs = views.merge(views, on="item_id")            # все пары кук с общим объявлением, по строке на общее объявление
    pairs = pairs[pairs["cookie_id_x"] < pairs["cookie_id_y"]]      # каждая пара один раз, без пар куки с самой собой
    inter = pairs.groupby(["cookie_id_x", "cookie_id_y"]).size().rename("inter").reset_index()     # число общих объявлений
    n_items = views.groupby("cookie_id")["item_id"].nunique()
    # объединение равно сумме размеров минус пересечение
    inter["jac"] = inter["inter"] / (inter["cookie_id_x"].map(n_items) + inter["cookie_id_y"].map(n_items) - inter["inter"])
    both = pd.concat([inter.rename(columns={"cookie_id_x": "c", "cookie_id_y": "o"}),
                      inter.rename(columns={"cookie_id_y": "c", "cookie_id_x": "o"})])[["c", "o", "jac"]]
    out["neighbor_jaccard_max"] = both.groupby("c")["jac"].max().reindex(feats.index).fillna(0)  # у куки без соседей ноль
    return out, both


def label_rate_oof(feats, key_col):
    """Доля ботов среди кук с тем же ключом, посчитанная по другой неделе train.

    Для кук первой недели доля считается по меткам второй и наоборот, для test по всему train.
    Это стартовые значения для проверки гипотез, модели пересчитывают их в каждом сплите."""
    res = pd.Series(np.nan, index=feats.index)
    train = feats[feats["is_test"] == 0]
    for week in train["week_id"].unique():
        source = train[train["week_id"] != week]        # метки берутся только с другой недели
        stats = source.groupby(key_col)["target"].agg(["sum", "count"])
        idx = train.index[train["week_id"] == week]
        s, c = feats.loc[idx, key_col].map(stats["sum"]), feats.loc[idx, key_col].map(stats["count"])
        res.loc[idx] = np.where(c >= 3, s / c, np.nan)  # редкий ключ даёт пропуск
    stats = train.groupby(key_col)["target"].agg(["sum", "count"])
    idx = feats.index[feats["is_test"] == 1]
    s, c = feats.loc[idx, key_col].map(stats["sum"]), feats.loc[idx, key_col].map(stats["count"])
    res.loc[idx] = np.where(c >= 3, s / c, np.nan)
    return res


def neighbor_rate_oof(feats, pairs, k=20):
    """Доля ботов среди k ближайших по Жаккару соседей, по другой неделе для train и по всему train для test."""
    labeled = feats[feats["is_test"] == 0].set_index("cookie_id")
    p = pairs[pairs["o"].isin(labeled.index)].copy()   # только пары, где сосед размечен
    p["t"] = p["o"].map(labeled["target"])
    p["week_o"] = p["o"].map(labeled["week_id"])
    info = feats.set_index("cookie_id")[["week_id", "is_test"]]
    p["week_c"] = p["c"].map(info["week_id"])
    p["is_test_c"] = p["c"].map(info["is_test"])
    # для test подходит любой размеченный сосед, для train только сосед с другой недели
    p = p[(p["is_test_c"] == 1) | (p["week_o"] != p["week_c"])]
    p = p.sort_values(["c", "jac"], ascending=[True, False]).groupby("c").head(k)     # k самых похожих соседей
    return feats["cookie_id"].map(p.groupby("c")["t"].mean())


# ----------------------------------------------------------------------------- сборка
def build_features(data_dir):
    """Полная таблица признаков по кукам train и test, пары соседей и разбиение колонок по группам."""
    events = pd.read_csv(data_dir / "events.csv.gz")
    train = pd.read_csv(data_dir / "train.csv").assign(is_test=0)
    test = pd.read_csv(data_dir / "test.csv").assign(is_test=1)
    meta = pd.concat([train, test], ignore_index=True)
    ev = preprocess(events, meta)
    print(f"событий в окне: {len(ev)}, кук: {ev['cookie_id'].nunique()}")

    # признаки по каждой куке отдельно, затем строки выстраиваются в порядке meta
    feats = pd.DataFrame.from_dict({cid: cookie_features(g) for cid, g in ev.groupby("cookie_id", sort=False)}, orient="index")
    feats.index.name = "cookie_id"
    feats = feats.reindex(meta["cookie_id"])            # кука без событий в окне получает строку из пропусков

    # признаки окна и возраста куки
    m = meta.set_index("cookie_id")
    created, ws, we = pd.to_datetime(m["cookie_created_at"]), pd.to_datetime(m["window_start_ts"]), pd.to_datetime(m["window_end_ts"])
    feats["is_test"] = m["is_test"]
    feats["target"] = m["target"]
    feats["week_id"] = ((ws - ws.min()).dt.days // 7).astype(int)
    feats["window_weekday"] = ws.dt.weekday
    feats["cookie_age_h"] = ((ws - created).dt.total_seconds() / 3600).clip(lower=0)     # возраст куки к началу окна в часах
    feats["created_in_window"] = ((created >= ws) & (created < we)).astype(int)
    feats["cookie_age_lt_12h"] = (feats["cookie_age_h"] < 12).astype(int)
    first_ts = ev.groupby("cookie_id")["ts"].min().reindex(feats.index)
    feats["first_event_minus_created_s"] = (first_ts - created).dt.total_seconds()
    feats["first_event_at_creation"] = (feats["first_event_minus_created_s"].abs() <= 1).astype(int)    # событие в ту же секунду, что и кука
    feats["active_span_ratio"] = feats["active_span_s"] / (we - ws).dt.total_seconds()
    feats["age_x_events"] = np.log1p(feats["cookie_age_h"]) * np.log1p(feats["n_events"])
    # ранг версии приложения по частоте, 1 у самой распространённой версии
    feats["app_version_rank"] = feats["app_version_mode"].map(
        pd.Series(np.arange(1, feats["app_version_mode"].nunique() + 1), index=feats["app_version_mode"].value_counts().index))

    cross, pairs = cross_cookie_features(ev, feats)
    feats = feats.join(cross)
    feats = feats.reset_index()

    # признаки, зависящие от людей train. Расстояние от медианы людей и порог всплеска.
    # Здесь они считаются по всем людям train для проверки гипотез, модели пересчитывают их в каждом сплите
    humans = feats[(feats["is_test"] == 0) & (feats["target"] == 0)]
    feats["old_cookie_burst"] = ((feats["cookie_age_h"] > 720) & (feats["n_events"] > humans["n_events"].quantile(0.95))).astype(int)
    for col in ["hour_entropy", "n_events", "active_span_ratio", "n_sessions", "items_per_serp",
                "n_distinct_queries", "n_categories", "jump_dist_median"]:
        feats[f"udist_{col}"] = (feats[col] - humans[col].median()).abs()
    feats["ua_hash_bot_rate_oof"] = label_rate_oof(feats, "ua_norm_mode")
    feats["neighbor_bot_rate_oof"] = neighbor_rate_oof(feats, pairs)

    # группы колонок по префиксам имён, нужны только для таблиц в ноутбуке
    groups = {
        "ритм": [c for c in feats if c.startswith(("iei_", "n_events", "few_events", "long_pause", "max_gap", "max_events",
                                                    "active_span", "night_share", "hour_entropy", "first_event_hour",
                                                    "n_sessions", "sec_", "minute_phase", "share_iei"))],
        "кука и окно": [c for c in feats if c.startswith(("cookie_age", "created_in", "window_", "age_x", "old_cookie", "first_event_minus", "first_event_at"))],
        "состав событий": [c for c in feats if c.startswith(("n_serp", "n_item", "n_photo", "n_seller", "n_contact", "n_fav", "n_login", "n_captcha",
                                                             "serp_share", "item_share", "photo_share", "seller_share", "contact_", "fav_",
                                                             "login_", "captcha_share", "items_per_serp", "photos_per", "seller_per",
                                                             "event_type_entropy", "bigram_", "dup_share", "first_is_item", "first_item_random",
                                                             "max_search_page", "page_", "n_distinct_queries", "max_query", "query_"))],
        "разнообразие": [c for c in feats if c.startswith(("n_categories", "n_locations", "cat_loc", "private_seller", "n_items",
                                                           "items_per_min", "item_revisit", "item_id_", "items_shared", "neighbor_jaccard"))],
        "курсор": [c for c in feats if c.startswith(("n_web_events", "pointer_", "jump_", "max_x", "viewport_", "coord_", "collinear_"))],
        "user-agent и платформа": [c for c in feats if c.startswith(("ua_", "browser_", "android_", "app_version", "is_firefox", "is_web_mode",
                                                                     "share_platform", "n_distinct_ua"))],
        "с метками и udist": [c for c in feats if c.endswith("_oof") or c.startswith("udist_")],
    }
    return feats, pairs, groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="out")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    feats, pairs, groups = build_features(Path(args.data))
    feats.to_csv(out / "features.csv", index=False)
    pairs.to_csv(out / "neighbor_pairs.csv", index=False)
    (out / "feature_groups.json").write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"признаков: {feats.shape[1]} колонок для {feats.shape[0]} кук, записано в {out}")


if __name__ == "__main__":
    main()
