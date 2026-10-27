from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERACTIONS = ROOT / "data" / "processed" / "interactions_v0"

KS = (10, 20, 100)  # 100: метрика стадии retrieval (попали ли ответы в кандидаты)
SEED = 42

# ranker(user_id, candidates) -> те же anime_id, упорядоченные от лучшего к худшему
Ranker = Callable[[int, np.ndarray], np.ndarray]


def load_interactions() -> tuple[pd.DataFrame, np.ndarray]:
    """Читает таблицу взаимодействий и каталог из interactions_v0.

    :return: (table, catalog): table с user_id, anime_id, status, is_positive,
        split и т. д.; catalog — массив anime_id допустимых кандидатов.
    """
    table = pd.read_csv(INTERACTIONS / "interactions.csv")
    catalog = pd.read_csv(INTERACTIONS / "catalog.csv")["anime_id"].to_numpy()
    return table, catalog


def build_cases(table: pd.DataFrame, group: str) -> dict[int, tuple[set[int], set[int]]]:
    """Возвращает {user_id: (seen, targets)} для группы 'val' или 'test'."""
    rows = table[table["split"].isin([f"{group}_input", f"{group}_target"])]
    cases = {}
    for user_id, user_rows in rows.groupby("user_id"):
        is_target = user_rows["split"] == f"{group}_target"
        # НЕ кандидаты, уже просмотренные пользователем аниме
        seen = set(user_rows.loc[~is_target, "anime_id"])
        # таргеты которые нужно найти
        targets = set(user_rows.loc[is_target, "anime_id"])
        cases[int(user_id)] = (seen, targets)
    return cases


# метрики

def recall_at_k(ranked: Sequence[int], targets: set[int], k: int) -> float:
    """Доля спрятанных позитивов, попавших в первые k позиций.

    :param ranked: Id аниме по убыванию полезности.
    :param targets: Спрятанные ответы.
    :param k: Всего мест в топе.
    :return: Число от 0 до 1.

    Знаменатель — число всех targets, а не k. Если у пользователя targets больше k,
    Recall@k не может достичь 1 даже у идеального ранжера: это нормально, метрика
    усредняется по пользователям. В targets всегда >= 1 элемент (проверено при
    подготовке данных), поэтому деления на ноль нет.
    """
    top_k = set(ranked[:k])  # первые k аниме из рекомендаций
    hits = top_k & targets  # из них те, что пользователь действительно досмотрел
    return len(hits) / len(targets)


def hit_rate_at_k(ranked: Sequence[int], targets: set[int], k: int) -> float:
    """Попал ли хотя бы один спрятанный позитив в первые k позиций.

    :param ranked: Id аниме по убыванию полезности.
    :param targets: Спрятанные ответы пользователя.
    :param k: Всего мест в топе.
    :return: 1.0, если среди первых k есть хотя бы один target, иначе 0.0.
        Усреднение по пользователям даёт долю пользователей, которым модель
        угадала хотя бы одно аниме.
    """
    top_k = set(ranked[:k])  # первые k аниме из рекомендаций
    hits = top_k & targets  # из них те, что пользователь действительно досмотрел
    return 1.0 if len(hits) > 0 else 0.0


def ndcg_at_k(ranked: Sequence[int], targets: set[int], k: int) -> float:
    """NDCG@k с бинарной релевантностью: учитывает не только попадание, но и позицию.

    Попадание на позиции i (счёт с 1) даёт вклад 1 / log2(i + 1), то есть чем выше,
    тем больше. DCG — сумма вкладов по попаданиям в первых k позициях. IDCG — DCG
    идеального списка, где все targets стоят в начале; в нём не больше
    min(k, len(targets)) слагаемых.

    :param ranked: Id аниме по убыванию полезности (порядок важен).
    :param targets: Спрятанные ответы пользователя (минимум один).
    :param k: Всего мест в топе.
    :return: DCG / IDCG, число от 0 до 1; 1.0, если все targets (или первые k из
        них) стоят в начале списка.
    """
    top_k = ranked[:k]  # порядок важен, поэтому список, а не множество
    dcg = sum(
        1 / np.log2(position + 1)
        for position, anime_id in enumerate(top_k, start=1)
        if anime_id in targets
    )
    n_ideal = min(k, len(targets))  # сколько targets вообще могут поместиться в топ-k
    idcg = sum(1 / np.log2(position + 1) for position in range(1, n_ideal + 1))
    return float(dcg / idcg)


# --- baseline'ы -------------------------------------------------------------

def popularity_scores(table: pd.DataFrame) -> pd.Series:
    """Популярность аниме по train-пользователям.

    Считается только по строкам split == 'train' со статусом completed: val/test
    сюда попасть не должны (утечка). Считаются разные пользователи, а не строки,
    чтобы кит не перетянул популярность.

    :param table: Таблица взаимодействий (user_id, anime_id, status, split, ...).
    :return: Series, индекс anime_id, значение — число разных train-пользователей
        с completed. Аниме без таких пользователей в результат не входят.
    """
    mask = (table["split"] == "train") & (table["status"] == "completed")
    train_completed = table[mask]
    return train_completed.groupby("anime_id")["user_id"].nunique()



def make_random_ranker(seed: int = SEED) -> Ranker:
    """Случайный ранжер: произвольный порядок кандидатов, воспроизводимый по seed."""
    # каждый вызов ranker даёт новую перестановку, но весь прогон воспроизводим по seed
    rng = np.random.default_rng(seed)

    def ranker(user_id: int, candidates: np.ndarray) -> np.ndarray:
        # user_id не используется: случайный порядок не зависит от пользователя.
        # permutation возвращает те же аниме, но в случайном порядке.
        return rng.permutation(candidates)

    return ranker


def make_popularity_ranker(scores: pd.Series) -> Ranker:
    """Ранжер по популярности: одинаков для всех пользователей."""
    def ranker(user_id: int, candidates: np.ndarray) -> np.ndarray:
        # Популярность каждого кандидата.
        # Аниме, которого нет в scores (никто из train его не досмотрел), получает 0 вместо пропуска.
        values = scores.reindex(candidates).fillna(0).to_numpy()

        # lexsort сортирует по ПОСЛЕДНЕМУ ключу, при равенстве — по предыдущему.
        # Поэтому главный ключ -values (минус даёт порядок по убыванию популярности),
        # запасной ключ candidates (anime_id) для воспроизводимости
        order = np.lexsort((candidates, -values))

        # order — позиции кандидатов от лучшего к худшему; берём их в этом порядке.
        return candidates[order]

    return ranker


# --- цикл оценки ------------------------------------------------------------

def evaluate(
    ranker: Ranker,
    cases: dict[int, tuple[set[int], set[int]]],
    catalog: np.ndarray,
    ks: Sequence[int] = KS,
) -> dict[str, float]:
    """Оценивает ранжер на val- или test-пользователях.

    Для каждого пользователя кандидаты — каталог без уже виденного (seen).
    Метрики усредняются по пользователям; покрытие — доля каталога, попавшая
    хотя бы раз в топ-k.

    :param ranker: Функция ranker(user_id, candidates).
    :param cases: {user_id: (seen, targets)} из build_cases.
    :param catalog: Массив anime_id допустимых кандидатов.
    :param ks: Значения k для метрик.
    :return: Словарь recall@k, hit_rate@k, ndcg@k, coverage@k и число users.
    """
    per_user = {f"{name}@{k}": [] for name in ("recall", "hit_rate", "ndcg") for k in ks}
    shown = {k: set() for k in ks}

    for user_id, (seen, targets) in cases.items():
        candidates = catalog[~np.isin(catalog, list(seen))]
        ranked = ranker(user_id, candidates)
        for k in ks:
            per_user[f"recall@{k}"].append(recall_at_k(ranked, targets, k))
            per_user[f"hit_rate@{k}"].append(hit_rate_at_k(ranked, targets, k))
            per_user[f"ndcg@{k}"].append(ndcg_at_k(ranked, targets, k))
            shown[k].update(ranked[:k].tolist())

    result = {name: float(np.mean(values)) for name, values in per_user.items()}
    for k in ks:
        result[f"coverage@{k}"] = len(shown[k]) / len(catalog)
    result["users"] = len(cases)
    return result


def evaluate_per_user(
    ranker: Ranker,
    cases: dict[int, tuple[set[int], set[int]]],
    catalog: np.ndarray,
    ks: Sequence[int] = KS,
) -> pd.DataFrame:
    """Метрики каждого пользователя отдельно (нужны для доверительных интервалов).

    Для каждого пользователя кандидаты — каталог без уже виденного (seen), как в evaluate.

    :param ranker: Функция ranker(user_id, candidates).
    :param cases: {user_id: (seen, targets)} из build_cases.
    :param catalog: Массив anime_id допустимых кандидатов.
    :param ks: Значения k для метрик.
    :return: DataFrame: индекс user_id, колонки recall@k, hit_rate@k и ndcg@k для каждого k.
        Покрытие не входит: оно считается по всей выдаче, а не по одному пользователю.
    """
    rows = {}
    for user_id, (seen, targets) in cases.items():
        candidates = catalog[~np.isin(catalog, list(seen))]
        ranked = ranker(user_id, candidates)
        row = {}
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(ranked, targets, k)
            row[f"hit_rate@{k}"] = hit_rate_at_k(ranked, targets, k)
            row[f"ndcg@{k}"] = ndcg_at_k(ranked, targets, k)
        rows[user_id] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, confidence: float = 0.95, seed: int = SEED
) -> tuple[float, float]:
    """YOU: доверительный интервал для среднего значения метрики по пользователям (bootstrap).

    :param values: Значение метрики для каждого пользователя, форма (n_users,).
    :param n_boot: Сколько раз повторять выборку.
    :param confidence: Уровень доверия (0.95 даёт границы по 2,5% и 97,5%).
    :param seed: Зерно генератора случайных чисел (результат должен воспроизводиться).
    :return: (нижняя граница, верхняя граница) интервала для среднего.

    Идея: повторяем много раз «как будто у нас другая выборка пользователей»: выбираем
    n_users значений из values С ВОЗВРАЩЕНИЕМ (один пользователь может попасть дважды,
    другой ни разу), считаем среднее. Получаем n_boot средних; их разброс и есть оценка
    шума. Границы интервала — квантили этих средних.

    Подсказки:
    - Генератор: np.random.default_rng(seed); выбор с возвращением: rng.integers или
      rng.choice(values, size=..., replace=True).
    - Все n_boot выборок можно получить одним массивом формы (n_boot, n_users) и взять
      среднее по оси 1: цикл не нужен. Память: 2000 × 173 чисел, это немного.
    - Квантили: np.quantile(means, [нижний, верхний]); при confidence=0.95 это 0.025 и 0.975.
    - Проверь на значениях, где ответ понятен: если все values одинаковы, обе границы
      равны этому значению.
    """
    raise NotImplementedError


def paired_bootstrap_diff(
    values_a: pd.Series, values_b: pd.Series, **kwargs
) -> tuple[float, float, float]:
    """Средняя разница метрики между двумя моделями и её доверительный интервал.

    Парный вариант: для каждого пользователя считаем разницу a − b и берём bootstrap
    уже по этим разницам. Так учитывается, что обе модели оценены на одних и тех же
    пользователях: общая «сложность» пользователя вычитается и не раздувает шум.
    Если интервал не содержит 0, разница не объясняется случайной выборкой пользователей.

    :param values_a: Метрика первой модели по пользователям (индекс user_id).
    :param values_b: Метрика второй модели по тем же пользователям.
    :param kwargs: Параметры для bootstrap_ci (n_boot, confidence, seed).
    :return: (средняя разница, нижняя граница, верхняя граница).
    """
    if not values_a.index.equals(values_b.index):
        raise ValueError("Models must be evaluated on the same users in the same order")
    diff = (values_a - values_b).to_numpy()
    low, high = bootstrap_ci(diff, **kwargs)
    return float(diff.mean()), low, high


def main(group: str = "val") -> None:
    """Считает метрики случайного и популярного ранжеров на выбранной группе.

    :param group: 'val' или 'test'.
    """
    table, catalog = load_interactions()
    cases = build_cases(table, group)

    rankers = {
        "random": make_random_ranker(),
        "popularity": make_popularity_ranker(popularity_scores(table)),
    }
    results = pd.DataFrame({name: evaluate(r, cases, catalog) for name, r in rankers.items()})
    print(f"group={group}, users={len(cases)}, catalog={len(catalog)}")
    print(results.round(4).to_string())


if __name__ == "__main__":
    main()
