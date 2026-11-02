from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from training.evaluation.metrics import hit_rate_at_k, ndcg_at_k, recall_at_k


KS = (10, 20, 100)  # 100: метрика стадии retrieval (попали ли ответы в кандидаты)
# ranker(user_id, candidates) -> те же anime_id, упорядоченные от лучшего к худшему
Ranker = Callable[[int, np.ndarray], np.ndarray]


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
