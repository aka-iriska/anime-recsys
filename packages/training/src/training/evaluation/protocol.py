from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from core.settings import EVALUATION, RETRIEVAL
from training.evaluation.metrics import ranked_metrics


# ranker(user_id, candidates) -> те же anime_id, упорядоченные от лучшего к худшему
Ranker = Callable[[int, np.ndarray], np.ndarray]
# главные метрики стадии retrieval: попали ли спрятанные ответы в кандидаты для ranking
RETRIEVAL_METRICS = [f"recall@{RETRIEVAL.n_candidates}", f"hit_rate@{RETRIEVAL.n_candidates}"]


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


def completed_input(table: pd.DataFrame, group: str) -> dict[int, np.ndarray]:
    """Что val/test-пользователь досмотрел в input – одно определение для всех стадий.

    Позитив – флаг is_positive (PROTOCOL.main_positive); спрятанные target сюда не попадают.

    :param table: Таблица взаимодействий из config.INTERACTIONS.
    :param group: 'val' или 'test'.
    :return: {user_id: anime_id досмотренного в порядке строк таблицы}.
    """
    rows = table[(table["split"] == f"{group}_input") & table["is_positive"]]
    completed = {}
    for user_id, user_rows in rows.groupby("user_id"):
        completed[int(user_id)] = user_rows["anime_id"].to_numpy()
    return completed


def rank_by_values(candidates: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Кандидаты по убыванию балла, при равенстве – по anime_id (одно правило для всех моделей).

    :param candidates: anime_id кандидатов.
    :param values: Балл каждого кандидата, форма (len(candidates),).
    :return: Те же anime_id от лучшего к худшему.
    """
    # lexsort сортирует по последнему ключу (минус – по убыванию), при равенстве – по anime_id
    order = np.lexsort((candidates, -values))
    return candidates[order]


def make_score_ranker(score_user: Callable[[int], np.ndarray], item_ids: np.ndarray) -> Ranker:
    """Ранжер из функции «пользователь → баллы всех аниме каталога».

    :param score_user: Функция user_id → вектор баллов формы (len(item_ids),).
    :param item_ids: Отсортированные anime_id (порядок баллов).
    :return: Функция ranker(user_id, candidates).
    """
    def ranker(user_id: int, candidates: np.ndarray) -> np.ndarray:
        # баллы всех аниме, из них – только кандидатов
        scores = score_user(user_id)
        columns = np.searchsorted(item_ids, candidates)
        return rank_by_values(candidates, scores[columns])

    return ranker


def rank_users(ranker: Ranker, cases: dict[int, tuple[set[int], set[int]]], catalog: np.ndarray) -> dict[int, np.ndarray]:
    """Список каждого пользователя: кандидаты – каталог без уже виденного (seen).

    :param ranker: Функция ranker(user_id, candidates).
    :param cases: {user_id: (seen, targets)} из build_cases.
    :param catalog: Массив anime_id допустимых кандидатов.
    :return: {user_id: anime_id по убыванию полезности} в порядке cases.
    """
    rankings = {}
    for user_id, (seen, _) in cases.items():
        candidates = catalog[~np.isin(catalog, list(seen))]
        rankings[user_id] = ranker(user_id, candidates)
    return rankings


def per_user_metrics(rankings: dict[int, np.ndarray], cases: dict, ks: Sequence[int] = EVALUATION.ks) -> pd.DataFrame:
    """Метрики каждого списка по спрятанным ответам его пользователя.

    :param rankings: {user_id: список anime_id}.
    :param cases: {user_id: (seen, targets)}.
    :param ks: Значения k.
    :return: DataFrame: индекс user_id (в порядке rankings), колонки metrics.ranked_metrics.
    """
    rows = {user_id: ranked_metrics(ranked, cases[user_id][1], ks) for user_id, ranked in rankings.items()}
    return pd.DataFrame.from_dict(rows, orient="index")


def evaluate(
    ranker: Ranker,
    cases: dict[int, tuple[set[int], set[int]]],
    catalog: np.ndarray,
    ks: Sequence[int] = EVALUATION.ks,
) -> dict[str, float]:
    """Оценивает ранжер на val- или test-пользователях.

    Для каждого пользователя кандидаты – каталог без уже виденного (seen).
    Метрики усредняются по пользователям; покрытие – доля каталога, попавшая
    хотя бы раз в топ-k.

    :param ranker: Функция ranker(user_id, candidates).
    :param cases: {user_id: (seen, targets)} из build_cases.
    :param catalog: Массив anime_id допустимых кандидатов.
    :param ks: Значения k для метрик.
    :return: Словарь recall@k, hit_rate@k, ndcg@k, coverage@k и число users.
    """
    rankings = rank_users(ranker, cases, catalog)
    per_user = per_user_metrics(rankings, cases, ks)

    # средние по пользователям: сначала все recall, потом hit_rate и ndcg
    result = {}
    for name in ("recall", "hit_rate", "ndcg"):
        for k in ks:
            column = per_user[f"{name}@{k}"]
            result[f"{name}@{k}"] = float(column.mean())
    # покрытие: какая доля каталога попала хотя бы в один топ-k
    for k in ks:
        shown = set()
        for ranked in rankings.values():
            top = ranked[:k]
            shown.update(top.tolist())
        result[f"coverage@{k}"] = len(shown) / len(catalog)
    result["users"] = len(cases)
    return result


def evaluate_per_user(
    ranker: Ranker,
    cases: dict[int, tuple[set[int], set[int]]],
    catalog: np.ndarray,
    ks: Sequence[int] = EVALUATION.ks,
) -> pd.DataFrame:
    """Метрики каждого пользователя отдельно (нужны для доверительных интервалов).

    Для каждого пользователя кандидаты – каталог без уже виденного (seen), как в evaluate.

    :param ranker: Функция ranker(user_id, candidates).
    :param cases: {user_id: (seen, targets)} из build_cases.
    :param catalog: Массив anime_id допустимых кандидатов.
    :param ks: Значения k для метрик.
    :return: DataFrame: индекс user_id, колонки recall@k, hit_rate@k и ndcg@k для каждого k.
        Покрытие не входит: оно считается по всей выдаче, а не по одному пользователю.
    """
    rankings = rank_users(ranker, cases, catalog)
    return per_user_metrics(rankings, cases, ks)
