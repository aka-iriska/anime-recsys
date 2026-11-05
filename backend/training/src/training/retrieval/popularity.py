from __future__ import annotations

import numpy as np
import pandas as pd

from core import logs
from core.config import SEED
from training.evaluation import protocol
from training.evaluation.protocol import Ranker
from training.interactions import store


def item_popularity(completed: pd.DataFrame) -> pd.Series:
    """Популярность аниме: log(1 + число разных пользователей, досмотревших его).

    Одно определение для baseline, признака ranking и штрафа re-ranking в выдаче; разные
    только строки, по которым считаем. Считаются разные пользователи, а не строки, чтобы
    кит не перетянул популярность.

    :param completed: Строки досмотренного (user_id, anime_id).
    :return: Series с индексом anime_id; аниме без зрителей в результат не входят.
    """
    viewers = completed.groupby("anime_id")["user_id"].nunique()
    return np.log1p(viewers).rename("popularity")


def popularity_scores(table: pd.DataFrame) -> pd.Series:
    """Популярность аниме по train-пользователям.

    Считается только по строкам split == 'train' со статусом completed: val/test
    сюда попасть не должны (утечка).

    :param table: Таблица взаимодействий (user_id, anime_id, status, split, ...).
    :return: Результат item_popularity по train-строкам; аниме без train-зрителей не входят.
    """
    mask = (table["split"] == "train") & (table["status"] == "completed")
    train_completed = table[mask]
    return item_popularity(train_completed)


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
        # по убыванию популярности, при равенстве – по anime_id
        return protocol.rank_by_values(candidates, values)

    return ranker


def main(group: str = "val") -> None:
    """Считает метрики случайного и популярного ранжеров на выбранной группе.

    :param group: 'val' или 'test'.
    """
    table, catalog = store.load_interactions()
    cases = protocol.build_cases(table, group)

    scores = popularity_scores(table)
    rankers = {
        "random": make_random_ranker(),
        "popularity": make_popularity_ranker(scores),
    }
    # одна и та же оценка для всех моделей на одних и тех же пользователях
    metrics_by_model = {}
    for name, ranker in rankers.items():
        metrics_by_model[name] = protocol.evaluate(ranker, cases, catalog)
    results = pd.DataFrame(metrics_by_model)
    print(f"group={group}, users={len(cases)}, catalog={len(catalog)}")
    rounded = results.round(4)
    print(rounded.to_string())


if __name__ == "__main__":
    # подробные логи по шагам; уровень – переменная окружения RECSYS_LOG_LEVEL (по умолчанию INFO, таблицы – DEBUG)
    logs.enable()
    main()
