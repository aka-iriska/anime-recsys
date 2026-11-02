from __future__ import annotations

import numpy as np
import pandas as pd

from core import logs
from core.config import SEED
from training.evaluation import protocol
from training.evaluation.protocol import Ranker
from training.interactions import store


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


def main(group: str = "val") -> None:
    """Считает метрики случайного и популярного ранжеров на выбранной группе.

    :param group: 'val' или 'test'.
    """
    table, catalog = store.load_interactions()
    cases = protocol.build_cases(table, group)

    rankers = {
        "random": make_random_ranker(),
        "popularity": make_popularity_ranker(popularity_scores(table)),
    }
    results = pd.DataFrame({name: protocol.evaluate(r, cases, catalog) for name, r in rankers.items()})
    print(f"group={group}, users={len(cases)}, catalog={len(catalog)}")
    print(results.round(4).to_string())


if __name__ == "__main__":
    # подробные логи по шагам; уровень — переменная окружения RECSYS_LOG_LEVEL (по умолчанию DEBUG)
    logs.enable()
    main()
