from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from core import logs
from serving.artifacts import ALS_ARTIFACTS, AlsParams, load_als

# подробности по шагам; включить: core.logs.enable("DEBUG")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Recommendation:
    """Одна позиция ленты.

    :param anime_id: Идентификатор аниме на Shikimori.
    :param name: Название (латиницей).
    :param russian: Русское название, если есть.
    :param score: Балл модели: чем больше, тем выше в ленте.
    """

    anime_id: int
    name: str
    russian: str | None
    score: float


class Recommender:
    """Лента по истории пользователя: retrieval iALS по готовым артефактам, без обучения.

    Хранит векторы аниме и заранее посчитанную матрицу Yᵀ·Y, поэтому вектор нового
    пользователя считается одной небольшой системой уравнений. Библиотека implicit и код
    обучения для выдачи не нужны.
    """

    def __init__(self, item_factors: np.ndarray, item_ids: np.ndarray, anime: pd.DataFrame, params: AlsParams):
        """Готовит модель к выдаче.

        :param item_factors: Векторы аниме, форма (n_items, factors).
        :param item_ids: anime_id в порядке строк item_factors (по возрастанию).
        :param anime: Справочник аниме: anime_id, name, russian.
        :param params: Параметры iALS (alpha, regularization).
        """
        self.item_factors = item_factors.astype(np.float64)
        self.item_ids = item_ids
        self.params = params
        self.yty = self.item_factors.T @ self.item_factors  # одинакова для всех пользователей
        self.anime = anime.set_index("anime_id")
        self.column_of = {int(a): i for i, a in enumerate(item_ids)}
        logs.matrix(logger, "векторы аниме", self.item_factors)
        logs.matrix(logger, "YtY (общая часть системы уравнений)", self.yty)

    @classmethod
    def load(cls, path: Path = ALS_ARTIFACTS) -> Recommender:
        """Загружает артефакты, сохранённые pipelines.export_models.

        :param path: Папка артефактов.
        :return: Готовый Recommender.
        """
        return cls(*load_als(path))

    def user_vector(self, history: Iterable[int]) -> np.ndarray:
        """Вектор пользователя по истории: один шаг iALS при фиксированных векторах аниме.

        Решаем (YᵀY + (c − 1)·Y_hᵀY_h + λI)·x = c·Σ y_h, где c = alpha (уверенность
        просмотра, как в implicit), h — аниме из истории. Аниме вне каталога пропускаются.

        :param history: anime_id досмотренных пользователем аниме.
        :return: Вектор формы (factors,); нулевой, если в истории нет аниме каталога.
        """
        columns = [self.column_of[a] for a in set(int(a) for a in history) if a in self.column_of]
        k = self.item_factors.shape[1]
        if not columns:
            return np.zeros(k)
        watched = self.item_factors[columns]
        logger.debug("история: %s аниме из каталога", logs.num(len(columns)))
        c = self.params.alpha
        a = self.yty + (c - 1) * watched.T @ watched + self.params.regularization * np.eye(k)
        return np.linalg.solve(a, c * watched.sum(axis=0))

    def recommend(self, history: Iterable[int], k: int = 10, exclude: Iterable[int] = ()) -> list[Recommendation]:
        """Топ-k аниме для пользователя.

        :param history: anime_id досмотренных аниме (по ним строится вектор).
        :param k: Длина ленты.
        :param exclude: Ещё anime_id, которые нельзя показывать (например, брошенные или в планах).
        :return: Список Recommendation по убыванию балла; без аниме из history и exclude.
        """
        history = list(history)
        scores = self.item_factors @ self.user_vector(history)
        banned = [self.column_of[a] for a in set(map(int, history)) | set(map(int, exclude)) if a in self.column_of]
        scores[banned] = -np.inf  # знакомое пользователю не рекомендуем
        top = np.argsort(-scores, kind="stable")[:k]
        result = []
        for column in top:
            anime_id = int(self.item_ids[column])
            row = self.anime.loc[anime_id] if anime_id in self.anime.index else None
            result.append(Recommendation(
                anime_id=anime_id,
                name=str(row["name"]) if row is not None else str(anime_id),
                russian=None if row is None or pd.isna(row["russian"]) else str(row["russian"]),
                score=float(scores[column]),
            ))
        return result
