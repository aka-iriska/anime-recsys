from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from core import config, logs

# подробности по шагам; включить: core.logs.enable("DEBUG")
logger = logging.getLogger(__name__)


def load_anime() -> pd.DataFrame:
    """Сырые поля аниме, из которых строятся признаки (по одной строке на anime_id).

    Сознательно не загружаем score, status, episodes_aired: это текущее состояние сайта,
    на момент рекомендации его не было (утечка из будущего).

    :return: DataFrame с колонками anime_id, kind, rating, origin, episodes, duration,
        aired_on, studio_id (первая студия аниме; у части аниме студии нет – NaN).
    """
    anime = pd.read_csv(
        config.SOURCE / "anime.csv",
        usecols=["anime_id", "kind", "rating", "origin", "episodes", "duration", "aired_on"],
    )
    # первая студия каждого аниме
    anime_studios = pd.read_csv(config.SOURCE / "anime_studios.csv")
    first_studio = anime_studios.groupby("anime_id")["studio_id"].first()
    first_studio = first_studio.rename("studio_id")
    return anime.merge(first_studio, left_on="anime_id", right_index=True, how="left")


def anime_features(anime: pd.DataFrame) -> pd.DataFrame:
    """Признаки аниме для ранжирования: одинаковы для всех пользователей.

    Числовые: год выхода, число серий, длительность серии и общая длительность в минутах
    (0 в исходных полях означает «нет данных», поэтому заменяется на NaN). Категориальные
    (строки, пропуск – "unknown"): тип, возрастной рейтинг, первоисточник, первая студия.

    :param anime: Результат load_anime (anime_id, kind, rating, origin, episodes, duration,
        aired_on, studio_id).
    :return: DataFrame с индексом anime_id: year, episodes, duration, total_minutes, kind,
        rating, origin, studio.
    """
    index = pd.Index(anime["anime_id"].to_numpy(), name="anime_id")
    features = pd.DataFrame(index=index)
    # год выхода – первые 4 символа даты начала показа
    year_text = anime["aired_on"].str[:4]
    year = pd.to_numeric(year_text, errors="coerce")
    features["year"] = year.to_numpy()
    # 0 серий или 0 минут – это пропуск данных, а не настоящий ноль
    episodes = anime["episodes"].replace(0, np.nan).to_numpy(dtype=float)
    duration = anime["duration"].replace(0, np.nan).to_numpy(dtype=float)
    features["episodes"] = episodes
    features["duration"] = duration
    # сколько минут займёт просмотр целиком
    features["total_minutes"] = episodes * duration
    # категориальные признаки CatBoost закодирует сам; ему нужны строки без пропусков
    for column in ("kind", "rating", "origin"):
        features[column] = anime[column].fillna("unknown").astype(str).to_numpy()
    studio = anime["studio_id"].map(studio_name)
    features["studio"] = studio.to_numpy()
    logs.table(logger, "признаки аниме (одинаковы для всех пользователей)", features)
    return features


def studio_name(studio_id: float) -> str:
    """Студия как категория для CatBoost: id строкой, пропуск – "unknown".

    :param studio_id: id студии (NaN, если студии нет).
    :return: Строка.
    """
    if pd.isna(studio_id):
        return "unknown"
    return str(int(studio_id))


CATEGORICAL_FEATURES: list[str] = ["kind", "rating", "origin", "studio"]


def categorical_as_text(features: pd.DataFrame) -> pd.DataFrame:
    """Категориальные колонки – обычные строки, пропуск – "unknown".

    CatBoost не принимает NaN в категориальных признаках. Одна функция для обучения и выдачи:
    так модель в сервисе получает категории в том же виде, что при обучении.

    :param features: Таблица признаков.
    :return: Копия таблицы, где колонки из CATEGORICAL_FEATURES приведены к строкам.
    """
    features = features.copy()
    categorical = features.columns.intersection(CATEGORICAL_FEATURES)
    for column in categorical:
        # pandas хранит пропуски в строковых колонках особым значением: превращаем их в "unknown"
        values = features[column].astype("object")
        is_present = values.notna()
        values = values.where(is_present, "unknown")
        features[column] = values.astype(str).to_numpy(dtype=object)
    return features
