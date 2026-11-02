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
        aired_on, studio_id (первая студия аниме; у части аниме студии нет — NaN).
    """
    anime = pd.read_csv(
        config.SOURCE / "anime.csv",
        usecols=["anime_id", "kind", "rating", "origin", "episodes", "duration", "aired_on"],
    )
    studios = pd.read_csv(config.SOURCE / "anime_studios.csv").groupby("anime_id")["studio_id"].first()
    return anime.merge(studios.rename("studio_id"), left_on="anime_id", right_index=True, how="left")


def anime_features(anime: pd.DataFrame) -> pd.DataFrame:
    """Признаки аниме для ранжирования: одинаковы для всех пользователей.

    Числовые: год выхода, число серий, длительность серии и общая длительность в минутах
    (0 в исходных полях означает «нет данных», поэтому заменяется на NaN). Категориальные
    (строки, пропуск — "unknown"): тип, возрастной рейтинг, первоисточник, первая студия.

    :param anime: Результат load_anime (anime_id, kind, rating, origin, episodes, duration,
        aired_on, studio_id).
    :return: DataFrame с индексом anime_id: year, episodes, duration, total_minutes, kind,
        rating, origin, studio.
    """
    features = pd.DataFrame(index=pd.Index(anime["anime_id"].to_numpy(), name="anime_id"))
    # год выхода — первые 4 символа даты начала показа
    features["year"] = pd.to_numeric(anime["aired_on"].str[:4], errors="coerce").to_numpy()
    # 0 серий или 0 минут — это пропуск данных, а не настоящий ноль
    episodes = anime["episodes"].replace(0, np.nan).to_numpy(dtype=float)
    duration = anime["duration"].replace(0, np.nan).to_numpy(dtype=float)
    features["episodes"] = episodes
    features["duration"] = duration
    # сколько минут займёт просмотр целиком
    features["total_minutes"] = episodes * duration
    # категориальные признаки CatBoost закодирует сам; ему нужны строки без пропусков
    for column in ("kind", "rating", "origin"):
        features[column] = anime[column].fillna("unknown").astype(str).to_numpy()
    features["studio"] = anime["studio_id"].map(lambda s: "unknown" if pd.isna(s) else str(int(s))).to_numpy()
    logs.table(logger, "признаки аниме (одинаковы для всех пользователей)", features)
    return features


CATEGORICAL_FEATURES: list[str] = ["kind", "rating", "origin", "studio"]
