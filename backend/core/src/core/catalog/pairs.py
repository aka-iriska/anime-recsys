from __future__ import annotations

import logging

import pandas as pd

from core import logs
from core.catalog import anime, franchise, history

# подробности по шагам; включить: core.logs.enable("DEBUG")
logger = logging.getLogger(__name__)

# признаки избранного: те же связи и жанры, но только по избранным аниме
FAVOURITE_FEATURES: list[str] = [f"fav_{c}" for c in [*franchise.FEATURES, *history.GENRE_FEATURES]]

# группы признаков ранжирования; порядок колонок модели – порядок групп и колонок внутри них
FEATURE_GROUPS: dict[str, list[str]] = {
    "retrieval": ["retrieval_score", "retrieval_rank"],
    "popularity": ["popularity"],
    "anime": ["year", "episodes", "duration", "total_minutes", *anime.CATEGORICAL_FEATURES],
    "franchise": franchise.FEATURES,
    "genres": history.GENRE_FEATURES,
    "profile": history.PROFILE_FEATURES,
    "favourites": FAVOURITE_FEATURES,
}
# группы шага 2 (без признаков пары)
BASE_GROUPS = ("retrieval", "popularity", "anime")


def group_columns(groups: list[str]) -> list[str]:
    """Колонки-признаки нескольких групп из FEATURE_GROUPS.

    :param groups: Названия групп.
    :return: Список колонок в порядке групп.
    """
    columns = []
    for name in groups:
        columns.extend(FEATURE_GROUPS[name])
    return columns


def favourite_history(history_rows: pd.DataFrame, favourites: pd.DataFrame) -> pd.DataFrame:
    """Избранные аниме пользователя, которые есть в его истории.

    В датасете избранное – итоговое состояние профиля и может содержать спрятанные ответы,
    поэтому берём только то, что пользователь и так досмотрел в известной модели истории.

    :param history_rows: История (user_id, anime_id): для датасета – input, в продукте – вся.
    :param favourites: Избранные аниме (user_id, anime_id).
    :return: DataFrame с колонками user_id, anime_id: подмножество истории.
    """
    # только избранное, которое есть в истории
    known = history_rows[["user_id", "anime_id"]]
    result = favourites[["user_id", "anime_id"]].merge(known, on=["user_id", "anime_id"]).drop_duplicates()
    logs.table(logger, "избранное из истории", result)
    return result


def pair_features(pairs: pd.DataFrame, history_rows: pd.DataFrame, favourites: pd.DataFrame,
                  item_features: pd.DataFrame, links: pd.DataFrame, genres: pd.DataFrame) -> pd.DataFrame:
    """Признаки пары «пользователь – кандидат»: франшиза, жанры, профиль, избранное.

    Один и тот же код для обучения и для выдачи: в датасете history_rows – input-история
    (без спрятанных ответов), в продукте – вся история пользователя.

    :param pairs: Пары (user_id, anime_id): кандидаты.
    :param history_rows: Что пользователь досмотрел и модель «знает» (user_id, anime_id).
    :param favourites: Избранные аниме пользователей (user_id, anime_id).
    :param item_features: Результат anime.anime_features.
    :param links: Результат franchise.load_anime_links.
    :param genres: Результат history.load_anime_genres.
    :return: DataFrame с колонками user_id, anime_id и признаками групп franchise, genres,
        profile, favourites; по строке на пару без повторов.
    """
    # пары без повторов
    pairs = pairs[["user_id", "anime_id"]].drop_duplicates()

    # связи кандидата с историей: продолжение, прямая связь, франшиза
    by_franchise = franchise.franchise_features(pairs, history_rows, links)
    # жанры кандидата в истории пользователя
    by_genres = history.genre_features(pairs, history_rows, genres)
    # пользователь: размер истории, средний год, доля типа кандидата
    by_profile = history.profile_features(pairs, history_rows, item_features)

    # те же связи и жанры, но только по избранному из истории
    favourite_rows = favourite_history(history_rows, favourites)
    fav_franchise = franchise.franchise_features(pairs, favourite_rows, links)
    fav_genres = history.genre_features(pairs, favourite_rows, genres)
    fav = fav_franchise.merge(fav_genres, on=["user_id", "anime_id"])
    fav = fav.rename(columns=dict(zip([*franchise.FEATURES, *history.GENRE_FEATURES], FAVOURITE_FEATURES)))

    # одна таблица по паре
    out = by_franchise.merge(by_genres, on=["user_id", "anime_id"])
    out = out.merge(by_profile, on=["user_id", "anime_id"])
    out = out.merge(fav, on=["user_id", "anime_id"])
    logs.table(logger, "признаки пары: франшиза, жанры, профиль, избранное", out)
    return out
