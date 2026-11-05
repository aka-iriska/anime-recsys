from __future__ import annotations

import logging

import pandas as pd

from core import config, logs

# подробности по шагам; включить: core.logs.enable("DEBUG")
logger = logging.getLogger(__name__)


def load_anime_genres() -> pd.DataFrame:
    """Жанры, темы и демография каждого аниме (по строке на пару).

    :return: DataFrame с колонками anime_id, genre_id.
    """
    return pd.read_csv(config.SOURCE / "anime_genres.csv")


def genre_features(pairs: pd.DataFrame, history: pd.DataFrame, genres: pd.DataFrame) -> pd.DataFrame:
    """Насколько жанры кандидата типичны для истории пользователя.

    Для каждого жанра пользователя считается доля его истории с этим жанром. Признаки
    кандидата – средняя и наибольшая такая доля по жанрам кандидата.

    :param pairs: Пары (user_id, anime_id): кандидаты.
    :param history: Что пользователь досмотрел и модель «знает» (user_id, anime_id).
    :param genres: Результат load_anime_genres.
    :return: DataFrame с колонками user_id, anime_id, genre_match, genre_max; строки в том же
        порядке, что pairs; у кандидата без жанров или пользователя без истории – NaN.
    """
    # истории нет ни у кого (в выдаче: один пользователь без избранного) – вкус неизвестен, NaN
    if history.empty:
        out = pairs[["user_id", "anime_id"]].copy()
        out[GENRE_FEATURES] = float("nan")
        return out

    # жанры каждого досмотренного аниме
    watched = history.merge(genres, on="anime_id")

    # доля истории пользователя с каждым жанром
    counts = watched.groupby(["user_id", "genre_id"]).size().rename("n").reset_index()
    sizes = history.groupby("user_id").size().rename("history_size")
    counts = counts.merge(sizes, left_on="user_id", right_index=True)
    counts["share"] = counts["n"] / counts["history_size"]
    logs.table(logger, "жанры: доля истории пользователя с каждым жанром", counts)

    # жанры кандидатов и доля каждого из них в истории; жанра нет в истории – 0
    asked = pairs[["user_id", "anime_id"]].drop_duplicates()
    candidate = asked.merge(genres, on="anime_id")
    candidate = candidate.merge(counts[["user_id", "genre_id", "share"]], on=["user_id", "genre_id"], how="left")
    candidate["share"] = candidate["share"].fillna(0.0)
    logs.table(logger, "жанры: каждый жанр кандидата и его доля в истории", candidate)

    # средняя и наибольшая доля по жанрам кандидата
    match = candidate.groupby(["user_id", "anime_id"])["share"].agg(genre_match="mean", genre_max="max")

    # признаки для каждой пары в исходном порядке
    out = pairs[["user_id", "anime_id"]].merge(match, left_on=["user_id", "anime_id"], right_index=True, how="left")
    # пользователь без истории: вкус неизвестен – NaN, а не 0 («жанры не совпали»)
    no_history = ~out["user_id"].isin(history["user_id"])
    out.loc[no_history, GENRE_FEATURES] = float("nan")
    logs.table(logger, "итог: жанровые признаки для каждой пары", out)
    return out


def profile_features(pairs: pd.DataFrame, history: pd.DataFrame, item_features: pd.DataFrame) -> pd.DataFrame:
    """Признаки пользователя по его истории и их сравнение с кандидатом.

    :param pairs: Пары (user_id, anime_id): кандидаты.
    :param history: Что пользователь досмотрел и модель «знает» (user_id, anime_id).
    :param item_features: Результат anime.anime_features (нужны year и kind), индекс anime_id.
    :return: DataFrame с колонками user_id, anime_id, history_size, mean_year, year_gap,
        kind_share; строки в том же порядке, что pairs.
    """
    # год и тип каждого досмотренного аниме
    watched = history.merge(item_features[["year", "kind"]], left_on="anime_id", right_index=True)

    # сколько досмотрено и средний год выхода истории
    user = watched.groupby("user_id").agg(history_size=("anime_id", "size"), mean_year=("year", "mean"))
    logs.table(logger, "профиль: размер истории и средний год у каждого пользователя", user)

    # доля каждого типа (tv, movie, ...) в истории пользователя
    kinds = watched.groupby(["user_id", "kind"]).size().rename("n").reset_index()
    kinds = kinds.merge(user["history_size"], left_on="user_id", right_index=True)
    kinds["kind_share"] = kinds["n"] / kinds["history_size"]
    logs.table(logger, "профиль: доля каждого типа в истории", kinds)

    # год и тип кандидата
    out = pairs[["user_id", "anime_id"]].merge(item_features[["year", "kind"]], left_on="anime_id", right_index=True, how="left")

    # признаки пользователя и разница года кандидата со средним годом истории
    out = out.merge(user, left_on="user_id", right_index=True, how="left")
    out["year_gap"] = out["year"] - out["mean_year"]

    # доля типа кандидата в истории; такого типа в истории нет – 0
    out = out.merge(kinds[["user_id", "kind", "kind_share"]], on=["user_id", "kind"], how="left")
    out["kind_share"] = out["kind_share"].fillna(0.0)
    out = out[["user_id", "anime_id", *PROFILE_FEATURES]]
    logs.table(logger, "итог: признаки профиля для каждой пары", out)
    return out


GENRE_FEATURES: list[str] = ["genre_match", "genre_max"]
PROFILE_FEATURES: list[str] = ["history_size", "mean_year", "year_gap", "kind_share"]
