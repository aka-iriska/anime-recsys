from __future__ import annotations

import numpy as np
import pandas as pd

from core import config


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
    """YOU: признаки аниме для ранжирования.

    :param anime: Результат load_anime (anime_id, kind, rating, origin, episodes, duration,
        aired_on, studio_id).
    :return: DataFrame с индексом anime_id и колонками-признаками. Категориальные признаки
        (тип, возрастной рейтинг, первоисточник, студия) — строки; числовые — числа.
        Названия категориальных колонок перечисли в CATEGORICAL_FEATURES ниже.

    Подсказки:
    - Год выхода: первые 4 символа aired_on, превращённые в число (pd.to_numeric(...,
      errors="coerce")): даты есть у всех аниме каталога, но функция не должна падать на пустых.
    - Длительность: duration — минуты одной серии, episodes — число серий; полезен и их
      произведение (общая длительность), и сами значения. 0 в этих полях означает «нет данных»,
      а не ноль: замени на NaN (CatBoost умеет работать с пропусками в числах).
    - Категориальные: kind, rating, origin как есть; studio_id превратить в строку, пропуск —
      в "unknown". CatBoost сам закодирует категории, кодировать вручную не нужно.
    - Не добавляй сюда ничего про пользователя: это признаки только аниме.
    - Проверка: число строк равно числу аниме в anime, индекс уникальный.
    """
    raise NotImplementedError


CATEGORICAL_FEATURES: list[str] = []  # YOU: впиши названия категориальных колонок из anime_features
