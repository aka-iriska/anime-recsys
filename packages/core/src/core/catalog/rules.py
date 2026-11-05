from __future__ import annotations

import logging

import pandas as pd

from core.logs import num
from core.settings import PROTOCOL

logger = logging.getLogger(__name__)


def build_catalog(anime: pd.DataFrame, rates: pd.DataFrame) -> set[int]:
    """Возвращает множество уникальных anime_id,
    которые можно использовать в качестве кандидатов для рекомендательной системы.

    :param anime: (anime_id, kind).
    :param rates: Оценки, по которым считается порог (в пайплайне – train-строки).
    :return: Множество уникальных anime_id
    """

    mask = anime["kind"].isin(PROTOCOL.kept_kinds)
    good_kind = anime.loc[mask, "anime_id"]

    mask = rates["status"] == "completed"
    completed = rates.loc[mask]
    completed_counts = completed.groupby("anime_id").size()  # completed на аниме
    popular = completed_counts[completed_counts >= PROTOCOL.min_completed_per_anime].index

    # в каталоге – аниме, прошедшие оба условия
    good_kind_ids = set(good_kind.tolist())
    popular_ids = set(popular.tolist())
    catalog = good_kind_ids & popular_ids
    logger.info("каталог: %s аниме (подходящий тип: %s, не меньше %d completed: %s)",
                num(len(catalog)), num(len(good_kind)), PROTOCOL.min_completed_per_anime, num(len(popular)))
    return catalog


def restrict_to_catalog(rates: pd.DataFrame, catalog: set[int]) -> pd.DataFrame:
    """Возвращает оценки только на разрешенные аниме."""
    return rates[rates["anime_id"].isin(catalog)].copy()
