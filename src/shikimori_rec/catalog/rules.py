from __future__ import annotations

import pandas as pd


# Protocol v0 parameters; see the status column in docs/system_design.md.
KEPT_KINDS = ("tv", "movie", "ova", "ona", "special", "tv_special")
MIN_COMPLETED_PER_ANIME = 5  # аниме с меньшим числом completed в каталог не входят


def build_catalog(anime: pd.DataFrame, rates: pd.DataFrame) -> set[int]:
    """Возвращает множество уникальных anime_id,
    которые можно использовать в качестве кандидатов для рекомендательной системы.

    :param anime: (anime_id, kind).
    :param rates: Оценки, по которым считается порог (в пайплайне — train-строки).
    :return: Множество уникальных anime_id
    """

    mask = anime["kind"].isin(KEPT_KINDS)
    good_kind = anime.loc[mask, "anime_id"]

    mask = rates["status"] == "completed"
    completed = rates.loc[mask]
    completed_counts = completed.groupby("anime_id").size()  # completed на аниме
    popular = completed_counts[completed_counts >= MIN_COMPLETED_PER_ANIME].index

    return set(good_kind.tolist()) & set(popular.tolist())


def restrict_to_catalog(rates: pd.DataFrame, catalog: set[int]) -> pd.DataFrame:
    """Возвращает оценки только на разрешенные аниме."""
    return rates[rates["anime_id"].isin(catalog)].copy()
