from __future__ import annotations

import numpy as np
import pandas as pd

from core import config


def load_interactions() -> tuple[pd.DataFrame, np.ndarray]:
    """Читает таблицу взаимодействий и каталог из config.INTERACTIONS.

    :return: (table, catalog): table с user_id, anime_id, status, is_positive,
        split и т. д.; catalog — массив anime_id допустимых кандидатов.
    """
    table = pd.read_csv(config.INTERACTIONS / "interactions.csv")
    catalog = pd.read_csv(config.INTERACTIONS / "catalog.csv")["anime_id"].to_numpy()
    return table, catalog
