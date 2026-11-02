from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core import config


ALS_ARTIFACTS = config.MODELS / "als_v0"  # папка с артефактами iALS для выдачи


@dataclass(frozen=True)
class AlsParams:
    """Параметры iALS, нужные для вектора нового пользователя.

    :param alpha: Уверенность просмотра (в implicit: alpha · значение клетки).
    :param regularization: Штраф за размер вектора.
    :param factors: Длина векторов.
    """

    alpha: float
    regularization: float
    factors: int


def save_als(path: Path, item_factors: np.ndarray, item_ids: np.ndarray, anime: pd.DataFrame,
             params: AlsParams, manifest: dict) -> None:
    """Сохраняет всё, что нужно выдаче: векторы аниме, их id, справочник названий и параметры.

    :param path: Папка артефактов (создаётся, существующие файлы перезаписываются).
    :param item_factors: Векторы аниме, форма (n_items, factors).
    :param item_ids: anime_id в порядке строк item_factors.
    :param anime: Справочник аниме каталога: anime_id, name, russian, kind, aired_on.
    :param params: Параметры iALS.
    :param manifest: Сводка обучения (дата, число пользователей и т. п.).
    """
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "item_factors.npy", item_factors.astype(np.float32))
    np.save(path / "item_ids.npy", item_ids.astype(np.int64))
    anime.to_csv(path / "anime.csv", index=False)
    with (path / "params.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(params), handle, ensure_ascii=False, indent=2)
    with (path / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def load_als(path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, AlsParams]:
    """Читает артефакты iALS.

    :param path: Папка артефактов.
    :return: (item_factors, item_ids, справочник аниме, параметры).
    """
    item_factors = np.load(path / "item_factors.npy")
    item_ids = np.load(path / "item_ids.npy")
    anime = pd.read_csv(path / "anime.csv")
    with (path / "params.json").open(encoding="utf-8") as handle:
        params = AlsParams(**json.load(handle))
    return item_factors, item_ids, anime, params
