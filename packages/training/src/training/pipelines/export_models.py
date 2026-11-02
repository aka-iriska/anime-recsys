from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from core import config
from training.interactions import store
from training.retrieval import als
from serving.artifacts import ALS_ARTIFACTS, AlsParams, save_als


FACTORS = 128  # лучшая конфигурация iALS по recall@100 (E05)


def production_matrix(table: pd.DataFrame, catalog: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Матрица «пользователь × аниме» по ВСЕМ пользователям (train, val, test) для итоговой модели.

    Для офлайн-оценки так делать нельзя (модель увидит спрятанные ответы), но итоговая модель
    для выдачи должна учиться на всех данных. Поэтому её качество меряем не на ней, а по
    протоколу в evaluation (E05).

    :param table: Таблица взаимодействий из interactions_v0.
    :param catalog: Массив anime_id каталога.
    :return: (matrix, item_ids): бинарная матрица completed формы (n_users, n_items) и
        отсортированные anime_id столбцов.
    """
    completed = table[table["status"] == "completed"]
    item_ids = np.sort(catalog)
    user_ids = np.sort(completed["user_id"].unique())
    matrix = np.zeros((len(user_ids), len(item_ids)), dtype=np.float32)
    matrix[np.searchsorted(user_ids, completed["user_id"].to_numpy()),
           np.searchsorted(item_ids, completed["anime_id"].to_numpy())] = 1.0
    return matrix, item_ids


def anime_directory(item_ids: np.ndarray) -> pd.DataFrame:
    """Справочник аниме каталога для показа в ленте.

    :param item_ids: anime_id каталога.
    :return: DataFrame с колонками anime_id, name, russian, kind, aired_on.
    """
    anime = pd.read_csv(config.SOURCE / "anime.csv", usecols=["anime_id", "name", "russian", "kind", "aired_on"])
    return anime[anime["anime_id"].isin(item_ids)].sort_values("anime_id")


def main() -> None:
    """Обучает iALS на всех пользователях и сохраняет артефакты выдачи в models/als_v0."""
    # 1. данные: все пользователи, только аниме каталога
    table, catalog = store.load_interactions()
    matrix, item_ids = production_matrix(table, catalog)
    # 2. обучение итоговой модели с параметрами, выбранными в E05
    model = als.fit_als(als.build_confidence_matrix(matrix), FACTORS)
    # 3. сохранение того, что нужно выдаче (без implicit и данных пользователей)
    params = AlsParams(alpha=als.ALPHA, regularization=als.REGULARIZATION, factors=FACTORS)
    manifest = {"built_at_utc": datetime.now(timezone.utc).isoformat(), "users": int(matrix.shape[0]),
                "items": int(matrix.shape[1]), "iterations": als.ITERATIONS, "seed": config.SEED}
    save_als(ALS_ARTIFACTS, model.item_factors, item_ids, anime_directory(item_ids), params, manifest)
    print(f"users={matrix.shape[0]}, items={matrix.shape[1]}, factors={FACTORS}")
    print(f"Saved {ALS_ARTIFACTS}")


if __name__ == "__main__":
    main()
