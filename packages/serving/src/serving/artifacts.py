from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRanker

from core import config
from core.catalog import anime as anime_catalog
from core.settings import RerankSettings


ALS_ARTIFACTS = config.MODELS / "als_v0"  # папка с артефактами iALS для выдачи
RERANK_ARTIFACTS = config.MODELS / "rerank_v0"  # данные и настройки правил ленты (serving.rerank)
RANKER_ARTIFACTS = config.MODELS / "ranker_v0"  # CatBoost и данные каталога для признаков пары


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
    :param anime: Справочник аниме каталога: anime_id, name, russian, kind, aired_on, episodes,
        genres, poster_url, poster_main_url.
    :param params: Параметры iALS.
    :param manifest: Сводка обучения (дата, число пользователей и т. п.).
    """

    path.mkdir(parents=True, exist_ok=True)
    factors32 = item_factors.astype(np.float32)
    ids64 = item_ids.astype(np.int64)
    np.save(path / "item_factors.npy", factors32)
    np.save(path / "item_ids.npy", ids64)
    anime.to_csv(path / "anime.csv", index=False)
    with (path / "params.json").open("w", encoding="utf-8") as handle:
        params_dict = asdict(params)
        json.dump(params_dict, handle, ensure_ascii=False, indent=2)
    with (path / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def save_rerank(path: Path, prequels: dict[int, set[int]], franchise_of: pd.Series, popularity: pd.Series,
                settings: RerankSettings, manifest: dict) -> None:
    """Сохраняет данные каталога и настройки, нужные правилам ленты в выдаче.

    :param path: Папка артефактов (создаётся, существующие файлы перезаписываются).
    :param prequels: anime_id → его прямые приквелы (franchise.load_prequels(catalog)).
    :param franchise_of: anime_id → франшиза для аниме каталога.
    :param popularity: anime_id → log(1 + число зрителей).
    :param settings: Принятые настройки правил.
    :param manifest: Сводка сборки (дата, версия данных и т. п.).
    """
    path.mkdir(parents=True, exist_ok=True)
    # приквелы – по строке на пару «аниме, его приквел»
    pairs = [(anime_id, prequel) for anime_id, own in prequels.items() for prequel in sorted(own)]
    pd.DataFrame(pairs, columns=["anime_id", "prequel_id"]).to_csv(path / "prequels.csv", index=False)
    franchise_of.rename_axis("anime_id").reset_index(name="franchise").to_csv(path / "franchise.csv", index=False)
    popularity.rename_axis("anime_id").reset_index(name="popularity").to_csv(path / "popularity.csv", index=False)
    with (path / "settings.json").open("w", encoding="utf-8") as handle:
        settings_dict = asdict(settings)
        json.dump(settings_dict, handle, ensure_ascii=False, indent=2)
    with (path / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class RerankRules:
    """Данные каталога и настройки правил ленты в том виде, в каком их ждёт serving.rerank.rerank.

    :param prequels: anime_id → его прямые приквелы.
    :param franchise_of: anime_id → франшиза.
    :param popularity: anime_id → log(1 + число зрителей) по всем пользователям.
    :param settings: Принятые настройки правил (длина ленты, лимит франшизы, штраф).
    """

    prequels: dict[int, set[int]]
    franchise_of: pd.Series
    popularity: pd.Series
    settings: RerankSettings


def load_rerank(path: Path) -> RerankRules:
    """Читает данные и настройки правил ленты.

    :param path: Папка артефактов.
    :return: RerankRules.
    """
    pairs = pd.read_csv(path / "prequels.csv")
    prequels = pairs.groupby("anime_id")["prequel_id"].apply(set).to_dict()
    franchise_table = pd.read_csv(path / "franchise.csv")
    franchise_of = franchise_table.set_index("anime_id")["franchise"]
    popularity_table = pd.read_csv(path / "popularity.csv")
    popularity = popularity_table.set_index("anime_id")["popularity"]
    with (path / "settings.json").open(encoding="utf-8") as handle:
        stored = json.load(handle)
    settings = RerankSettings(**stored)
    return RerankRules(prequels=prequels, franchise_of=franchise_of, popularity=popularity, settings=settings)


def load_als(path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, AlsParams]:
    """Читает артефакты iALS.

    :param path: Папка артефактов.
    :return: (item_factors, item_ids, справочник аниме, параметры).
    """

    item_factors = np.load(path / "item_factors.npy")
    item_ids = np.load(path / "item_ids.npy")
    anime = pd.read_csv(path / "anime.csv")
    with (path / "params.json").open(encoding="utf-8") as handle:
        stored = json.load(handle)
    params = AlsParams(**stored)
    return item_factors, item_ids, anime, params


@dataclass(frozen=True)
class RankerArtifacts:
    """Всё, что нужно ranking в выдаче: модель, порядок признаков и данные каталога для признаков.

    :param model: Обученный CatBoostRanker.
    :param columns: Колонки-признаки в том порядке, в котором модель их ждёт.
    :param item_features: Признаки аниме каталога (anime.anime_features), индекс anime_id.
    :param popularity: anime_id → log(1 + число train-зрителей) – признак в том же масштабе,
        что при обучении и оценке модели.
    :param links: Связи аниме каталога (franchise.load_anime_links): source, anime_id, link.
    :param genres: Жанры аниме каталога (history.load_anime_genres): anime_id, genre_id.
    """

    model: CatBoostRanker
    columns: list[str]
    item_features: pd.DataFrame
    popularity: pd.Series
    links: pd.DataFrame
    genres: pd.DataFrame


def save_ranker(path: Path, ranker: RankerArtifacts, manifest: dict) -> None:
    """Сохраняет CatBoost вместе с порядком признаков и данными каталога для признаков пары.

    :param path: Папка артефактов (создаётся, существующие файлы перезаписываются).
    :param ranker: Модель и данные (RankerArtifacts).
    :param manifest: Сводка обучения (дата, версия данных, метрики и т. п.).
    """
    path.mkdir(parents=True, exist_ok=True)
    ranker.model.save_model(str(path / "ranker.cbm"))
    # порядок признаков: модель применяется к колонкам строго в этом порядке
    categorical = [c for c in ranker.columns if c in anime_catalog.CATEGORICAL_FEATURES]
    features = {"columns": ranker.columns, "categorical": categorical}
    with (path / "features.json").open("w", encoding="utf-8") as handle:
        json.dump(features, handle, ensure_ascii=False, indent=2)
    ranker.item_features.to_csv(path / "item_features.csv")
    ranker.popularity.rename_axis("anime_id").reset_index(name="popularity").to_csv(path / "popularity.csv", index=False)
    ranker.links.to_csv(path / "links.csv", index=False)
    ranker.genres.to_csv(path / "genres.csv", index=False)
    with (path / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def load_ranker(path: Path) -> RankerArtifacts:
    """Читает CatBoost и данные для его признаков.

    :param path: Папка артефактов.
    :return: RankerArtifacts.
    """
    model = CatBoostRanker()
    model.load_model(str(path / "ranker.cbm"))
    with (path / "features.json").open(encoding="utf-8") as handle:
        features = json.load(handle)
    # категории – строки как при обучении: "123" (студия) не должна стать числом, "none" – пропуском
    text_columns = {c: str for c in features["categorical"]}
    item_features = pd.read_csv(path / "item_features.csv", index_col="anime_id", dtype=text_columns,
                                keep_default_na=False, na_values=[""])
    item_features = anime_catalog.categorical_as_text(item_features)
    popularity_table = pd.read_csv(path / "popularity.csv")
    popularity = popularity_table.set_index("anime_id")["popularity"]
    links = pd.read_csv(path / "links.csv")
    genres = pd.read_csv(path / "genres.csv")
    return RankerArtifacts(model=model, columns=features["columns"], item_features=item_features,
                           popularity=popularity, links=links, genres=genres)


def load_manifest(path: Path) -> dict:
    """Сводка сборки артефактов (manifest.json) – для ответа /info.

    :param path: Папка артефактов.
    :return: Содержимое manifest.json.
    """
    with (path / "manifest.json").open(encoding="utf-8") as handle:
        return json.load(handle)
