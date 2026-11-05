from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from core import config, logs
from core.catalog import anime, franchise, history
from core.settings import PROTOCOL, RANKING, RERANK, RETRIEVAL, as_manifest
from serving.artifacts import (ALS_ARTIFACTS, RANKER_ARTIFACTS, RERANK_ARTIFACTS, AlsParams, RankerArtifacts,
                               save_als, save_ranker, save_rerank)
from training.interactions import store
from training.ranking import model as rk
from training.retrieval import als
from training.retrieval import item_knn as knn
from training.retrieval.popularity import item_popularity

# подробности по шагам; включить: core.logs.enable("DEBUG")
logger = logging.getLogger(__name__)


def production_matrix(table: pd.DataFrame, catalog: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Матрица «пользователь × аниме» по ВСЕМ пользователям (train, val, test) для итоговой модели.

    Для офлайн-оценки так делать нельзя (модель увидит спрятанные ответы), но итоговая модель
    для выдачи должна учиться на всех данных. Поэтому её качество меряем не на ней, а по
    протоколу в evaluation (E05).

    :param table: Таблица взаимодействий из config.INTERACTIONS.
    :param catalog: Массив anime_id каталога.
    :return: (matrix, item_ids): бинарная матрица completed формы (n_users, n_items) и
        отсортированные anime_id столбцов.
    """
    completed = table[table["status"] == "completed"]
    item_ids = np.sort(catalog)
    matrix, _ = knn.build_matrix(completed, item_ids)
    return matrix, item_ids


def load_posters(item_ids: np.ndarray) -> pd.DataFrame:
    """Ссылки на постеры аниме каталога из config.POSTERS (если файл скачан).

    Файл собирает examples/parse_posters.py в репозитории shikimori-parse. Без него лента
    работает, просто без картинок.

    :param item_ids: anime_id каталога.
    :return: DataFrame с колонками anime_id, poster_url (превью для ленты), poster_main_url
        (крупнее, для карточки); пустой, если файла нет.
    """
    columns = ["anime_id", "poster_url", "poster_main_url"]
    if not config.POSTERS.exists():
        logger.warning("нет %s: лента будет без постеров (см. shikimori-parse, parse_posters.py)", config.POSTERS)
        return pd.DataFrame(columns=columns)
    posters = pd.read_csv(config.POSTERS, usecols=["anime_id", "preview_url", "main_url"])
    posters = posters.rename(columns={"preview_url": "poster_url", "main_url": "poster_main_url"})
    posters = posters[posters["anime_id"].isin(item_ids)]
    n_with = logs.num(int(posters["poster_url"].notna().sum()))
    logger.info("постеры: %s из %s аниме каталога", n_with, logs.num(len(item_ids)))
    return posters[columns]


def genre_names(item_ids: np.ndarray) -> pd.Series:
    """Жанры каждого аниме каталога одной строкой для карточки: «Сёнен, Экшен, ...».

    :param item_ids: anime_id каталога.
    :return: Series с индексом anime_id и строкой русских названий жанров.
    """
    anime_genres = history.load_anime_genres()
    anime_genres = anime_genres[anime_genres["anime_id"].isin(item_ids)]
    names = pd.read_csv(config.SOURCE / "genres.csv", usecols=["genre_id", "russian"])
    named = anime_genres.merge(names, on="genre_id")
    return named.groupby("anime_id")["russian"].agg(", ".join).rename("genres")


def anime_directory(item_ids: np.ndarray) -> pd.DataFrame:
    """Справочник аниме каталога для ленты, карточки и поиска.

    :param item_ids: anime_id каталога.
    :return: DataFrame с колонками anime_id, name, russian, kind, aired_on, episodes, genres,
        poster_url, poster_main_url; строки по возрастанию anime_id.
    """
    source = pd.read_csv(config.SOURCE / "anime.csv",
                         usecols=["anime_id", "name", "russian", "kind", "aired_on", "episodes"])
    directory = source[source["anime_id"].isin(item_ids)].sort_values("anime_id")
    # жанры одной строкой и ссылки на постеры
    genres = genre_names(item_ids)
    directory = directory.merge(genres, left_on="anime_id", right_index=True, how="left")
    posters = load_posters(item_ids)
    directory = directory.merge(posters, on="anime_id", how="left")
    logs.table(logger, "справочник аниме для выдачи", directory)
    return directory


def export_als(table: pd.DataFrame, catalog: np.ndarray, manifest: dict) -> np.ndarray:
    """Обучает итоговый iALS на всех пользователях и сохраняет models/als_v0.

    :param table: Таблица взаимодействий из config.INTERACTIONS.
    :param catalog: Массив anime_id каталога.
    :param manifest: Общая сводка сборки (дата, данные, seed).
    :return: anime_id каталога по возрастанию – порядок строк векторов.
    """
    # 1. данные: все пользователи, только аниме каталога
    matrix, item_ids = production_matrix(table, catalog)
    # 2. обучение с параметрами, выбранными в E05
    confidence = als.build_confidence_matrix(matrix)
    model = als.fit_als(confidence, RETRIEVAL.factors)
    # 3. сохранение того, что нужно выдаче (без implicit и данных пользователей)
    params = AlsParams(alpha=RETRIEVAL.alpha, regularization=RETRIEVAL.regularization, factors=RETRIEVAL.factors)
    n_users, n_items = matrix.shape
    als_manifest = {**manifest, "users": n_users, "items": n_items, "settings": as_manifest(retrieval=RETRIEVAL)}
    directory = anime_directory(item_ids)
    save_als(ALS_ARTIFACTS, model.item_factors, item_ids, directory, params, als_manifest)
    logger.info("iALS: пользователей %s, аниме %s, сохранено в %s", logs.num(n_users), logs.num(n_items), ALS_ARTIFACTS)
    return item_ids


def export_ranker(table: pd.DataFrame, item_ids: np.ndarray, manifest: dict) -> None:
    """Обучает CatBoost так же, как при оценке на val и test, и сохраняет models/ranker_v0.

    Модель учится на кандидатах train-пользователей из config.RANKING_DATA (out-of-fold iALS) –
    это ровно та модель, которую мерили в E07–E09. Вместе с ней сохраняются порядок признаков
    и данные каталога, по которым выдача считает признаки пары для нового пользователя.

    :param table: Таблица взаимодействий из config.INTERACTIONS.
    :param item_ids: anime_id каталога по возрастанию.
    :param manifest: Общая сводка сборки.
    """
    # 1. модель: все train-кандидаты, все признаки (как в final_test)
    data, columns = rk.prepare()
    train = data[data["group"] == "train"]
    train_pool = rk.make_pool(train, columns)
    model = rk.train_ranker(train_pool)

    # 2. признак популярности в том же масштабе, что при обучении: по train-пользователям
    is_train_completed = (table["split"] == "train") & (table["status"] == "completed")
    popularity = item_popularity(table[is_train_completed])
    popularity = popularity.reindex(item_ids, fill_value=0.0)

    # 3. данные каталога для признаков пары: только аниме каталога (история и кандидаты – из него)
    raw_anime = anime.load_anime()
    item_features = anime.anime_features(raw_anime)
    item_features = item_features[item_features.index.isin(item_ids)]
    links = franchise.load_anime_links()
    in_catalog = links["source"].isin(item_ids) & links["anime_id"].isin(item_ids)
    links = links[in_catalog]
    genres = history.load_anime_genres()
    genres = genres[genres["anime_id"].isin(item_ids)]

    ranker = RankerArtifacts(model=model, columns=columns, item_features=item_features,
                             popularity=popularity, links=links, genres=genres)
    ranker_manifest = {**manifest, "train_lists": int(train["user_id"].nunique()), "train_rows": len(train),
                       "ranking_data": config.RANKING_DATA.name, "settings": as_manifest(ranking=RANKING)}
    save_ranker(RANKER_ARTIFACTS, ranker, ranker_manifest)
    logger.info("CatBoost: %d признаков, связей %s, сохранено в %s", len(columns), logs.num(len(links)), RANKER_ARTIFACTS)


def export_rerank(table: pd.DataFrame, item_ids: np.ndarray, manifest: dict) -> None:
    """Сохраняет данные и настройки правил ленты: приквелы, франшизы, популярность (models/rerank_v0).

    :param table: Таблица взаимодействий из config.INTERACTIONS.
    :param item_ids: anime_id каталога по возрастанию.
    :param manifest: Общая сводка сборки.
    """
    catalog_ids = set(item_ids.tolist())
    prequels = franchise.load_prequels(catalog_ids)
    franchise_of = franchise.load_franchise_names()
    franchise_of = franchise_of[franchise_of.index.isin(item_ids)]
    # популярность для штрафа – по всем пользователям (штраф стандартизуется, масштаб не важен)
    completed = table[table["status"] == "completed"]
    popularity = item_popularity(completed)
    popularity = popularity.reindex(item_ids, fill_value=0.0)
    save_rerank(RERANK_ARTIFACTS, prequels, franchise_of, popularity, RERANK, manifest)
    logger.info("правила ленты: приквелы у %s аниме, сохранено в %s", logs.num(len(prequels)), RERANK_ARTIFACTS)


def main() -> None:
    """Собирает все артефакты выдачи: models/als_v0, models/ranker_v0 и models/rerank_v0."""
    table, catalog = store.load_interactions()
    built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # общая сводка: когда и из каких данных собрано
    manifest = {"built_at_utc": built_at, "interactions": config.INTERACTIONS.name,
                "source": config.SOURCE.name, "seed": PROTOCOL.seed}
    item_ids = export_als(table, catalog, manifest)
    export_ranker(table, item_ids, manifest)
    export_rerank(table, item_ids, manifest)
    print(f"Saved {ALS_ARTIFACTS}, {RANKER_ARTIFACTS} and {RERANK_ARTIFACTS}")


if __name__ == "__main__":
    # подробные логи по шагам; уровень – переменная окружения RECSYS_LOG_LEVEL (по умолчанию INFO, таблицы – DEBUG)
    logs.enable()
    main()
