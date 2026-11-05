from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from core import logs
from core.catalog import anime as anime_catalog
from core.catalog import pairs
from core.settings import RETRIEVAL
from serving.artifacts import (ALS_ARTIFACTS, RANKER_ARTIFACTS, RERANK_ARTIFACTS, AlsParams, RankerArtifacts,
                               RerankRules, load_als, load_manifest, load_ranker, load_rerank)
from serving.directory import AnimeCard, AnimeDirectory
from serving.rerank import rerank

# подробности по шагам; включить: core.logs.enable("DEBUG")
logger = logging.getLogger(__name__)

USER = 0  # у запроса один пользователь; признаки пары считаются по таблицам с user_id
PERSONAL = "personal"  # лента по истории: iALS → CatBoost → правила
POPULAR = "popular"  # истории в каталоге нет: популярное с правилами ленты


@dataclass(frozen=True)
class FeedItem:
    """Одна позиция ленты.

    :param card: Сведения об аниме (название, постер, жанры).
    :param score: Балл CatBoost (в популярной ленте – популярность): чем больше, тем лучше.
    :param reason: Почему аниме в ленте, для показа: «Продолжение «…»», «Похоже на «…»».
    :param reason_anime_id: Аниме из истории, на которое ссылается причина; None у популярного.
    """

    card: AnimeCard
    score: float
    reason: str
    reason_anime_id: int | None


@dataclass(frozen=True)
class Feed:
    """Лента пользователя.

    :param items: Позиции по порядку показа.
    :param strategy: PERSONAL или POPULAR (истории в каталоге нет).
    :param history_used: Сколько аниме истории нашлось в каталоге и учтено моделью.
    """

    items: list[FeedItem]
    strategy: str
    history_used: int


class Recommender:
    """Лента по истории пользователя из готовых артефактов, без обучения.

    Три стадии, как в офлайн-оценке (E09): iALS отбирает RETRIEVAL.n_candidates кандидатов,
    CatBoost переставляет их по признакам пары (считаются только по истории из запроса),
    правила ленты убирают продолжения без приквела, повторы франшиз и сдвигают ленту от
    самых популярных аниме. Библиотеки implicit и кода обучения в выдаче нет.
    """

    def __init__(self, item_factors: np.ndarray, item_ids: np.ndarray, anime: pd.DataFrame, params: AlsParams,
                 ranker: RankerArtifacts, rules: RerankRules, manifests: dict[str, dict] | None = None):
        """Готовит модели к выдаче.

        :param item_factors: Векторы аниме iALS, форма (n_items, factors).
        :param item_ids: anime_id в порядке строк item_factors (по возрастанию).
        :param anime: Справочник аниме (models/als_v0/anime.csv).
        :param params: Параметры iALS (alpha, regularization).
        :param ranker: CatBoost и данные для его признаков (load_ranker).
        :param rules: Данные и настройки правил ленты (load_rerank).
        :param manifests: Сводки сборки артефактов по именам папок – для /info.
        """
        # retrieval: векторы аниме и общая часть системы уравнений YᵀY
        self.item_factors = item_factors.astype(np.float64)
        self.item_ids = item_ids
        self.params = params
        self.yty = self.item_factors.T @ self.item_factors
        self.column_of = {int(a): i for i, a in enumerate(item_ids)}
        # единичные векторы аниме – для причины «похоже на …» (косинусная близость)
        norms = np.linalg.norm(self.item_factors, axis=1, keepdims=True)
        self.unit_factors = self.item_factors / np.where(norms > 0, norms, 1.0)

        # ranking и правила ленты
        self.ranker = ranker
        self.rules = rules

        # связи «кандидат ← аниме истории» для причин: anime_id → [(source, link), ...]
        self.links_to = {}
        for row in ranker.links.itertuples(index=False):
            self.links_to.setdefault(int(row.anime_id), []).append((int(row.source), str(row.link)))

        self.directory = AnimeDirectory(anime, rules.popularity)
        self.manifests = manifests or {}
        logs.matrix(logger, "векторы аниме", self.item_factors)
        logger.debug("CatBoost: %d признаков; правила: %s", len(ranker.columns), rules.settings)

    @classmethod
    def load(cls, als_path: Path = ALS_ARTIFACTS, ranker_path: Path = RANKER_ARTIFACTS,
             rerank_path: Path = RERANK_ARTIFACTS) -> Recommender:
        """Загружает артефакты, сохранённые pipelines.export_models.

        :param als_path: Папка iALS и справочника аниме.
        :param ranker_path: Папка CatBoost.
        :param rerank_path: Папка правил ленты.
        :return: Готовый Recommender.
        """
        item_factors, item_ids, anime, params = load_als(als_path)
        ranker = load_ranker(ranker_path)
        rules = load_rerank(rerank_path)
        manifests = {path.name: load_manifest(path) for path in (als_path, ranker_path, rerank_path)}
        return cls(item_factors, item_ids, anime, params, ranker, rules, manifests)

    def catalog_history(self, history: Iterable[int]) -> list[int]:
        """Аниме истории, которые есть в каталоге, без повторов и в исходном порядке.

        :param history: anime_id досмотренного.
        :return: Список anime_id каталога.
        """
        result = []
        for anime_id in history:
            anime_id = int(anime_id)
            if anime_id in self.column_of and anime_id not in result:
                result.append(anime_id)
        return result

    def user_vector(self, history_ids: list[int]) -> np.ndarray:
        """Вектор пользователя по истории: один шаг iALS при фиксированных векторах аниме.

        Решаем (YᵀY + (c − 1)·Y_hᵀY_h + λI)·x = c·Σ y_h, где c = alpha (уверенность
        просмотра, как в implicit), h – аниме из истории.

        :param history_ids: anime_id досмотренного из каталога (catalog_history).
        :return: Вектор формы (factors,); нулевой, если история пуста.
        """
        k = self.item_factors.shape[1]
        if not history_ids:
            return np.zeros(k)
        columns = [self.column_of[a] for a in history_ids]
        watched = self.item_factors[columns]
        c = self.params.alpha
        identity = np.eye(k)
        a = self.yty + (c - 1) * watched.T @ watched + self.params.regularization * identity
        b = c * watched.sum(axis=0)
        return np.linalg.solve(a, b)

    def retrieve(self, history_ids: list[int], banned: set[int]) -> pd.DataFrame:
        """Стадия 1: RETRIEVAL.n_candidates лучших кандидатов iALS (как при обучении CatBoost).

        :param history_ids: anime_id досмотренного из каталога.
        :param banned: anime_id, которые нельзя рекомендовать (история и исключения).
        :return: DataFrame с колонками user_id, anime_id, retrieval_score, retrieval_rank
            (1 – лучший), по убыванию балла.
        """
        # баллы всех аниме каталога
        vector = self.user_vector(history_ids)
        scores = self.item_factors @ vector
        # знакомое пользователю не рекомендуем
        banned_columns = [self.column_of[a] for a in banned if a in self.column_of]
        scores[banned_columns] = -np.inf
        # лучшие n по убыванию балла (меньше, если каталог почти весь просмотрен)
        n_allowed = len(scores) - len(banned_columns)
        n = min(RETRIEVAL.n_candidates, n_allowed)
        top = np.argsort(-scores, kind="stable")[:n]
        candidates = pd.DataFrame({
            "user_id": USER,
            "anime_id": self.item_ids[top],
            "retrieval_score": scores[top],
            "retrieval_rank": np.arange(1, n + 1),
        })
        logs.table(logger, "кандидаты iALS", candidates)
        return candidates

    def features(self, candidates: pd.DataFrame, history_ids: list[int], favourites: Iterable[int] = ()) -> pd.DataFrame:
        """Стадия 2а: признаки CatBoost для кандидатов – тот же код, что при обучении.

        Популярность и признаки аниме – из артефактов, признаки пары (франшиза, жанры, профиль,
        избранное) – core.catalog.pairs по истории из запроса.

        :param candidates: Результат retrieve.
        :param history_ids: anime_id досмотренного из каталога.
        :param favourites: anime_id избранного (учитывается только то, что есть в истории).
        :return: Кандидаты со всеми колонками ranker.columns, тот же порядок строк.
        """
        # популярность и признаки аниме: одинаковы для всех пользователей
        popularity = self.ranker.popularity.reindex(candidates["anime_id"]).fillna(0.0)
        data = candidates.assign(popularity=popularity.to_numpy())
        data = data.merge(self.ranker.item_features, left_on="anime_id", right_index=True, how="left")
        # признаки пары по истории и избранному из запроса
        history_rows = pd.DataFrame({"user_id": USER, "anime_id": history_ids})
        favourite_ids = [int(a) for a in favourites]
        favourite_rows = pd.DataFrame({"user_id": USER, "anime_id": favourite_ids}, dtype=int)
        by_pair = pairs.pair_features(candidates, history_rows, favourite_rows, self.ranker.item_features,
                                      self.ranker.links, self.ranker.genres)
        data = data.merge(by_pair, on=["user_id", "anime_id"], how="left")
        logs.table(logger, "признаки CatBoost", data)
        return data

    def rank(self, data: pd.DataFrame) -> np.ndarray:
        """Стадия 2б: балл CatBoost для каждого кандидата.

        :param data: Результат features.
        :return: Баллы формы (len(data),): чем больше, тем выше.
        """
        columns = data[self.ranker.columns]
        features = anime_catalog.categorical_as_text(columns)
        return self.ranker.model.predict(features)

    def explain(self, anime_id: int, history_ids: list[int]) -> tuple[str, int | None]:
        """Причина показа аниме: связь с историей или самое похожее досмотренное.

        Порядок: продолжение досмотренного → прямая связь → та же франшиза → самое близкое
        по векторам iALS аниме истории.

        :param anime_id: Аниме ленты.
        :param history_ids: anime_id досмотренного из каталога.
        :return: (текст причины, anime_id из истории, на которое она ссылается).
        """
        history_set = set(history_ids)
        templates = {"sequel": "Продолжение «{}»", "related": "Связано с «{}»", "franchise": "Из франшизы «{}»"}
        own_links = self.links_to.get(int(anime_id), [])
        # связи с историей по силе: продолжение, прямая связь, франшиза
        for link, template in templates.items():
            for source, kind in own_links:
                if kind == link and source in history_set:
                    return template.format(self.title(source)), source
        # иначе – самое похожее аниме истории по косинусу векторов iALS
        columns = [self.column_of[a] for a in history_ids]
        candidate = self.unit_factors[self.column_of[int(anime_id)]]
        similarity = self.unit_factors[columns] @ candidate
        closest = history_ids[int(np.argmax(similarity))]
        return f"Похоже на «{self.title(closest)}»", closest

    def title(self, anime_id: int) -> str:
        """Название аниме для текста причины.

        :param anime_id: Аниме каталога.
        :return: Русское название, иначе латиницей; если аниме нет в справочнике – id.
        """
        card = self.directory.card(anime_id)
        return card.title if card is not None else str(anime_id)

    def popular_feed(self, banned: set[int], k: int, completed: Iterable[int] = ()) -> Feed:
        """Лента по популярности с правилами ленты: без продолжений к недосмотренному, одна франшиза – раз.

        В выдаче – лента нового пользователя (истории в каталоге нет); в симуляции – вариант
        «популярность» A/B-теста.

        :param banned: anime_id, которые нельзя показывать.
        :param k: Длина ленты.
        :param completed: anime_id досмотренного: продолжения к ним разрешены.
        :return: Feed со strategy = POPULAR.
        """
        allowed = ~np.isin(self.item_ids, list(banned))
        anime_ids = self.item_ids[allowed]
        popularity = self.rules.popularity.reindex(anime_ids).fillna(0.0).to_numpy()
        completed_ids = {int(a) for a in completed}
        # балл – сама популярность, поэтому штраф за популярность здесь не нужен
        feed_ids = rerank(anime_ids, popularity, popularity, self.rules.franchise_of, completed_ids,
                          self.rules.prequels, k=k, max_per_franchise=self.rules.settings.max_per_franchise,
                          popularity_weight=0.0)
        score_of = dict(zip(anime_ids.tolist(), popularity.tolist()))
        items = []
        for anime_id in feed_ids:
            card = self.directory.card(anime_id)
            item = FeedItem(card=card, score=score_of[int(anime_id)], reason="Популярно у зрителей", reason_anime_id=None)
            items.append(item)
        return Feed(items=items, strategy=POPULAR, history_used=0)

    def feed(self, history: Iterable[int], exclude: Iterable[int] = (), favourites: Iterable[int] = (),
             k: int | None = None) -> Feed:
        """Лента пользователя: iALS → CatBoost → правила ленты.

        :param history: anime_id досмотренного (по нему строятся вектор и признаки пары).
        :param exclude: Ещё anime_id, которые нельзя показывать (брошенные, в планах, смотрю).
        :param favourites: anime_id избранного.
        :param k: Длина ленты; None – принятая длина из настроек правил (RERANK.feed_length).
        :return: Feed: позиции по порядку показа с причинами.
        """
        k = k or self.rules.settings.feed_length
        history = [int(a) for a in history]
        history_ids = self.catalog_history(history)
        banned = set(history) | {int(a) for a in exclude}
        logger.debug("история: %d аниме, из каталога %d, нельзя показывать %d", len(history), len(history_ids), len(banned))
        if not history_ids:
            return self.popular_feed(banned, k)

        # 1. retrieval: кандидаты iALS
        candidates = self.retrieve(history_ids, banned)
        # 2. ranking: признаки и балл CatBoost
        data = self.features(candidates, history_ids, favourites)
        scores = self.rank(data)
        # 3. правила ленты поверх порядка CatBoost
        anime_ids = data["anime_id"].to_numpy()
        popularity = self.rules.popularity.reindex(anime_ids).fillna(0.0).to_numpy()
        feed_ids = rerank(anime_ids, scores, popularity, self.rules.franchise_of, set(history_ids), self.rules.prequels, k=k,
                          max_per_franchise=self.rules.settings.max_per_franchise,
                          popularity_weight=self.rules.settings.popularity_weight)

        # позиции ленты с причинами
        score_of = dict(zip(anime_ids.tolist(), scores.tolist()))
        items = []
        for anime_id in feed_ids:
            card = self.directory.card(anime_id)
            reason, source = self.explain(anime_id, history_ids)
            item = FeedItem(card=card, score=score_of[int(anime_id)], reason=reason, reason_anime_id=source)
            items.append(item)
        return Feed(items=items, strategy=PERSONAL, history_used=len(history_ids))
