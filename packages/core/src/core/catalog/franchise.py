from __future__ import annotations

import logging

import pandas as pd

from core import config, logs

# подробности по шагам; включить: core.logs.enable("DEBUG")
logger = logging.getLogger(__name__)


def sequel_pairs(relations: pd.DataFrame) -> pd.DataFrame:
    """Пары «аниме → его прямое продолжение» из связей между аниме.

    Связи в данных несимметричны: часть пар записана только как «приквел» со стороны
    продолжения. Поэтому берём и строки sequel, и развёрнутые строки prequel.

    :param relations: Строки anime_relations.csv со связями аниме → аниме
        (anime_id, target_id, relation_kind).
    :return: DataFrame с колонками source, anime_id без повторов: anime_id — продолжение source.
    """
    # строка (A, sequel, B): B — продолжение A
    sequel_rows = relations[relations["relation_kind"] == "sequel"]
    forward = sequel_rows.rename(columns={"anime_id": "source", "target_id": "anime_id"})[["source", "anime_id"]]
    # строка (B, prequel, A): A — приквел B, то есть B — продолжение A
    prequel_rows = relations[relations["relation_kind"] == "prequel"]
    backward = prequel_rows.rename(columns={"target_id": "source"})[["source", "anime_id"]]
    pairs = pd.concat([forward, backward], ignore_index=True)
    return pairs.drop_duplicates(ignore_index=True)


def load_anime_links() -> pd.DataFrame:
    """Связи между аниме: продолжение, любая прямая связь, общая франшиза.

    Это данные каталога, а не пользователей: в момент рекомендации они уже известны,
    поэтому утечки здесь нет. Связь направленная: «если досмотрено source, то anime_id связано с ним».

    :return: DataFrame с колонками source, anime_id, link, где link:
        "sequel" — anime_id является продолжением source;
        "related" — у source есть любая прямая связь с anime_id (сиквел, приквел, спин-офф, ...);
        "franchise" — source и anime_id стоят в одной хронологии франшизы.
    """

    # прямые связи между аниме (связи с мангой не нужны)
    relations = pd.read_csv(config.SOURCE / "anime_relations.csv")
    relations = relations[relations["target_type"] == "Anime"]

    # продолжения: anime_id — сиквел source (то же определение, что в протоколе оценки)
    sequels = sequel_pairs(relations)
    sequels = sequels.assign(link="sequel")

    # любая прямая связь
    direct = relations.rename(columns={"anime_id": "source", "target_id": "anime_id"})
    related = direct[["source", "anime_id"]].assign(link="related")

    # франшиза: все пары аниме из одной хронологии, кроме пары аниме с самим собой
    chronology = pd.read_csv(config.SOURCE / "anime_chronology.csv")
    pairs = chronology.merge(chronology, on="seed_anime_id", suffixes=("_source", ""))
    pairs = pairs.rename(columns={"anime_id_source": "source"})[["source", "anime_id"]]
    pairs = pairs[pairs["source"] != pairs["anime_id"]]
    franchise = pairs.assign(link="franchise")

    # одна пара может встретиться в нескольких хронологиях — оставляем один раз
    links = pd.concat([sequels, related, franchise], ignore_index=True)
    links = links.drop_duplicates(ignore_index=True)
    logs.table(logger, "связи между аниме", links)
    logs.counts(logger, "связи по видам", links["link"])
    return links


def load_sequels() -> dict[int, set[int]]:
    """Прямые продолжения каждого аниме: anime_id → множество anime_id его сиквелов.

    :return: Словарь только для аниме, у которых есть хотя бы одно продолжение.
    """
    # связи между аниме и пары «аниме → продолжение» из них
    relations = pd.read_csv(config.SOURCE / "anime_relations.csv")
    relations = relations[relations["target_type"] == "Anime"]
    pairs = sequel_pairs(relations)
    return pairs.groupby("source")["anime_id"].apply(set).to_dict()


def franchise_features(pairs: pd.DataFrame, history: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Признаки пары «пользователь — кандидат» по связям кандидата с историей пользователя.

    Для каждой пары считается, сколько аниме из истории пользователя связано с кандидатом
    каждым видом связи из links. Например, prequel_completed = 2 значит, что пользователь
    досмотрел два аниме, продолжением которых является кандидат.

    :param pairs: Пары (user_id, anime_id), для которых нужны признаки: кандидаты.
    :param history: Что пользователь досмотрел и модель «знает» (user_id, anime_id): для
        датасета — input, без спрятанных ответов; в продукте — вся история.
    :param links: Результат load_anime_links (source, anime_id, link).
    :return: DataFrame с колонками user_id, anime_id, prequel_completed, n_related,
        n_franchise; строки в том же порядке, что pairs; где связей нет — 0.
    """
    # для каждого досмотренного аниме — все аниме, связанные с ним: (user_id, anime_id, link)
    watched = history.rename(columns={"anime_id": "source"})
    linked = watched.merge(links, on="source")
    logs.table(logger, "шаг 1: досмотренное (source) и связанные с ним аниме", linked)

    # оставляем только связи, которые ведут к кандидатам
    asked = pairs[["user_id", "anime_id"]].drop_duplicates()
    linked = linked.merge(asked, on=["user_id", "anime_id"])
    logs.table(logger, "шаг 2: только связи, ведущие к кандидатам", linked)

    # сколько аниме истории связано с кандидатом каждым видом связи; вид связи — в колонки
    counts = linked.groupby(["user_id", "anime_id", "link"]).size()
    logs.table(logger, "шаг 3а: число связей по (пользователь, кандидат, вид) — длинный вид", counts)
    counts = counts.unstack("link", fill_value=0)
    counts = counts.reindex(columns=list(LINK_FEATURES), fill_value=0)
    counts = counts.rename(columns=LINK_FEATURES)
    logs.table(logger, "шаг 3б: вид связи развёрнут в колонки (только пары со связями)", counts)

    # признаки для каждой пары в исходном порядке; нет связей — 0
    out = pairs[["user_id", "anime_id"]].merge(counts, left_on=["user_id", "anime_id"], right_index=True, how="left")
    out[FEATURES] = out[FEATURES].fillna(0).astype(int)
    logs.table(logger, "итог: признаки франшизы для каждой пары (нет связей — 0)", out)
    return out


# вид связи → название признака
LINK_FEATURES: dict[str, str] = {"sequel": "prequel_completed", "related": "n_related", "franchise": "n_franchise"}
FEATURES: list[str] = list(LINK_FEATURES.values())
