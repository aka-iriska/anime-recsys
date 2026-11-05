from __future__ import annotations

import numpy as np
import pandas as pd

from core.settings import RERANK


def allowed_by_prequel(anime_ids: np.ndarray, completed: set[int], prequels: dict[int, set[int]]) -> np.ndarray:
    """Какие кандидаты можно показывать по правилу «сначала приквел».

    Продолжение показываем, только если пользователь досмотрел хотя бы один его прямой
    приквел; аниме без приквелов показывать можно всегда.

    :param anime_ids: anime_id кандидатов.
    :param completed: anime_id досмотренного пользователем.
    :param prequels: Результат franchise.load_prequels(catalog): аниме → его прямые приквелы.
    :return: Маска формы (len(anime_ids),): True – кандидата можно показывать.
    """
    allowed = []
    for anime_id in anime_ids.tolist():
        own_prequels = prequels.get(anime_id, set())
        # приквелов нет или хотя бы один из них досмотрен
        allowed.append(not own_prequels or bool(own_prequels & completed))
    return np.array(allowed, dtype=bool)


def standardize(values: np.ndarray) -> np.ndarray:
    """Приводит значения к среднему 0 и разбросу 1 внутри списка одного пользователя.

    Баллы ranking у разных пользователей в разных единицах; после стандартизации сила штрафа
    за популярность значит одно и то же для всех.

    :param values: Числа одного списка.
    :return: (values − среднее) / стандартное отклонение; нули, если разброса нет.
    """
    if len(values) == 0:
        return np.zeros(0)
    spread = values.std()
    if spread == 0:
        return np.zeros_like(values, dtype=float)
    return (values - values.mean()) / spread


def rerank(anime_ids: np.ndarray, scores: np.ndarray, popularity: np.ndarray, franchise_of: pd.Series,
           completed: set[int], prequels: dict[int, set[int]], k: int = RERANK.feed_length,
           max_per_franchise: int | None = RERANK.max_per_franchise,
           popularity_weight: float = RERANK.popularity_weight,
           require_prequel: bool = True) -> np.ndarray:
    """Итоговая лента пользователя: продуктовые правила поверх порядка ranking.

    Шаги: (1) убираем продолжения без досмотренного приквела; (2) итоговый балл –
    стандартизованный балл ranking минус popularity_weight × стандартизованная популярность;
    (3) идём по убыванию итогового балла и пропускаем аниме сверх max_per_franchise из одной
    франшизы; (4) если лента так и не набрала k, добираем пропущенными по лимиту франшизы
    (правило о приквелах не ослабляем). По умолчанию – принятая настройка core.settings.RERANK (E08).

    :param anime_ids: Кандидаты пользователя (anime_id), форма (n,).
    :param scores: Балл ranking для каждого кандидата, форма (n,).
    :param popularity: Популярность каждого кандидата (log(1 + число зрителей)), форма (n,).
    :param franchise_of: anime_id → франшиза (franchise.load_franchise_names или артефакт).
    :param completed: anime_id досмотренного пользователем.
    :param prequels: anime_id → прямые приквелы (franchise.load_prequels(catalog) или артефакт).
    :param k: Длина ленты.
    :param max_per_franchise: Не больше стольких аниме одной франшизы; None – без ограничения.
    :param popularity_weight: Сила штрафа за популярность; 0 – без штрафа.
    :param require_prequel: False – не применять правило «сначала приквел» (для сравнения).
    :return: anime_id ленты по порядку показа, не длиннее k (короче, только если кандидатов
        после правила о приквелах меньше k).
    """
    # 1. продолжения без досмотренного приквела не показываем
    if require_prequel:
        allowed = allowed_by_prequel(anime_ids, completed, prequels)
        anime_ids = anime_ids[allowed]
        scores = scores[allowed]
        popularity = popularity[allowed]

    # 2. итоговый балл: ranking минус штраф за популярность, в сопоставимых единицах
    final = standardize(scores) - popularity_weight * standardize(popularity)
    order = np.argsort(-final, kind="stable")

    # 3. по убыванию балла, не больше max_per_franchise аниме одной франшизы
    franchises = franchise_of.reindex(anime_ids).to_numpy()
    feed = []
    skipped = []
    shown_per_franchise = {}
    for i in order:
        if len(feed) == k:
            break
        name = franchises[i]
        shown = shown_per_franchise.get(name, 0)
        # эта франшиза уже набрала свой лимит – откладываем и берём следующего кандидата
        if max_per_franchise is not None and shown >= max_per_franchise:
            skipped.append(i)
            continue
        feed.append(i)
        shown_per_franchise[name] = shown + 1

    # 4. кандидатов разных франшиз не хватило – добираем отложенными, лучшими первыми
    missing = k - len(feed)
    feed.extend(skipped[:missing])
    return anime_ids[np.array(feed, dtype=int)]
