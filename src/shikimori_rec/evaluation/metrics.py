from __future__ import annotations

from typing import Sequence

import numpy as np


def recall_at_k(ranked: Sequence[int], targets: set[int], k: int) -> float:
    """Доля спрятанных позитивов, попавших в первые k позиций.

    :param ranked: Id аниме по убыванию полезности.
    :param targets: Спрятанные ответы.
    :param k: Всего мест в топе.
    :return: Число от 0 до 1.

    Знаменатель — число всех targets, а не k. Если у пользователя targets больше k,
    Recall@k не может достичь 1 даже у идеального ранжера: это нормально, метрика
    усредняется по пользователям. В targets всегда >= 1 элемент (проверено при
    подготовке данных), поэтому деления на ноль нет.
    """
    top_k = set(ranked[:k])  # первые k аниме из рекомендаций
    hits = top_k & targets  # из них те, что пользователь действительно досмотрел
    return len(hits) / len(targets)


def hit_rate_at_k(ranked: Sequence[int], targets: set[int], k: int) -> float:
    """Попал ли хотя бы один спрятанный позитив в первые k позиций.

    :param ranked: Id аниме по убыванию полезности.
    :param targets: Спрятанные ответы пользователя.
    :param k: Всего мест в топе.
    :return: 1.0, если среди первых k есть хотя бы один target, иначе 0.0.
        Усреднение по пользователям даёт долю пользователей, которым модель
        угадала хотя бы одно аниме.
    """
    top_k = set(ranked[:k])  # первые k аниме из рекомендаций
    hits = top_k & targets  # из них те, что пользователь действительно досмотрел
    return 1.0 if len(hits) > 0 else 0.0


def ndcg_at_k(ranked: Sequence[int], targets: set[int], k: int) -> float:
    """NDCG@k с бинарной релевантностью: учитывает не только попадание, но и позицию.

    Попадание на позиции i (счёт с 1) даёт вклад 1 / log2(i + 1), то есть чем выше,
    тем больше. DCG — сумма вкладов по попаданиям в первых k позициях. IDCG — DCG
    идеального списка, где все targets стоят в начале; в нём не больше
    min(k, len(targets)) слагаемых.

    :param ranked: Id аниме по убыванию полезности (порядок важен).
    :param targets: Спрятанные ответы пользователя (минимум один).
    :param k: Всего мест в топе.
    :return: DCG / IDCG, число от 0 до 1; 1.0, если все targets (или первые k из
        них) стоят в начале списка.
    """
    top_k = ranked[:k]  # порядок важен, поэтому список, а не множество
    dcg = sum(
        1 / np.log2(position + 1)
        for position, anime_id in enumerate(top_k, start=1)
        if anime_id in targets
    )
    n_ideal = min(k, len(targets))  # сколько targets вообще могут поместиться в топ-k
    idcg = sum(1 / np.log2(position + 1) for position in range(1, n_ideal + 1))
    return float(dcg / idcg)
