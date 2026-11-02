from __future__ import annotations

import pandas as pd


MAIN_POSITIVE = "p_completed"  # proposal; the other flags are control slices
MIN_EXPLICIT_FOR_MEAN = 5  # минимум явных оценок, чтобы считать p_above_mean
FLAG_COLUMNS = ["p_completed", "p_score8", "p_above_mean", "n_dropped"]


def mark_positives(rates: pd.DataFrame) -> pd.DataFrame:
    """Возвращает копию rates с флагами позитивов и колонкой `is_positive`.

    Флаги зависят только от самой строки (status, score), поэтому их можно
    считать до разбиения. `p_above_mean` зависит от других строк пользователя,
    его добавляет add_above_mean после разбиения.

    :param rates: Оценки (user_id, anime_id, score, status, ...).
    :return: Копия rates с колонками p_score8, p_completed, n_dropped, is_positive.
    """

    rates_c = rates.copy()

    rates_c["p_score8"] = rates_c.score >= 8
    rates_c["p_completed"] = rates_c.status.eq("completed")
    rates_c["n_dropped"] = rates_c.status.eq("dropped")
    rates_c["is_positive"] = rates_c[MAIN_POSITIVE]

    return rates_c


def add_above_mean(table: pd.DataFrame, visible: pd.Series) -> pd.DataFrame:
    """Добавляет `p_above_mean` — оценка выше средней оценки пользователя.

    Среднее считается только по строкам с visible=True (train, input), поэтому
    скрытые target-оценки не влияют на флаг. Флаг применяется ко всем строкам.
    Пользователи с менее чем MIN_EXPLICIT_FOR_MEAN явными оценками получают False.

    :param table: Таблица с user_id и score.
    :param visible: Маска строк, известных модели (split из VISIBLE_SPLITS).
    :return: Копия table с колонкой p_above_mean.
    """

    table = table.copy()
    explicit = table.score.where(table.score > 0)  # где поставили оценки
    seen = explicit.where(visible)  # только известные модели оценки
    user_mean = seen.groupby(table.user_id).transform("mean")
    user_n = seen.groupby(table.user_id).transform("count")

    table["p_above_mean"] = (explicit > user_mean) & (user_n >= MIN_EXPLICIT_FOR_MEAN)
    return table
