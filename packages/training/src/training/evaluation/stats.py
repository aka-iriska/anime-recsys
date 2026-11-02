from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import SEED


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, confidence: float = 0.95, seed: int = SEED
) -> tuple[float, float]:
    """Доверительный интервал для среднего значения метрики по пользователям (bootstrap).

    :param values: Значение метрики для каждого пользователя, форма (n_users,).
    :param n_boot: Сколько раз повторять выборку.
    :param confidence: Уровень доверия (0.95 даёт границы по 2,5% и 97,5%).
    :param seed: Зерно генератора случайных чисел (результат должен воспроизводиться).
    :return: (нижняя граница, верхняя граница) интервала для среднего.

    Идея: повторяем много раз «как будто у нас другая выборка пользователей»: выбираем
    n_users значений из values С ВОЗВРАЩЕНИЕМ (один пользователь может попасть дважды,
    другой ни разу), считаем среднее. Получаем n_boot средних; их разброс и есть оценка
    шума. Границы интервала — квантили этих средних.
    """
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [(1 - confidence) / 2, 1 - (1 - confidence) / 2])
    return float(lo), float(hi)


def paired_bootstrap_diff(
    values_a: pd.Series, values_b: pd.Series, **kwargs
) -> tuple[float, float, float]:
    """Средняя разница метрики между двумя моделями и её доверительный интервал.

    Парный вариант: для каждого пользователя считаем разницу a − b и берём bootstrap
    уже по этим разницам. Так учитывается, что обе модели оценены на одних и тех же
    пользователях: общая «сложность» пользователя вычитается и не раздувает шум.
    Если интервал не содержит 0, разница не объясняется случайной выборкой пользователей.

    :param values_a: Метрика первой модели по пользователям (индекс user_id).
    :param values_b: Метрика второй модели по тем же пользователям.
    :param kwargs: Параметры для bootstrap_ci (n_boot, confidence, seed).
    :return: (средняя разница, нижняя граница, верхняя граница).
    """
    if not values_a.index.equals(values_b.index):
        raise ValueError("Models must be evaluated on the same users in the same order")
    diff = (values_a - values_b).to_numpy()
    low, high = bootstrap_ci(diff, **kwargs)
    return float(diff.mean()), low, high
