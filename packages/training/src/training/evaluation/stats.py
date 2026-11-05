from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config import SEED


@dataclass(frozen=True)
class Estimate:
    """Среднее по пользователям и его 95% интервал (bootstrap).

    :param mean: Среднее.
    :param low: Нижняя граница интервала.
    :param high: Верхняя граница интервала.
    """

    mean: float
    low: float
    high: float

    @property
    def significant(self) -> bool:
        """Интервал не содержит 0 – для разницы это значит «не объясняется шумом»."""
        return self.low > 0 or self.high < 0


@dataclass(frozen=True)
class Comparison:
    """Модель рядом с опорной: её оценка и парная разница с опорной на тех же пользователях.

    :param value: Оценка метрики модели.
    :param diff: Оценка разницы «модель − опорная».
    """

    value: Estimate
    diff: Estimate


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
    шума. Границы интервала – квантили этих средних.
    """
    rng = np.random.default_rng(seed)
    # n_boot выборок пользователей с возвращением и среднее каждой
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = samples.mean(axis=1)
    # границы – квантили средних
    tail = (1 - confidence) / 2
    lo, hi = np.quantile(means, [tail, 1 - tail])
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


def summarize(values: pd.Series) -> Estimate:
    """Среднее метрики по пользователям и его 95% интервал.

    :param values: Метрика по пользователям.
    :return: Estimate.
    """
    array = np.asarray(values)
    low, high = bootstrap_ci(array)
    return Estimate(mean=float(array.mean()), low=low, high=high)


def compare(values: pd.Series, reference: pd.Series) -> Comparison:
    """Оценка модели и её парная разница с опорной – одна форма для всех таблиц сравнения.

    :param values: Метрика модели по пользователям (индекс user_id).
    :param reference: Метрика опорной модели по тем же пользователям.
    :return: Comparison.
    """
    value = summarize(values)
    mean, low, high = paired_bootstrap_diff(values, reference)
    diff = Estimate(mean=mean, low=low, high=high)
    return Comparison(value=value, diff=diff)


def independent_bootstrap_diff(values_a: pd.Series, values_b: pd.Series, n_boot: int = 2000,
                               confidence: float = 0.95, seed: int = SEED) -> Estimate:
    """Разница средних двух НЕЗАВИСИМЫХ групп пользователей и её интервал – как в A/B-тесте.

    В A/B каждый пользователь видит только один вариант, поэтому парных разниц нет: группы
    перевыбираются с возвращением отдельно, и разброс разницы средних шире, чем в парном
    сравнении на тех же пользователях (paired_bootstrap_diff).

    :param values_a: Метрика пользователей группы A.
    :param values_b: Метрика пользователей группы B (другие пользователи).
    :param n_boot: Сколько раз повторять выборку.
    :param confidence: Уровень доверия.
    :param seed: Зерно генератора.
    :return: Estimate разницы «a − b».
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    # каждая группа перевыбирается отдельно, разница – по средним
    means_a = rng.choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
    means_b = rng.choice(b, size=(n_boot, len(b)), replace=True).mean(axis=1)
    diffs = means_a - means_b
    tail = (1 - confidence) / 2
    low, high = np.quantile(diffs, [tail, 1 - tail])
    return Estimate(mean=float(a.mean() - b.mean()), low=float(low), high=float(high))
