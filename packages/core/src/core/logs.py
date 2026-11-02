from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd


# логгеры проекта: имена модулей начинаются с них; "__main__" — модуль, запущенный через python -m
PACKAGES = ("core", "training", "serving", "__main__")


def enable(level: int | str | None = None) -> None:
    """Включает подробные логи проекта (в ноутбуке или скрипте); логи библиотек не трогает.

    :param level: Уровень: logging.DEBUG (или "DEBUG") — таблицы и формы на каждом шаге,
        logging.INFO — только итоги. None — из переменной окружения RECSYS_LOG_LEVEL, а без
        неё INFO. Неизвестное имя уровня не роняет программу: берётся INFO.
    """
    # уровень не задан явно — берём из окружения
    if level is None:
        level = os.environ.get("RECSYS_LOG_LEVEL", "INFO")
    # имя уровня → число; опечатка в имени не должна ронять обучение
    if isinstance(level, str):
        known = logging.getLevelNamesMapping()
        level = known.get(level.upper(), logging.INFO)
    for name in PACKAGES:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # обработчик один раз: повторный вызов в ноутбуке не удваивает вывод
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            logger.addHandler(handler)
        logger.propagate = False


def num(value: int | float) -> str:
    """Целое число с разделителем разрядов «_»: 793344 -> "793_344".

    :param value: Число (дробная часть отбрасывается).
    :return: Строка для лога.
    """
    return f"{int(value):_}"


def table(logger: logging.Logger, title: str, frame: pd.DataFrame | pd.Series, rows: int = 5) -> None:
    """Пишет в лог таблицу: название, размер и первые строки.

    Если DEBUG выключен, ничего не считает.

    :param logger: Логгер модуля.
    :param title: Что это за таблица.
    :param frame: Таблица или Series.
    :param rows: Сколько первых строк показать.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    # размер: строки x колонки (у Series одна колонка)
    n_columns = frame.shape[1] if frame.ndim == 2 else 1
    text = frame.head(rows).to_string()
    logger.debug("%s: %s строк x %s колонок\n%s\n", title, num(len(frame)), num(n_columns), text)


def counts(logger: logging.Logger, title: str, values: pd.Series) -> None:
    """Пишет в лог, сколько раз встречается каждое значение (с долями).

    :param logger: Логгер модуля.
    :param title: Что считаем.
    :param values: Значения, например колонка split или link.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    n = values.value_counts()
    summary = pd.DataFrame({"строк": n, "доля": (n / n.sum()).round(3)})
    # число строк — с разделителем разрядов
    text = summary.to_string(formatters={"строк": num})
    logger.debug("%s:\n%s\n", title, text)


def matrix(logger: logging.Logger, title: str, values) -> None:
    """Пишет в лог форму матрицы или вектора, тип и долю ненулевых клеток.

    Подходит и для numpy, и для разреженных матриц scipy (у них есть nnz).

    :param logger: Логгер модуля.
    :param title: Что это за матрица.
    :param values: numpy-массив или разреженная матрица.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    shape = " x ".join(num(s) for s in values.shape)
    size = int(np.prod(values.shape))
    # разреженная матрица хранит число ненулевых сама, у плотной считаем
    if hasattr(values, "nnz"):
        nonzero = values.nnz
        kind = "разреженная"
    else:
        nonzero = int(np.count_nonzero(values))
        kind = "плотная"
    share = nonzero / size if size else 0.0
    logger.debug("%s: %s (%s, %s), ненулевых %s из %s (%.2f%%)", title, shape, kind, values.dtype, num(nonzero), num(size), 100 * share)
