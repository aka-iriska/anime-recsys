from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from core import config, logs
from training.evaluation import protocol
from training.interactions import store
from training.retrieval import popularity
from training.retrieval import item_knn as knn
from training.retrieval import svd as emb


# Стартовые значения параметров: взяты как у библиотеки по умолчанию, подбираются на val.
FACTORS = (32, 64, 128)  # размерности эмбеддингов, которые сравниваем
REGULARIZATION = 0.01  # штраф за размер векторов; больше — сильнее сглаживание
ALPHA = 5.0  # библиотека умножает все значения матрицы на alpha (общая сила уверенности); 5 лучше 1 и 20 по recall@100 на val
ITERATIONS = 15  # число чередований «пересчитать пользователей / пересчитать аниме»
RETRIEVAL_METRICS = ["recall@100", "hit_rate@100"]  # главные метрики стадии retrieval

# подробности по шагам; включить: core.logs.enable("DEBUG")
logger = logging.getLogger(__name__)


def user_confidence(n_completed: np.ndarray | int) -> np.ndarray | float:
    """Уверенность, с которой каждый просмотр пользователя считается позитивом.

    :param n_completed: Сколько аниме досмотрел пользователь: число (для одного
        пользователя) или массив формы (n_users,). Для train-пользователя это число всех
        его completed, для val/test-пользователя число completed в его input-истории.
    :return: Положительное число той же формы, что n_completed: значение, которое
        будет стоять в каждой клетке просмотра этого пользователя.
    """
    # Бинарная уверенность: каждый просмотр весит одинаково, общую силу задаёт ALPHA.
    # Вариант 1 / sqrt(n) на val проигрывал при alpha 1–5 и догонял только при alpha=20
    n_completed = np.asarray(n_completed, dtype=np.float32)
    return np.ones_like(n_completed)


def build_confidence_matrix(matrix: np.ndarray) -> csr_matrix:
    """Матрица уверенностей для обучения ALS: просмотр умножен на уверенность пользователя.

    :param matrix: Матрица (n_users, n_items) из knn.build_train_matrix, значения 0/1.
    :return: Разреженная матрица (n_users, n_items) типа float32: в клетке просмотра
        значение user_confidence(число completed пользователя), в остальных нули.
    """
    # сколько аниме досмотрел каждый пользователь (сумма его строки)
    n_completed = matrix.sum(axis=1)
    # уверенность каждого пользователя (сейчас у всех 1)
    confidence = np.asarray(user_confidence(n_completed), dtype=np.float32)
    # каждая строка умножается на уверенность своего пользователя
    weighted = matrix * confidence[:, None]
    # храним только ненулевые клетки: матрица заполнена на ~3%
    confidence_matrix = csr_matrix(weighted)
    logs.matrix(logger, "матрица уверенностей для iALS", confidence_matrix)
    return confidence_matrix


def history_row(history: np.ndarray, n_items: int) -> csr_matrix:
    """Строка (1, n_items) с уверенностями для истории одного val/test-пользователя.

    :param history: Индексы столбцов матрицы (не anime_id) аниме, которые он досмотрел;
        может быть пустым.
    :param n_items: Число аниме в каталоге.
    :return: Разреженная матрица (1, n_items) типа float32.
    """
    # та же уверенность, что при обучении, но по числу аниме в истории этого пользователя
    value = np.float32(user_confidence(len(history)))
    # ненулевые клетки строки: значения, номер строки (всегда 0) и номера столбцов
    values = np.full(len(history), value, dtype=np.float32)
    rows = np.zeros(len(history), dtype=int)
    return csr_matrix((values, (rows, history)), shape=(1, n_items))


def fit_als(confidence: csr_matrix, factors: int) -> AlternatingLeastSquares:
    """Обучает ALS на матрице уверенностей train-пользователей.

    :param confidence: Результат build_confidence_matrix, форма (n_users, n_items).
    :param factors: Размерность эмбеддинга.
    :return: Обученная модель; item_factors имеет форму (n_items, factors).
    """
    model = AlternatingLeastSquares(
        factors=factors, # длина векторов
        regularization=REGULARIZATION, # штраф за их размер
        alpha=ALPHA, # сила уверенности просмотра
        iterations=ITERATIONS, # число чередований
        random_state=config.SEED,
    )
    # обучение: ITERATIONS раз пересчитываем векторы пользователей, затем векторы аниме
    model.fit(confidence, show_progress=False)
    logs.matrix(logger, "iALS: векторы пользователей", model.user_factors)
    logs.matrix(logger, "iALS: векторы аниме", model.item_factors)
    return model


def score_items(model: AlternatingLeastSquares, history: np.ndarray, n_items: int) -> np.ndarray:
    """Баллы всех аниме каталога для одного пользователя по его истории.

    Общий шаг для ленты (make_als_ranker) и для датасета ранжирования (баллы кандидатов).

    :param model: Результат fit_als.
    :param history: Номера столбцов матрицы (не anime_id) аниме, которые пользователь досмотрел;
        может быть пустым.
    :param n_items: Число аниме в каталоге.
    :return: Вектор формы (n_items,): чем больше, тем лучше аниме подходит пользователю.
    """
    if len(history) == 0:
        # без истории вектора нет: все баллы нулевые
        return np.zeros(n_items, dtype=np.float32)

    # история в том же виде, что строки обучающей матрицы
    row = history_row(history, n_items)
    # вектор пользователя: один шаг iALS по его истории при фиксированных векторах аниме
    user_vector = model.recalculate_user(0, row)  # (factors,)
    # балл каждого аниме: скалярное произведение его вектора с вектором пользователя
    return model.item_factors @ user_vector  # (n_items,)


def make_als_ranker(
    model: AlternatingLeastSquares, item_ids: np.ndarray, histories: dict[int, np.ndarray]
) -> protocol.Ranker:
    """Ранжер по ALS: вектор пользователя считается по его истории, балл — скалярное произведение.

    Пользователь без истории получает нулевые баллы, порядок для него задаётся только
    по anime_id.

    :param model: Результат fit_als.
    :param item_ids: Отсортированный массив anime_id (порядок строк model.item_factors).
    :param histories: Результат knn.build_histories.
    :return: Функция ranker(user_id, candidates).
    """
    empty = np.array([], dtype=int)
    n_items = len(item_ids)

    def ranker(user_id: int, candidates: np.ndarray) -> np.ndarray:
        # что пользователь досмотрел (его input): номера столбцов матрицы
        history = histories.get(user_id, empty)
        # баллы всех аниме каталога по его истории
        scores = score_items(model, history, n_items)

        # баллы только кандидатов: переводим anime_id кандидатов в номера столбцов
        columns = np.searchsorted(item_ids, candidates)
        values = scores[columns]
        # порядок по убыванию балла, при равенстве — по anime_id
        order = np.lexsort((candidates, -values))
        return candidates[order]

    return ranker


def main(group: str = "val") -> None:
    """Сравнивает на val популярность, item-item, PureSVD и ALS.

    :param group: 'val' или 'test'. Тест оставляем для финального сравнения.
    """
    # данные протокола: что каждый val-пользователь знает и что у него спрятано
    table, catalog = store.load_interactions()
    cases = protocol.build_cases(table, group)

    # матрица train-пользователей, веса пользователей и истории val-пользователей
    matrix, train_users, item_ids = knn.build_train_matrix(table, catalog)
    weights = knn.user_weights(matrix, form=emb.WEIGHT_FORM)
    histories = knn.build_histories(table, group, item_ids)

    # модели для сравнения: популярность, item-item, PureSVD
    scores = popularity.popularity_scores(table)
    similarity = knn.item_similarity(matrix, weights)
    svd_factors = emb.fit_item_factors(matrix, weights, 64)
    rankers = {
        "popularity": popularity.make_popularity_ranker(scores),
        "item_knn": knn.make_item_knn_ranker(similarity, item_ids, histories),
        "svd_k64": emb.make_svd_ranker(svd_factors, item_ids, histories),
    }

    # iALS с разной длиной векторов на одной и той же матрице уверенностей
    confidence = build_confidence_matrix(matrix)
    for factors in FACTORS:
        model = fit_als(confidence, factors)
        rankers[f"als_f{factors}"] = make_als_ranker(model, item_ids, histories)

    # одна и та же оценка для всех моделей на одних и тех же пользователях
    results = {}
    for name, ranker in rankers.items():
        results[name] = protocol.evaluate(ranker, cases, catalog)
    results = pd.DataFrame(results)

    # печать: сначала главные метрики retrieval, затем остальные
    print(f"group={group}, users={len(cases)}, catalog={len(catalog)}, train users={len(train_users)}")
    print(f"ALS: regularization={REGULARIZATION}, alpha={ALPHA}, iterations={ITERATIONS}")
    print("метрики retrieval (главные):")
    print(results.loc[RETRIEVAL_METRICS].round(4).to_string())
    print()
    print("бонус (итоговый список, опираться на него не нужно):")
    print(results.drop(index=RETRIEVAL_METRICS).round(4).to_string())


if __name__ == "__main__":
    # подробные логи по шагам; уровень — переменная окружения RECSYS_LOG_LEVEL (по умолчанию INFO, таблицы — DEBUG)
    logs.enable()
    main()
