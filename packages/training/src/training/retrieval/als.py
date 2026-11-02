from __future__ import annotations

import numpy as np
import pandas as pd
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from core import config
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
    return np.ones_like(np.asarray(n_completed, dtype=np.float32))


def build_confidence_matrix(matrix: np.ndarray) -> csr_matrix:
    """Матрица уверенностей для обучения ALS: просмотр умножен на уверенность пользователя.

    :param matrix: Матрица (n_users, n_items) из knn.build_train_matrix, значения 0/1.
    :return: Разреженная матрица (n_users, n_items) типа float32: в клетке просмотра
        значение user_confidence(число completed пользователя), в остальных нули.
    """
    confidence = np.asarray(user_confidence(matrix.sum(axis=1)), dtype=np.float32)
    return csr_matrix(matrix * confidence[:, None])


def history_row(history: np.ndarray, n_items: int) -> csr_matrix:
    """Строка (1, n_items) с уверенностями для истории одного val/test-пользователя.

    :param history: Индексы столбцов матрицы (не anime_id) аниме, которые он досмотрел;
        может быть пустым.
    :param n_items: Число аниме в каталоге.
    :return: Разреженная матрица (1, n_items) типа float32.
    """
    value = np.float32(user_confidence(len(history)))
    return csr_matrix(
        (np.full(len(history), value, dtype=np.float32), (np.zeros(len(history), dtype=int), history)),
        shape=(1, n_items),
    )


def fit_als(confidence: csr_matrix, factors: int) -> AlternatingLeastSquares:
    """Обучает ALS на матрице уверенностей train-пользователей.

    :param confidence: Результат build_confidence_matrix, форма (n_users, n_items).
    :param factors: Размерность эмбеддинга.
    :return: Обученная модель; item_factors имеет форму (n_items, factors).
    """
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=REGULARIZATION,
        alpha=ALPHA,
        iterations=ITERATIONS,
        random_state=config.SEED,
    )
    model.fit(confidence, show_progress=False)
    return model


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
        history = histories.get(user_id, empty)
        if len(history) == 0:
            scores = np.zeros(n_items, dtype=np.float32)
        else:
            user_vector = model.recalculate_user(0, history_row(history, n_items))  # (factors,)
            scores = model.item_factors @ user_vector  # (n_items,): баллы всех аниме каталога
        values = scores[np.searchsorted(item_ids, candidates)]  # баллы только кандидатов
        order = np.lexsort((candidates, -values))  # по убыванию балла, при равенстве по id
        return candidates[order]

    return ranker


def main(group: str = "val") -> None:
    """Сравнивает на val популярность, item-item, PureSVD и ALS.

    :param group: 'val' или 'test'. Тест оставляем для финального сравнения.
    """
    table, catalog = store.load_interactions()
    cases = protocol.build_cases(table, group)

    matrix, train_users, item_ids = knn.build_train_matrix(table, catalog)
    weights = knn.user_weights(matrix, form=emb.WEIGHT_FORM)
    histories = knn.build_histories(table, group, item_ids)

    similarity = knn.item_similarity(matrix, weights)
    svd_factors = emb.fit_item_factors(matrix, weights, 64)
    rankers = {
        "popularity": popularity.make_popularity_ranker(popularity.popularity_scores(table)),
        "item_knn": knn.make_item_knn_ranker(similarity, item_ids, histories),
        "svd_k64": emb.make_svd_ranker(svd_factors, item_ids, histories),
    }

    confidence = build_confidence_matrix(matrix)
    for factors in FACTORS:
        rankers[f"als_f{factors}"] = make_als_ranker(fit_als(confidence, factors), item_ids, histories)

    results = pd.DataFrame({name: protocol.evaluate(r, cases, catalog) for name, r in rankers.items()})
    print(f"group={group}, users={len(cases)}, catalog={len(catalog)}, train users={len(train_users)}")
    print(f"ALS: regularization={REGULARIZATION}, alpha={ALPHA}, iterations={ITERATIONS}")
    print("метрики retrieval (главные):")
    print(results.loc[RETRIEVAL_METRICS].round(4).to_string())
    print("\nбонус (итоговый список, опираться на него не нужно):")
    print(results.drop(index=RETRIEVAL_METRICS).round(4).to_string())


if __name__ == "__main__":
    main()
