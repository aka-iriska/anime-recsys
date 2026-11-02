from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from training.evaluation import protocol
from training.interactions import store
from training.retrieval import popularity
from training.retrieval import item_knn as knn
from core import logs
from core.validation import require

# подробности по шагам; включить: core.logs.enable()
logger = logging.getLogger(__name__)


K_VALUES = (16, 32, 64, 128)  # размерности эмбеддингов, которые сравниваем на val
WEIGHT_FORM = 1  # форма веса пользователя из item_knn.user_weights (1 = 1/sqrt(n), лучшая по recall@100 в E02)
RETRIEVAL_METRICS = ["recall@100", "hit_rate@100"]  # главные метрики стадии retrieval
K_NORMALISED = 64  # размерность для экспериментов с нормировкой столбцов
BETAS = (0.25, 0.5)  # сила нормировки на популярность аниме (E04b): 0 — нет, 0,5 — как косинус


def column_scale(matrix: np.ndarray, weights: np.ndarray, beta: float) -> np.ndarray:
    """Множитель для столбцов (аниме): 1 / (взвешенная аудитория аниме) ** beta.

    Нормировка на популярность, как в косинусном сходстве item-item: хиты с большой
    аудиторией ослабляются. beta = 0 даёт множитель 1 (нормировки нет), beta = 0,5 — как
    в косинусе, промежуточные значения ослабляют хиты мягче.

    :param matrix: Матрица (n_users, n_items) из knn.build_train_matrix.
    :param weights: Вес вклада каждого пользователя, форма (n_users,).
    :param beta: Сила нормировки, не меньше 0.
    :return: Вектор формы (n_items,) типа float32.
    """
    audience = (matrix * weights[:, None]).sum(axis=0)  # такая же аудитория, как в item_similarity
    return (np.maximum(audience, 1e-12) ** (-beta)).astype(np.float32)


def fit_item_factors(
    matrix: np.ndarray, weights: np.ndarray, k: int, scale: np.ndarray | None = None
) -> np.ndarray:
    """Векторы аниме — первые k правых сингулярных векторов взвешенной матрицы.

    :param matrix: Матрица (n_users, n_items) из knn.build_train_matrix, значения 0/1.
    :param weights: Вес вклада каждого пользователя, форма (n_users,), из knn.user_weights.
    :param k: Размерность эмбеддинга, строго меньше min(n_users, n_items).
    :param scale: Множитель для столбцов из column_scale (нормировка на популярность);
        None — без нормировки.
    :return: Матрица факторов формы (n_items, k) типа float32: строка — вектор одного
        аниме; столбцы ортонормированы (Vᵀ·V ≈ единичная матрица).
    """
    n_users, n_items = matrix.shape
    if k >= min(n_users, n_items):
        raise ValueError(f"k={k} must be less than min(n_users, n_items)={min(n_users, n_items)}")

    # умножаем строки на корень из веса: в произведении Xᵀ·X он встречается дважды, и
    # вклад пользователя в сходство аниме получается равным его весу, как в item-item
    weighted = matrix * np.sqrt(weights)[:, None]
    if scale is not None:
        weighted = weighted * scale[None, :]  # каждый столбец (аниме) умножаем на его множитель

    # X  ≈  U · Σ · Vᵀ
    # U (n_users, k) как каждый пользователь «вписывается» в каждый фактор
    # Σ (k,) насколько каждый фактор важен
    # Vᵀ (k, n_items) как каждое аниме «вписывается» в каждый скрытый фактор
    u, s, vt = svds(csr_matrix(weighted), k=k, rng=0)

    # svds возвращает факторы по возрастанию важности: ставим самые важные первыми
    vt = vt[np.argsort(-s)]
    logs.matrix(logger, "SVD: U, векторы пользователей", u)
    logs.matrix(logger, "SVD: Vt, векторы аниме (строки — факторы)", vt)

    # делаем копию матрицы с ПОСТРОЧНЫМ расположением; транспонируем, чтобы строка = вектор одного аниме
    return np.ascontiguousarray(vt.T, dtype=np.float32)


def fold_in(item_factors: np.ndarray, history: np.ndarray) -> np.ndarray:
    """Вектор пользователя из его истории (проекция, без обучения).

    :param item_factors: Матрица (n_items, k) из fit_item_factors.
    :param history: Индексы строк item_factors (не anime_id) аниме, которые досмотрел
        пользователь; может быть пустым.
    :return: Вектор формы (k,) типа float32. Для пустой истории — нулевой вектор.

    Идея: вектор пользователя — сумма векторов аниме из его истории. Это то же самое,
    что умножить его бинарный вектор «что смотрел» (длины n_items) на матрицу факторов.
    """
    result = np.zeros(shape=(item_factors.shape[1]))

    user_animes = item_factors[history] # (m_items, k)
    result = user_animes.sum(axis=0)

    return result


def check_item_factors(item_factors: np.ndarray, k: int, n_items: int) -> None:
    """Проверяет форму и свойства факторов; при нарушении бросает ValueError.

    :param item_factors: Результат fit_item_factors.
    :param k: Ожидаемая размерность.
    :param n_items: Число аниме в каталоге.
    """
    require(item_factors.shape == (n_items, k), f"Factors must have shape ({n_items}, {k})")
    require(np.isfinite(item_factors).all(), "Factors contain NaN or inf")
    gram = item_factors.T @ item_factors  # для ортонормированных столбцов это единичная матрица
    require(np.allclose(gram, np.eye(k), atol=1e-3), "Factor columns must be orthonormal")


def make_svd_ranker(
    item_factors: np.ndarray,
    item_ids: np.ndarray,
    histories: dict[int, np.ndarray],
    scale: np.ndarray | None = None,
    undo_scale: bool = False,
) -> protocol.Ranker:
    """Ранжер по эмбеддингам: балл кандидата — скалярное произведение с вектором пользователя.

    Пользователь без истории получает нулевые баллы, порядок для него задаётся только
    по anime_id.

    Если факторы построены с нормировкой столбцов (scale), историю пользователя нужно
    масштабировать так же, как столбцы при обучении. По умолчанию кандидаты ранжируются
    по баллам в нормированном пространстве (хиты ослаблены); undo_scale=True возвращает
    баллы в исходные единицы (делит на scale), и тогда ослабление хитов частично снимается.

    :param item_factors: Матрица (n_items, k) из fit_item_factors.
    :param item_ids: Отсортированный массив anime_id (порядок строк item_factors).
    :param histories: Результат knn.build_histories.
    :param scale: Множитель столбцов из column_scale, тот же, что при обучении; None — без нормировки.
    :param undo_scale: Делить ли баллы на scale (имеет смысл только вместе с scale).
    :return: Функция ranker(user_id, candidates).
    """
    empty = np.array([], dtype=int)
    # вектор пользователя = сумма векторов просмотренных аниме, каждое умножено на scale
    history_factors = item_factors if scale is None else item_factors * scale[:, None]

    def ranker(user_id: int, candidates: np.ndarray) -> np.ndarray:
        user_vector = fold_in(history_factors, histories.get(user_id, empty))  # (k,)
        scores = item_factors @ user_vector  # (n_items,): баллы всех аниме каталога
        if scale is not None and undo_scale:
            scores = scores / scale
        values = scores[np.searchsorted(item_ids, candidates)]  # баллы только кандидатов
        order = np.lexsort((candidates, -values))  # по убыванию балла, при равенстве по id
        return candidates[order]

    return ranker


def main(group: str = "val") -> None:
    """Сравнивает на val популярность, item-item и PureSVD при разных k.

    :param group: 'val' или 'test'. Тест оставляем для финального сравнения.
    """
    table, catalog = store.load_interactions()
    cases = protocol.build_cases(table, group)

    matrix, train_users, item_ids = knn.build_train_matrix(table, catalog)
    weights = knn.user_weights(matrix, form=WEIGHT_FORM)
    histories = knn.build_histories(table, group, item_ids)

    similarity = knn.item_similarity(matrix, weights)
    rankers = {
        "popularity": popularity.make_popularity_ranker(popularity.popularity_scores(table)),
        "item_knn": knn.make_item_knn_ranker(similarity, item_ids, histories),
    }
    for k in K_VALUES:
        factors = fit_item_factors(matrix, weights, k)
        check_item_factors(factors, k, len(item_ids))
        rankers[f"svd_k{k}"] = make_svd_ranker(factors, item_ids, histories)

    # E04b: нормировка столбцов на популярность аниме при k = K_NORMALISED
    for beta in BETAS:
        scale = column_scale(matrix, weights, beta)
        factors = fit_item_factors(matrix, weights, K_NORMALISED, scale=scale)
        check_item_factors(factors, K_NORMALISED, len(item_ids))
        rankers[f"svd_k{K_NORMALISED}_b{beta}"] = make_svd_ranker(factors, item_ids, histories, scale=scale)
        rankers[f"svd_k{K_NORMALISED}_b{beta}_undo"] = make_svd_ranker(
            factors, item_ids, histories, scale=scale, undo_scale=True
        )

    results = pd.DataFrame({name: protocol.evaluate(r, cases, catalog) for name, r in rankers.items()})
    print(f"group={group}, users={len(cases)}, catalog={len(catalog)}, train users={len(train_users)}, weight form={WEIGHT_FORM}")
    print("метрики retrieval (главные):")
    print(results.loc[RETRIEVAL_METRICS].round(4).to_string())
    print("\nбонус (итоговый список, опираться на него не нужно):")
    print(results.drop(index=RETRIEVAL_METRICS).round(4).to_string())


if __name__ == "__main__":
    # подробные логи по шагам; уровень — переменная окружения RECSYS_LOG_LEVEL (по умолчанию DEBUG)
    logs.enable()
    main()
