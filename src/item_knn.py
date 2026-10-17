from __future__ import annotations

import numpy as np
import pandas as pd

import evaluate as ev


ACCURACY_SHARE = 0.9  # точность модели не ниже этой доли от популярности
COVERAGE_FACTOR = 10  # покрытие не меньше, чем в столько раз больше, чем у популярности


def build_train_matrix(
    table: pd.DataFrame, catalog: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Строит матрицу «пользователь × аниме» по train-пользователям (бинарное completed).

    :param table: Таблица взаимодействий из interactions_v0.
    :param catalog: Массив anime_id допустимого каталога.
    :return: (matrix, user_ids, item_ids): matrix формы (n_users, n_items) из 0.0 и 1.0
        типа float32; строки соответствуют отсортированным user_ids, столбцы —
        отсортированным item_ids. Память: n_users × n_items × 4 байта (~40 МБ).
    """
    train = table[(table["split"] == "train") & (table["status"] == "completed")]
    item_ids = np.sort(catalog)
    user_ids = np.sort(train["user_id"].unique())

    rows = np.searchsorted(user_ids, train["user_id"].to_numpy())
    cols = np.searchsorted(item_ids, train["anime_id"].to_numpy())

    matrix = np.zeros((len(user_ids), len(item_ids)), dtype=np.float32)
    matrix[rows, cols] = 1.0
    return matrix, user_ids, item_ids


def user_weights(matrix: np.ndarray, form: int = 0) -> np.ndarray:
    """Вес вклада каждого пользователя в сходство между аниме.

    :param matrix: Матрица (n_users, n_items) из build_train_matrix.
    :param form: Форма веса,
        где 0 - 1/n;
        1 - 1/sqrt(n);
        2 - 1/log(1 + n).
    :return: Вектор формы (n_users,) из положительных чисел.
    """

    completed = matrix.sum(axis=1)  # сумма строки: сколько аниме досмотрел каждый пользователь
    completed = np.maximum(completed, 1)  # пустая строка: делим на 1, а не на 0
    match form:
        case 1: return 1/np.sqrt(completed)
        case 2: return 1/np.log1p(completed)  # log(1 + n): при n = 1 обычный log дал бы деление на 0

    return 1/completed


def item_similarity(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Косинусное сходство между аниме с учётом весов пользователей.

    :param matrix: Матрица (n_users, n_items) из build_train_matrix с 1, если человек посмотрел аниме и 0, если нет.
    :param weights: Вес каждого пользователя, форма (n_users,).
    :return: Матрица (n_items, n_items) типа float32: симметричная, значения от 0
        до 1, на диагонали 0 (аниме не должно быть похоже само на себя).

    Что считаем: для пары аниме A и B — взвешенное число общих зрителей,
    делённое на корень из произведения взвешенных размеров их аудиторий
    (нормировка на популярность, чтобы хиты не были «похожи на всё»).
    """

    weighted = matrix * weights[:, None]  # каждая строка (пользователь) умножена на его вес

    # взвешенные общие зрители пар: Σ_пользователей вес × (смотрел A) × (смотрел B)
    co_viewers = matrix.T @ weighted  # (n_items, n_items)
    audience = weighted.sum(axis=0)  # взвешенный размер аудитории каждого аниме, (n_items,)

    # нормировка на размеры аудиторий обеих аниме; пустая аудитория даёт 0, а не деление на 0
    multiply_audience = np.outer(audience, audience) # [i, j] равен a[i] * b[j]
    denominator = np.sqrt(multiply_audience) # Итого: знаменатель для косинусного сходства
    similarity = np.divide(co_viewers, denominator, out=np.zeros_like(co_viewers), where=denominator > 0)

    np.fill_diagonal(similarity, 0.0)  # аниме не должно быть похоже само на себя
    return similarity.astype(np.float32)


def require(condition: bool, message: str) -> None:
    """Бросает ValueError с сообщением message, если condition ложно."""
    if not condition:
        raise ValueError(message)


def check_similarity(similarity: np.ndarray) -> None:
    """Проверяет форму и свойства матрицы сходства; при нарушении бросает ValueError.

    :param similarity: Результат item_similarity.
    """
    n = similarity.shape[0]
    require(similarity.shape == (n, n), "Similarity must be square")
    require(np.isfinite(similarity).all(), "Similarity contains NaN or inf")
    require(np.allclose(similarity, similarity.T, atol=1e-5), "Similarity must be symmetric")
    require(np.allclose(np.diag(similarity), 0.0), "Diagonal of similarity must be zero")
    require(similarity.min() >= -1e-6 and similarity.max() <= 1 + 1e-5, "Similarity must be in [0, 1]")


def build_histories(table: pd.DataFrame, group: str, item_ids: np.ndarray) -> dict[int, np.ndarray]:
    """Для каждого val/test-пользователя собирает индексы его досмотренного из input.

    :param table: Таблица взаимодействий из interactions_v0.
    :param group: 'val' или 'test'.
    :param item_ids: Отсортированный массив anime_id (порядок столбцов матрицы).
    :return: {user_id: массив индексов столбцов}; только строки <group>_input со
        статусом completed (спрятанные target сюда попасть не должны).
    """
    rows = table[(table["split"] == f"{group}_input") & (table["status"] == "completed")]
    histories = {}
    for user_id, user_rows in rows.groupby("user_id"):
        histories[int(user_id)] = np.searchsorted(item_ids, user_rows["anime_id"].to_numpy())
    return histories


def score_candidates(similarity: np.ndarray, history: np.ndarray) -> np.ndarray:
    """Балл каждого аниме каталога для пользователя по его истории.

    :param similarity: Матрица сходства (n_items, n_items).
    :param history: Индексы столбцов (не anime_id) аниме, которые досмотрел
        пользователь; может быть пустым.
    :return: Вектор формы (n_items,): чем больше, тем лучше кандидат.

    Идея: балл аниме — сумма его сходств с тем, что пользователь уже досмотрел.

    Подсказки:
    - Выбери строки матрицы для history и просуммируй их по оси 0.
    - Пустая история должна дать нулевой вектор, а не ошибку.
    - Аниме из history получат баллы тоже, но они отфильтрованы выше (их нет в
      кандидатах), поэтому здесь убирать их не нужно.
    - Подумай, что произойдёт, если у пользователя 500 досмотренных и 5: баллы
      первого будут больше просто из-за размера истории. Для порядка внутри одного
      пользователя это не важно.
    """
    # для каждого пользователя в history смотрим его историю
    taste = similarity[history].sum(axis=0) # общий вектор "вкуса" пользователя
    # по этой истории в матрице similarity находим похожие аниме
    return taste


def make_item_knn_ranker(
    similarity: np.ndarray, item_ids: np.ndarray, histories: dict[int, np.ndarray]
) -> ev.Ranker:
    """Ранжер по сходству с историей пользователя.

    Пользователь без истории получает нулевые баллы, и порядок для него задаётся
    только по anime_id (в будущем сюда можно подставить популярность).

    :param similarity: Матрица сходства (n_items, n_items).
    :param item_ids: Отсортированный массив anime_id (порядок строк и столбцов).
    :param histories: Результат build_histories.
    :return: Функция ranker(user_id, candidates).
    """
    empty = np.array([], dtype=int)

    def ranker(user_id: int, candidates: np.ndarray) -> np.ndarray:
        scores = score_candidates(similarity, histories.get(user_id, empty))
        values = scores[np.searchsorted(item_ids, candidates)]  # баллы только кандидатов
        order = np.lexsort((candidates, -values))  # по убыванию балла, при равенстве по id
        return candidates[order]

    return ranker


def check_criterion(results: pd.DataFrame, model: str, baseline: str = "popularity") -> dict[str, bool]:
    """Проверяет критерий успеха: точность не ниже ACCURACY_SHARE от baseline, покрытие в COVERAGE_FACTOR раз больше.

    :param results: Таблица метрик (колонки — модели, строки — метрики).
    :param model: Название оцениваемой модели.
    :param baseline: Название опорной модели.
    :return: {'accuracy': bool, 'coverage': bool}.
    """
    return {
        "accuracy": results.loc["ndcg@10", model] >= ACCURACY_SHARE * results.loc["ndcg@10", baseline],
        "coverage": results.loc["coverage@10", model] >= COVERAGE_FACTOR * results.loc["coverage@10", baseline],
    }


def main(group: str = "val") -> None:
    """Строит item-item модель по train, оценивает её на val и сравнивает с популярностью.

    :param group: 'val' или 'test'. Тест оставляем для финального сравнения.
    """
    table, catalog = ev.load_interactions()
    cases = ev.build_cases(table, group)

    matrix, user_ids, item_ids = build_train_matrix(table, catalog)
    similarity = item_similarity(matrix, user_weights(matrix))
    check_similarity(similarity)

    histories = build_histories(table, group, item_ids)
    rankers = {
        "popularity": ev.make_popularity_ranker(ev.popularity_scores(table)),
        "item_knn": make_item_knn_ranker(similarity, item_ids, histories),
    }
    results = pd.DataFrame({name: ev.evaluate(r, cases, catalog) for name, r in rankers.items()})

    print(f"group={group}, users={len(cases)}, catalog={len(catalog)}, train users={len(user_ids)}")
    print(results.round(4).to_string())
    print("criterion:", check_criterion(results, "item_knn"))


if __name__ == "__main__":
    main()
    # similarity = np.array([
    #     [0.0, 0.5, 0.1, 0.0],  # строка 0: сходство аниме 0 со всеми остальными
    #     [0.5, 0.0, 0.2, 0.3],  # строка 1
    #     [0.1, 0.2, 0.0, 0.4],  # строка 2
    #     [0.0, 0.3, 0.4, 0.0],  # строка 3
    # ])
    # score_candidates(similarity, np.array([], dtype=int))
