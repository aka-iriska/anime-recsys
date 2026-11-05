from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from core import logs
from core.settings import EVALUATION
from training.evaluation import protocol
from training.interactions import store
from training.retrieval import popularity
from core.validation import require



# подробности по шагам; включить: core.logs.enable("DEBUG")
logger = logging.getLogger(__name__)


def build_matrix(rows: pd.DataFrame, item_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Бинарная матрица «пользователь × аниме» по строкам взаимодействий.

    Общий шаг для офлайн-оценки (train-пользователи) и итоговой модели (все пользователи):
    различается только то, какие строки передаются.

    :param rows: Строки (user_id, anime_id) – обычно досмотренное.
    :param item_ids: Отсортированные anime_id каталога (порядок столбцов).
    :return: (matrix, user_ids): matrix формы (n_users, n_items) из 0.0 и 1.0 типа float32,
        строки – отсортированные user_ids.
    """
    user_column = rows["user_id"].to_numpy()
    anime_column = rows["anime_id"].to_numpy()
    user_ids = np.sort(np.unique(user_column))
    # номер строки и столбца для каждой пары
    row_index = np.searchsorted(user_ids, user_column)
    column_index = np.searchsorted(item_ids, anime_column)
    # 1 там, где пользователь досмотрел аниме
    matrix = np.zeros((len(user_ids), len(item_ids)), dtype=np.float32)
    matrix[row_index, column_index] = 1.0
    return matrix, user_ids


def build_train_matrix(
    table: pd.DataFrame, catalog: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Строит матрицу «пользователь × аниме» по train-пользователям (бинарное completed).

    :param table: Таблица взаимодействий из config.INTERACTIONS.
    :param catalog: Массив anime_id допустимого каталога.
    :return: (matrix, user_ids, item_ids): matrix формы (n_users, n_items) из 0.0 и 1.0
        типа float32; строки соответствуют отсортированным user_ids, столбцы –
        отсортированным item_ids. Память: n_users × n_items × 4 байта (~40 МБ).
    """
    train = table[(table["split"] == "train") & (table["status"] == "completed")]
    item_ids = np.sort(catalog)
    matrix, user_ids = build_matrix(train, item_ids)
    logs.matrix(logger, "матрица train: пользователи x аниме, 1 = досмотрел", matrix)
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

    Что считаем: для пары аниме A и B – взвешенное число общих зрителей,
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
    zeros = np.zeros_like(co_viewers)
    similarity = np.divide(co_viewers, denominator, out=zeros, where=denominator > 0)

    np.fill_diagonal(similarity, 0.0)  # аниме не должно быть похоже само на себя
    logs.matrix(logger, "матрица сходства аниме x аниме", similarity)
    return similarity.astype(np.float32)


def check_similarity(similarity: np.ndarray) -> None:
    """Проверяет форму и свойства матрицы сходства; при нарушении бросает ValueError.

    :param similarity: Результат item_similarity.
    """
    n = similarity.shape[0]
    require(similarity.shape == (n, n), "Similarity must be square")
    finite = np.isfinite(similarity).all()
    require(finite, "Similarity contains NaN or inf")
    symmetric = np.allclose(similarity, similarity.T, atol=1e-5)
    require(symmetric, "Similarity must be symmetric")
    diagonal = np.diag(similarity)
    require(np.allclose(diagonal, 0.0), "Diagonal of similarity must be zero")
    in_range = similarity.min() >= -1e-6 and similarity.max() <= 1 + 1e-5
    require(in_range, "Similarity must be in [0, 1]")


def build_histories(table: pd.DataFrame, group: str, item_ids: np.ndarray) -> dict[int, np.ndarray]:
    """Для каждого val/test-пользователя собирает индексы его досмотренного из input.

    :param table: Таблица взаимодействий из config.INTERACTIONS.
    :param group: 'val' или 'test'.
    :param item_ids: Отсортированный массив anime_id (порядок столбцов матрицы).
    :return: {user_id: массив индексов столбцов}; досмотренное из input
        (protocol.completed_input), спрятанные target сюда не попадают.
    """
    completed = protocol.completed_input(table, group)
    histories = {user_id: np.searchsorted(item_ids, anime_ids) for user_id, anime_ids in completed.items()}
    # размеры историй для лога (только для отладки)
    if logger.isEnabledFor(logging.DEBUG):
        sizes = pd.Series({u: len(h) for u, h in histories.items()}, name="аниме в истории")
        described = sizes.describe().round(1)
        logs.table(logger, f"размер истории {group}-пользователей", described, rows=10)
    return histories


def score_candidates(similarity: np.ndarray, history: np.ndarray) -> np.ndarray:
    """Балл каждого аниме каталога для пользователя по его истории.

    :param similarity: Матрица сходства (n_items, n_items).
    :param history: Индексы столбцов (не anime_id) аниме, которые досмотрел
        пользователь; может быть пустым.
    :return: Вектор формы (n_items,): чем больше, тем лучше кандидат.

    Идея: балл аниме – сумма его сходств с тем, что пользователь уже досмотрел.

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
    similarity: np.ndarray,
    item_ids: np.ndarray,
    histories: dict[int, np.ndarray],
    audience: np.ndarray | None = None,
    alpha: float = 0.0,
) -> protocol.Ranker:
    """Ранжер по сходству с историей пользователя.

    Пользователь без истории получает нулевые баллы, и порядок для него задаётся
    только по anime_id (в будущем сюда можно подставить популярность).

    Поправка на популярность (эксперимент E02c): балл кандидата делится на
    размер его аудитории в степени alpha. При alpha = 0 поправки нет; чем больше
    alpha, тем сильнее ослабляются хиты.

    :param similarity: Матрица сходства (n_items, n_items).
    :param item_ids: Отсортированный массив anime_id (порядок строк и столбцов).
    :param histories: Результат build_histories.
    :param audience: Число train-пользователей, досмотревших каждое аниме, форма
        (n_items,); нужно, если alpha > 0 (matrix.sum(axis=0)).
    :param alpha: Сила поправки на популярность, не меньше 0.
    :return: Функция ranker(user_id, candidates).
    """
    empty = np.array([], dtype=int)
    if alpha > 0:
        require(audience is not None, "audience is required when alpha > 0")
        penalty = np.maximum(audience, 1.0) ** alpha  # популярные аниме делим на большее число

    def score_user(user_id: int) -> np.ndarray:
        # сумма сходств каждого аниме с историей пользователя
        history = histories.get(user_id, empty)
        scores = score_candidates(similarity, history)
        # поправка на популярность: популярные аниме делим на большее число
        if alpha > 0:
            scores = scores / penalty
        return scores

    return protocol.make_score_ranker(score_user, item_ids)


def check_criterion(results: pd.DataFrame, model: str, baseline: str = "popularity") -> dict[str, bool]:
    """Проверяет критерий успеха: точность не ниже EVALUATION.accuracy_share от baseline, покрытие в EVALUATION.coverage_factor раз больше.

    :param results: Таблица метрик (колонки – модели, строки – метрики).
    :param model: Название оцениваемой модели.
    :param baseline: Название опорной модели.
    :return: {'accuracy': bool, 'coverage': bool}.
    """
    ndcg_min = EVALUATION.accuracy_share * results.loc["ndcg@10", baseline]
    coverage_min = EVALUATION.coverage_factor * results.loc["coverage@10", baseline]
    accurate = results.loc["ndcg@10", model] >= ndcg_min
    diverse = results.loc["coverage@10", model] >= coverage_min
    return {"accuracy": accurate, "coverage": diverse}


def main(group: str = "val") -> None:
    """Строит item-item модель по train, оценивает её на val и сравнивает с популярностью.

    :param group: 'val' или 'test'. Тест оставляем для финального сравнения.
    """
    table, catalog = store.load_interactions()
    cases = protocol.build_cases(table, group)

    matrix, user_ids, item_ids = build_train_matrix(table, catalog)
    weights = user_weights(matrix)
    similarity = item_similarity(matrix, weights)
    check_similarity(similarity)

    histories = build_histories(table, group, item_ids)
    popularity_scores = popularity.popularity_scores(table)
    rankers = {
        "popularity": popularity.make_popularity_ranker(popularity_scores),
        "item_knn": make_item_knn_ranker(similarity, item_ids, histories),
    }
    # одна и та же оценка для всех моделей на одних и тех же пользователях
    metrics_by_model = {}
    for name, ranker in rankers.items():
        metrics_by_model[name] = protocol.evaluate(ranker, cases, catalog)
    results = pd.DataFrame(metrics_by_model)

    print(f"group={group}, users={len(cases)}, catalog={len(catalog)}, train users={len(user_ids)}")
    rounded = results.round(4)
    print(rounded.to_string())
    criterion = check_criterion(results, "item_knn")
    print("criterion:", criterion)


if __name__ == "__main__":
    # подробные логи по шагам; уровень – переменная окружения RECSYS_LOG_LEVEL (по умолчанию INFO, таблицы – DEBUG)
    logs.enable()
    main()
