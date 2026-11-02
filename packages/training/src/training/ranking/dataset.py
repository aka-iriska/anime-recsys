from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from core import config
from training.evaluation import protocol
from training.interactions import store
from training.retrieval import als
from training.retrieval import item_knn as knn


OUTPUT = config.RANKING_DATA

N_FOLDS = 5             # на сколько частей делим train-пользователей
N_CANDIDATES = 100      # сколько кандидатов retrieval передаёт в ranking
FACTORS = 128           # iALS из E05
MIN_COMPLETED = 5       # train-пользователь даёт примеры, если у него столько completed в каталоге
TARGET_FRACTION = 0.2   # доля completed, которую прячем (как у val/test)
SEED = 42


def split_train_users(table: pd.DataFrame, item_ids: np.ndarray, train_users: np.ndarray) -> dict[int, dict]:
    """Для train-пользователей с >= MIN_COMPLETED completed: часть, input и target.

    :param table: Таблица взаимодействий из interactions_v0.
    :param item_ids: Отсортированные anime_id каталога (порядок столбцов матрицы).
    :param train_users: user_id в порядке строк матрицы из knn.build_train_matrix.
    :return: {user_id: {"fold", "row", "input_completed", "input_all", "target"}}, где
        индексы — номера столбцов матрицы; row — номер строки пользователя в матрице.
    """
    rng = np.random.default_rng(SEED)

    # все строки train-пользователей (любой статус), в фиксированном порядке для воспроизводимости
    train = table[table["split"] == "train"]
    train = train.sort_values(["user_id", "anime_id"])
    by_user = dict(tuple(train.groupby("user_id")))

    # номер строки каждого пользователя в обучающей матрице
    row_of = {int(u): i for i, u in enumerate(train_users)}

    # оставляем пользователей, у которых достаточно completed, чтобы было что прятать
    eligible = []
    for user_id, rows in by_user.items():
        n_completed = (rows["status"] == "completed").sum()
        if int(user_id) in row_of and n_completed >= MIN_COMPLETED:
            eligible.append(int(user_id))

    # случайно раскладываем их по N_FOLDS частям примерно равного размера
    shuffled = rng.permutation(len(eligible))
    folds = shuffled % N_FOLDS

    users = {}
    for user_id, fold in zip(eligible, folds):
        rows = by_user[user_id]

        # все аниме пользователя как номера столбцов матрицы и отдельно его completed
        cols = np.searchsorted(item_ids, rows["anime_id"].to_numpy())
        is_completed = (rows["status"] == "completed").to_numpy()
        completed = cols[is_completed]

        # прячем 20% completed (не меньше одного): это будущие метки 1
        n_target = max(1, round(TARGET_FRACTION * len(completed)))
        target = rng.choice(completed, size=n_target, replace=False)

        # input — всё остальное: по completed из него считаем вектор, всё из него исключаем из кандидатов
        users[user_id] = {
            "fold": int(fold),
            "row": row_of[user_id],
            "input_completed": np.setdiff1d(completed, target),
            "input_all": np.setdiff1d(cols, target),
            "target": np.sort(target),
        }
    return users


def top_candidates(model, input_completed: np.ndarray, input_all: np.ndarray, n_items: int) -> tuple[np.ndarray, np.ndarray]:
    """Топ-N_CANDIDATES кандидатов iALS для одного пользователя.

    :param model: Обученная iALS-модель (als.fit_als).
    :param input_completed: Индексы его completed из input (для вектора пользователя).
    :param input_all: Индексы всех его аниме из input (не могут быть кандидатами).
    :param n_items: Число аниме в каталоге.
    :return: (индексы кандидатов, их баллы), по убыванию балла.
    """

    # баллы всех аниме по истории пользователя — тот же шаг, что в ленте iALS
    scores = als.score_items(model, input_completed, n_items)
    scores = scores.astype(np.float64)

    # уже знакомое пользователю не рекомендуем
    scores[input_all] = -np.inf

    # быстро берём индексы N_CANDIDATES лучших без полной сортировки
    top = np.argpartition(-scores, N_CANDIDATES)[:N_CANDIDATES]
    # индексы внутри scores[top] по убыванию балла
    order = np.argsort(-scores[top], kind="stable")
    top = top[order]
    return top, scores[top]


def candidate_rows(group: str, fold: int, user_id: int, top: np.ndarray, scores: np.ndarray,
                   target: np.ndarray, item_ids: np.ndarray) -> pd.DataFrame:
    """Строки датасета для одного пользователя.

    :param group: "train" или "val".
    :param fold: Номер части для train, -1 для val.
    :param user_id: Пользователь.
    :param top: Индексы кандидатов (номера столбцов матрицы), по убыванию балла.
    :param scores: Баллы iALS для top.
    :param target: Индексы спрятанных ответов пользователя.
    :param item_ids: Отсортированные anime_id каталога.
    :return: DataFrame из len(top) строк: group, fold, user_id, anime_id, retrieval_score,
        retrieval_rank, label.
    """

    # место у retrieval: 1 — лучший кандидат
    ranks = np.arange(1, len(top) + 1)
    # метка 1, если кандидат из спрятанных ответов пользователя
    labels = np.isin(top, target).astype(int)
    return pd.DataFrame({
        "group": group,
        "fold": fold,
        "user_id": user_id,
        "anime_id": item_ids[top],
        "retrieval_score": scores,
        "retrieval_rank": ranks,
        "label": labels,
    })


def build_train_candidates(matrix: np.ndarray, item_ids: np.ndarray, users: dict[int, dict]) -> pd.DataFrame:
    """Кандидаты для train-пользователей: iALS для каждой части обучен без неё.

    Обучая iALS без части, мы делаем её пользователей новыми для модели, как val-пользователи для итоговой модели.
    Стоимость: N_FOLDS обучений iALS (около секунды каждое) и один fold-in на пользователя.

    :param matrix: Матрица (n_users, n_items) из knn.build_train_matrix.
    :param item_ids: Отсортированные anime_id каталога.
    :param users: Результат split_train_users.
    :return: DataFrame кандидатов всех train-пользователей (см. candidate_rows).
    """

    frames = []
    n_items = matrix.shape[1]
    for fold in range(N_FOLDS):
        # пользователи этой части и их строки в матрице
        in_fold = [u for u, info in users.items() if info["fold"] == fold]
        rows_out = np.array([users[u]["row"] for u in in_fold])

        # обучаем iALS на всех строках, кроме пользователей этой части
        keep = np.ones(matrix.shape[0], dtype=bool)
        keep[rows_out] = False
        confidence = als.build_confidence_matrix(matrix[keep])
        model = als.fit_als(confidence, FACTORS)

        # для каждого пользователя части: кандидаты по его input и метки по спрятанному
        for user_id in in_fold:
            info = users[user_id]
            top, scores = top_candidates(model, info["input_completed"], info["input_all"], n_items)
            rows = candidate_rows("train", fold, user_id, top, scores, info["target"], item_ids)
            frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def build_eval_candidates(table: pd.DataFrame,
                          catalog: np.ndarray,
                          matrix: np.ndarray,
                          item_ids: np.ndarray,
                          group: str = "val"
                          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Кандидаты для val (или test): iALS обучен на всех train-пользователях.

    :param table: Таблица взаимодействий из interactions_v0.
    :param catalog: Массив anime_id каталога.
    :param matrix: Матрица (n_users, n_items) из knn.build_train_matrix.
    :param item_ids: Отсортированные anime_id каталога.
    :param group: "val" или "test".
    :return: (кандидаты, число спрятанных ответов на пользователя).
    """

    # итоговый iALS по всем train-пользователям: val-пользователей он не видел
    confidence = als.build_confidence_matrix(matrix)
    model = als.fit_als(confidence, FACTORS)

    # для каждого val-пользователя: что он уже знает (seen), что спрятано (targets) и его completed из input
    cases = protocol.build_cases(table, group)
    histories = knn.build_histories(table, group, item_ids)
    empty = np.array([], dtype=int)

    frames, n_targets = [], []
    for user_id, (seen, targets) in cases.items():
        # переводим anime_id в номера столбцов матрицы
        seen_ids = np.array(sorted(seen), dtype=int)
        target_ids = np.array(sorted(targets), dtype=int)
        input_all = np.searchsorted(item_ids, seen_ids)
        target = np.searchsorted(item_ids, target_ids)
        input_completed = histories.get(user_id, empty)

        # кандидаты и строки датасета, как для train
        top, scores = top_candidates(model, input_completed, input_all, len(item_ids))
        rows = candidate_rows(group, -1, user_id, top, scores, target, item_ids)
        frames.append(rows)
        n_targets.append({"group": group, "user_id": user_id, "n_targets": len(target)})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(n_targets)


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Собирает кандидатов для train (по частям) и val.

    :return: (candidates, targets): кандидаты всех пользователей и число спрятанных ответов
        у каждого.
    """
    # данные протокола и матрица train-пользователей
    table, catalog = store.load_interactions()
    matrix, train_users, item_ids = knn.build_train_matrix(table, catalog)

    # train: делим на части, прячем часть просмотров, кандидаты от iALS без своей части
    users = split_train_users(table, item_ids, train_users)
    train_candidates = build_train_candidates(matrix, item_ids, users)
    n_targets = [len(info["target"]) for info in users.values()]
    train_targets = pd.DataFrame({"group": "train", "user_id": list(users), "n_targets": n_targets})

    # val: кандидаты от iALS по всем train-пользователям
    val_candidates, val_targets = build_eval_candidates(table, catalog, matrix, item_ids, "val")

    # общий датасет: train и val в одной таблице, различаются колонкой group
    candidates = pd.concat([train_candidates, val_candidates], ignore_index=True)
    targets = pd.concat([train_targets, val_targets], ignore_index=True)
    return candidates, targets


def recall_of_candidates(candidates: pd.DataFrame, targets: pd.DataFrame) -> pd.Series:
    """recall@N_CANDIDATES по группам и частям: доля спрятанных ответов, попавших в кандидаты.

    Проверка на утечку: у train-частей и у val значение должно быть примерно одинаковым.

    :param candidates: Результат build (или candidates.csv).
    :param targets: Результат build (или targets.csv).
    :return: Series с индексом (group, fold) и средним recall по пользователям.
    """
    # сколько спрятанных ответов нашлось среди кандидатов каждого пользователя
    found = candidates.groupby(["group", "fold", "user_id"])["label"].sum()
    found = found.rename("found").reset_index()
    # делим на число всех спрятанных ответов (не все попадают в кандидаты)
    found = found.merge(targets, on=["group", "user_id"])
    found["recall"] = found["found"] / found["n_targets"]
    # среднее по пользователям внутри каждой группы и части
    return found.groupby(["group", "fold"])["recall"].mean()


def main() -> None:
    """Строит датасет, печатает сводку и сохраняет в OUTPUT (не перезаписывает)."""
    if OUTPUT.exists():
        raise FileExistsError(f"Output already exists; nothing was overwritten: {OUTPUT}")

    # сборка датасета и проверка на утечку
    candidates, targets = build()
    recall = recall_of_candidates(candidates, targets)

    # сводка для печати и manifest.json
    is_train = candidates["group"] == "train"
    is_val = candidates["group"] == "val"
    summary = {
        "rows": len(candidates),
        "users_train": int(candidates.loc[is_train, "user_id"].nunique()),
        "users_val": int(candidates.loc[is_val, "user_id"].nunique()),
        "positive_share": round(float(candidates["label"].mean()), 4),
        "recall_by_group_fold": {f"{g}/{f}": round(float(v), 4) for (g, f), v in recall.items()},
    }

    # сохранение: кандидаты, число спрятанных ответов, параметры сборки
    OUTPUT.mkdir(parents=True)
    candidates.to_csv(OUTPUT / "candidates.csv", index=False)
    targets.to_csv(OUTPUT / "targets.csv", index=False)
    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"n_folds": N_FOLDS, "n_candidates": N_CANDIDATES, "factors": FACTORS,
                   "alpha": als.ALPHA, "regularization": als.REGULARIZATION, "iterations": als.ITERATIONS,
                   "min_completed": MIN_COMPLETED, "target_fraction": TARGET_FRACTION, "seed": SEED},
        "summary": summary,
    }
    with (OUTPUT / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Built {OUTPUT}")


if __name__ == "__main__":
    main()
