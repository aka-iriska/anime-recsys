from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from implicit.cpu.als import AlternatingLeastSquares

from core import config, logs
from core.validation import require
from core.catalog import franchise
from core.settings import PROTOCOL, RANKING, RETRIEVAL, as_manifest
from training.evaluation import protocol
from training.interactions import split, store
from training.retrieval import als
from training.retrieval import item_knn as knn


OUTPUT = config.RANKING_DATA

# подробности по шагам; включить: core.logs.enable("DEBUG")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UserSplit:
    """Что модель знает о пользователе и что у него спрятано (номера столбцов матрицы).

    :param user_id: Пользователь.
    :param input_completed: Досмотренное из input – по нему считается вектор пользователя.
    :param input_all: Всё из input (любой статус) – это не может быть кандидатом.
    :param target: Спрятанные ответы – будущие метки 1.
    :param fold: Часть для train-пользователя, -1 для val/test.
    :param row: Строка пользователя в обучающей матрице, -1 для val/test (их там нет).
    """

    user_id: int
    input_completed: np.ndarray
    input_all: np.ndarray
    target: np.ndarray
    fold: int = -1
    row: int = -1


def split_train_users(table: pd.DataFrame, item_ids: np.ndarray, train_users: np.ndarray) -> dict[int, UserSplit]:
    """Для train-пользователей с >= RANKING.min_completed_per_user completed: часть, input и target.

    target выбирается тем же правилом, что у val/test (split.pick_targets): без аниме,
    продолжение которого пользователь досмотрел.

    :param table: Таблица взаимодействий из config.INTERACTIONS.
    :param item_ids: Отсортированные anime_id каталога (порядок столбцов матрицы).
    :param train_users: user_id в порядке строк матрицы из knn.build_train_matrix.
    :return: {user_id: UserSplit} с номерами столбцов матрицы, частью и строкой в матрице.
    """
    rng = np.random.default_rng(PROTOCOL.seed)
    sequels = franchise.load_sequels()

    # все строки train-пользователей (любой статус), в фиксированном порядке для воспроизводимости
    train = table[table["split"] == "train"]
    train = train.sort_values(["user_id", "anime_id"])
    by_user = {}
    for user_id, rows in train.groupby("user_id"):
        by_user[user_id] = rows

    # номер строки каждого пользователя в обучающей матрице
    row_of = {}
    for row, user_id in enumerate(train_users):
        row_of[int(user_id)] = row

    # оставляем пользователей, у которых достаточно completed, чтобы было что прятать
    eligible = []
    for user_id, rows in by_user.items():
        n_completed = (rows["status"] == "completed").sum()
        if int(user_id) in row_of and n_completed >= RANKING.min_completed_per_user:
            eligible.append(int(user_id))

    # случайно раскладываем их по RANKING.n_folds частям примерно равного размера
    shuffled = rng.permutation(len(eligible))
    folds = shuffled % RANKING.n_folds

    users = {}
    n_without_target = 0
    for user_id, fold in zip(eligible, folds):
        rows = by_user[user_id]

        # все аниме пользователя как номера столбцов матрицы и отдельно его completed
        anime_ids = rows["anime_id"].to_numpy()
        cols = np.searchsorted(item_ids, anime_ids)
        is_completed = (rows["status"] == "completed").to_numpy()
        completed = cols[is_completed]

        # прячем 20% completed без «дыр» во франшизах: это будущие метки 1
        completed_ids = item_ids[completed]
        hidden = split.pick_targets(completed_ids, sequels, rng, PROTOCOL.target_fraction)
        target = np.searchsorted(item_ids, hidden)
        # спрятать нечего: пользователь не даёт примеров
        if len(target) == 0:
            n_without_target += 1
            continue

        # input – всё остальное: по completed из него считаем вектор, всё из него исключаем из кандидатов
        input_completed = np.setdiff1d(completed, target)
        input_all = np.setdiff1d(cols, target)
        users[user_id] = UserSplit(user_id=user_id, input_completed=input_completed, input_all=input_all,
                                   target=np.sort(target), fold=int(fold), row=row_of[user_id])
    logger.info("train-пользователей без аниме, которые можно спрятать (пропущены): %d из %d", n_without_target, len(eligible))

    # правило протокола: ни у одного спрятанного аниме продолжение не лежит в input
    holes = 0
    for user in users.values():
        kept_ids = item_ids[user.input_completed]
        kept = set(kept_ids.tolist())
        target_ids = item_ids[user.target]
        for anime_id in target_ids.tolist():
            own_sequels = sequels.get(anime_id, set())
            holes += len(own_sequels & kept)
    require(holes == 0, f"{holes} target anime of train users are prequels of their input")

    # сводка для лога: сколько пользователей в части и сколько у них input / target (только для отладки)
    if logger.isEnabledFor(logging.DEBUG):
        summary_rows = []
        for user in users.values():
            summary_rows.append({"fold": user.fold, "input": len(user.input_completed), "target": len(user.target)})
        summary = pd.DataFrame(summary_rows)
        by_fold = summary.groupby("fold").agg(users=("input", "size"), input_mean=("input", "mean"), target_mean=("target", "mean"))
        by_fold = by_fold.round(1)
        logs.table(logger, "train-пользователи по частям (input – знает модель, target – спрятано)", by_fold)
    return users


def input_histories(table: pd.DataFrame, item_ids: np.ndarray, users: dict[int, UserSplit], group: str = "val") -> pd.DataFrame:
    """Что «знает» модель о каждом пользователе датасета: его completed из input.

    Признаки пары считаются только по этой истории: спрятанные ответы в неё не входят.

    :param table: Таблица взаимодействий из config.INTERACTIONS.
    :param item_ids: Отсортированные anime_id каталога.
    :param users: Результат split_train_users.
    :param group: Группа проверочных пользователей: "val" или "test".
    :return: DataFrame с колонками user_id, anime_id (только аниме каталога).
    """
    frames = []
    # train: completed без спрятанных 20% – тот же input, что видела модель части
    for user_id, user in users.items():
        anime_ids = item_ids[user.input_completed]
        frame = pd.DataFrame({"user_id": user_id, "anime_id": anime_ids})
        frames.append(frame)

    # val/test: completed из их input
    histories = knn.build_histories(table, group, item_ids)
    for user_id, columns in histories.items():
        anime_ids = item_ids[columns]
        frame = pd.DataFrame({"user_id": user_id, "anime_id": anime_ids})
        frames.append(frame)
    history = pd.concat(frames, ignore_index=True)
    logs.table(logger, "input-истории (что модель знает о пользователях)", history)
    return history


def top_candidates(model, input_completed: np.ndarray, input_all: np.ndarray, n_items: int) -> tuple[np.ndarray, np.ndarray]:
    """Топ-RETRIEVAL.n_candidates кандидатов iALS для одного пользователя.

    :param model: Обученная iALS-модель (als.fit_als).
    :param input_completed: Индексы его completed из input (для вектора пользователя).
    :param input_all: Индексы всех его аниме из input (не могут быть кандидатами).
    :param n_items: Число аниме в каталоге.
    :return: (индексы кандидатов, их баллы), по убыванию балла.
    """

    # баллы всех аниме по истории пользователя – тот же шаг, что в ленте iALS
    scores = als.score_items(model, input_completed, n_items)
    scores = scores.astype(np.float64)

    # уже знакомое пользователю не рекомендуем
    scores[input_all] = -np.inf

    # быстро берём индексы RETRIEVAL.n_candidates лучших без полной сортировки
    partitioned = np.argpartition(-scores, RETRIEVAL.n_candidates)
    top = partitioned[:RETRIEVAL.n_candidates]
    # индексы внутри scores[top] по убыванию балла
    top_scores = scores[top]
    order = np.argsort(-top_scores, kind="stable")
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

    # место у retrieval: 1 – лучший кандидат
    ranks = np.arange(1, len(top) + 1)
    # метка 1, если кандидат из спрятанных ответов пользователя
    is_hidden = np.isin(top, target)
    labels = is_hidden.astype(int)
    return pd.DataFrame({
        "group": group,
        "fold": fold,
        "user_id": user_id,
        "anime_id": item_ids[top],
        "retrieval_score": scores,
        "retrieval_rank": ranks,
        "label": labels,
    })


def fit_without(matrix: np.ndarray, rows_out: np.ndarray) -> AlternatingLeastSquares:
    """iALS, обученный на всех строках матрицы, кроме rows_out.

    Для train-части это пользователи части (для них модель должна быть «новой», как для
    val-пользователей); для val – пустой список: val-пользователей в матрице и так нет.

    :param matrix: Матрица (n_users, n_items) из knn.build_train_matrix.
    :param rows_out: Номера строк, которые модель не должна видеть.
    :return: Обученная модель (als.fit_als с RETRIEVAL.factors).
    """
    keep = np.ones(matrix.shape[0], dtype=bool)
    keep[rows_out] = False
    n_kept = logs.num(keep.sum())
    n_rows = logs.num(matrix.shape[0])
    logger.debug("iALS учится на %s строках матрицы из %s", n_kept, n_rows)
    confidence = als.build_confidence_matrix(matrix[keep])
    return als.fit_als(confidence, RETRIEVAL.factors)


def candidates_for(model, users: list[UserSplit], group: str, item_ids: np.ndarray) -> pd.DataFrame:
    """Строки датасета для пользователей, которых модель не видела: кандидаты и метки.

    :param model: Результат fit_without.
    :param users: Пользователи (их input и спрятанное).
    :param group: "train", "val" или "test".
    :param item_ids: Отсортированные anime_id каталога.
    :return: DataFrame кандидатов всех этих пользователей (см. candidate_rows).
    """
    frames = []
    n_items = len(item_ids)
    for user in users:
        # кандидаты по input и метки по спрятанному
        top, scores = top_candidates(model, user.input_completed, user.input_all, n_items)
        rows = candidate_rows(group, user.fold, user.user_id, top, scores, user.target, item_ids)
        frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def build_train_candidates(matrix: np.ndarray, item_ids: np.ndarray, users: dict[int, UserSplit]) -> pd.DataFrame:
    """Кандидаты для train-пользователей: iALS для каждой части обучен без неё.

    Обучая iALS без части, мы делаем её пользователей новыми для модели, как val-пользователи для итоговой модели.
    Стоимость: RANKING.n_folds обучений iALS (около секунды каждое) и один fold-in на пользователя.

    :param matrix: Матрица (n_users, n_items) из knn.build_train_matrix.
    :param item_ids: Отсортированные anime_id каталога.
    :param users: Результат split_train_users.
    :return: DataFrame кандидатов всех train-пользователей (см. candidate_rows).
    """
    frames = []
    for fold in range(RANKING.n_folds):
        # пользователи этой части и iALS без них
        in_fold = [user for user in users.values() if user.fold == fold]
        rows_out = np.array([user.row for user in in_fold])
        model = fit_without(matrix, rows_out)
        # кандидаты и метки для каждого пользователя части
        fold_candidates = candidates_for(model, in_fold, "train", item_ids)
        frames.append(fold_candidates)
    candidates = pd.concat(frames, ignore_index=True)
    logs.table(logger, "кандидаты train (100 на пользователя)", candidates)
    return candidates


def build_eval_candidates(table: pd.DataFrame,
                          catalog: np.ndarray,
                          matrix: np.ndarray,
                          item_ids: np.ndarray,
                          group: str = "val"
                          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Кандидаты для val (или test): iALS обучен на всех train-пользователях.

    :param table: Таблица взаимодействий из config.INTERACTIONS.
    :param catalog: Массив anime_id каталога.
    :param matrix: Матрица (n_users, n_items) из knn.build_train_matrix.
    :param item_ids: Отсортированные anime_id каталога.
    :param group: "val" или "test".
    :return: (кандидаты, число спрятанных ответов на пользователя).
    """
    # итоговый iALS по всем train-пользователям: val-пользователей он не видел
    nobody = np.array([], dtype=int)
    model = fit_without(matrix, nobody)

    # для каждого пользователя: что он уже знает (seen), что спрятано (targets) и его completed из input
    cases = protocol.build_cases(table, group)
    histories = knn.build_histories(table, group, item_ids)
    empty = np.array([], dtype=int)
    users = []
    for user_id, (seen, targets) in cases.items():
        # anime_id → номера столбцов матрицы
        seen_ids = np.array(sorted(seen), dtype=int)
        target_ids = np.array(sorted(targets), dtype=int)
        input_all = np.searchsorted(item_ids, seen_ids)
        target = np.searchsorted(item_ids, target_ids)
        input_completed = histories.get(user_id, empty)
        users.append(UserSplit(user_id=user_id, input_completed=input_completed, input_all=input_all, target=target))

    candidates = candidates_for(model, users, group, item_ids)
    # сколько ответов спрятано у каждого пользователя (для recall_of_candidates)
    target_rows = []
    for user in users:
        target_rows.append({"group": group, "user_id": user.user_id, "n_targets": len(user.target)})
    n_targets = pd.DataFrame(target_rows)
    logs.table(logger, f"кандидаты {group} (100 на пользователя)", candidates)
    return candidates, n_targets


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
    n_targets = [len(user.target) for user in users.values()]
    train_targets = pd.DataFrame({"group": "train", "user_id": list(users), "n_targets": n_targets})

    # val: кандидаты от iALS по всем train-пользователям
    val_candidates, val_targets = build_eval_candidates(table, catalog, matrix, item_ids, "val")

    # общий датасет: train и val в одной таблице, различаются колонкой group
    candidates = pd.concat([train_candidates, val_candidates], ignore_index=True)
    targets = pd.concat([train_targets, val_targets], ignore_index=True)
    return candidates, targets


def recall_of_candidates(candidates: pd.DataFrame, targets: pd.DataFrame) -> pd.Series:
    """recall@RETRIEVAL.n_candidates по группам и частям: доля спрятанных ответов, попавших в кандидаты.

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
    users_train = candidates.loc[is_train, "user_id"].nunique()
    users_val = candidates.loc[is_val, "user_id"].nunique()
    positive_share = candidates["label"].mean()
    recall_by_group_fold = {}
    for (group, fold), value in recall.items():
        recall_by_group_fold[f"{group}/{fold}"] = round(float(value), 4)
    summary = {
        "rows": len(candidates),
        "users_train": int(users_train),
        "users_val": int(users_val),
        "positive_share": round(float(positive_share), 4),
        "recall_by_group_fold": recall_by_group_fold,
    }

    # сохранение: кандидаты, число спрятанных ответов, параметры сборки
    OUTPUT.mkdir(parents=True)
    candidates.to_csv(OUTPUT / "candidates.csv", index=False)
    targets.to_csv(OUTPUT / "targets.csv", index=False)
    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": as_manifest(protocol=PROTOCOL, retrieval=RETRIEVAL, ranking=RANKING),
        "summary": summary,
    }
    with (OUTPUT / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(summary_text)
    print(f"Built {OUTPUT}")


if __name__ == "__main__":
    # подробные логи по шагам; уровень – переменная окружения RECSYS_LOG_LEVEL (по умолчанию INFO, таблицы – DEBUG)
    logs.enable()
    main()
