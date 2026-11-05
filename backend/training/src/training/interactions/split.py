from __future__ import annotations

import numpy as np
import pandas as pd

from core.settings import PROTOCOL


SPLITS = ("train", "val_input", "val_target", "test_input", "test_target")
VISIBLE_SPLITS = ("train", "val_input", "test_input")  # строки, известные модели
TARGET_SPLITS = ("val_target", "test_target")


def assign_user_groups(table: pd.DataFrame) -> pd.Series:
    """Возвращает группу каждого пользователя: 'train', 'val' или 'test'.

    Выбирает пользователей с >= PROTOCOL.min_positives_per_user позитивов, делит их на train, val (10%) и test(10%).

    :param table: с user_id и is_positive.
    :returns Series: индекс – user_id.
    """

    positives = table.loc[table["is_positive"]].groupby("user_id").size()
    # по умолчанию все пользователи в train
    user_ids = np.sort(table["user_id"].unique())
    groups = pd.Series("train", index=user_ids, name="user_group")

    # оцениваемые – с достаточным числом позитивов, в случайном порядке
    enough = positives[positives >= PROTOCOL.min_positives_per_user]
    eligible = np.sort(enough.index.to_numpy())
    rng = np.random.default_rng(PROTOCOL.seed)
    shuffled = rng.permutation(eligible)
    n_test = round(PROTOCOL.test_user_fraction * len(eligible))
    n_val = round(PROTOCOL.val_user_fraction * len(eligible))

    groups.loc[shuffled[:n_test]] = "test"
    groups.loc[shuffled[n_test:n_test + n_val]] = "val"
    return groups


def pick_targets(watched: np.ndarray, sequels: dict[int, set[int]], rng: np.random.Generator,
                 fraction: float = PROTOCOL.target_fraction) -> np.ndarray:
    """Какие из досмотренных аниме спрятать: fraction от них, но без «дыр» во франшизах.

    Прятать можно только аниме, продолжение которого пользователь не досмотрел. Иначе в
    истории остаётся, например, 1-й и 3-й сезон, а спрятан 2-й: такую «дыру» легко угадать
    по связям франшизы, а в продукте её почти не бывает – сезоны смотрят по порядку. Задача
    проверки – угадать, что человек посмотрит дальше, а не что пропущено в середине.

    :param watched: anime_id досмотренных пользователем аниме (позитивы).
    :param sequels: Результат franchise.load_sequels: anime_id → его прямые продолжения.
    :param rng: Генератор случайных чисел (для воспроизводимости).
    :param fraction: Доля досмотренного, которую прячем (не меньше одного аниме).
    :return: anime_id спрятанных аниме: fraction от всего досмотренного, но не больше, чем
        можно спрятать. Предел срабатывает редко (на interactions_v2 – у 2 из 1731
        пользователей): обычно можно спрятать около 70% истории, а прячется 20%. Пустой
        массив, если спрятать нечего.
    """
    watched_set = set(watched.tolist())
    # можно прятать: ни одно прямое продолжение аниме не досмотрено
    hideable = []
    for anime_id in watched.tolist():
        own_sequels = sequels.get(anime_id, set())
        sequel_watched = bool(own_sequels & watched_set)
        if not sequel_watched:
            hideable.append(anime_id)
    # сколько прятать: доля от всего досмотренного, но не больше, чем можно
    n_target = max(1, round(fraction * len(watched)))
    n_target = min(n_target, len(hideable))
    options = np.array(hideable, dtype=watched.dtype)
    return rng.choice(options, size=n_target, replace=False)


def split_rows(table: pd.DataFrame, groups: pd.Series, sequels: dict[int, set[int]]) -> pd.Series:
    """Возвращает split для каждой строки: значения из SPLITS.

    :param table: (user_id, anime_id, is_positive)
    :param groups: Группы пользователей на train, val и test.
    :param sequels: Результат franchise.load_sequels (чтобы не прятать «дыры», см. pick_targets).
    :return: Строки train с пометками датасета строк.
    """

    group = table["user_id"].map(groups)
    split = pd.Series("train", index=table.index)

    is_eval = group.isin(["val", "test"])
    split[is_eval] = group[is_eval] + "_input"

    # кандидаты: с позитивной оценкой и для test/val
    candidates = table[table["is_positive"] & is_eval].sort_values(["user_id", "anime_id"])
    rng = np.random.default_rng(PROTOCOL.seed)
    for _, rows in candidates.groupby("user_id"):
        # 20% позитивов пользователя – в target, но не аниме с досмотренным продолжением
        watched = rows["anime_id"].to_numpy()
        hidden = pick_targets(watched, sequels, rng)
        chosen = rows.index[rows["anime_id"].isin(hidden)]
        split.loc[chosen] = group.loc[chosen] + "_target"
    return split


def drop_unusable_users(table: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Убирает val/test-пользователей без input- или target-позитива в каталоге.

    :return: (таблица, число убранных пользователей).
    """

    positives = table[table["is_positive"]]
    # input и target для валидационной и тестовой выборок
    n_input = positives[positives["split"].isin(["val_input", "test_input"])].groupby("user_id").size()
    n_target = positives[positives["split"].isin(TARGET_SPLITS)].groupby("user_id").size()

    eval_users = table.loc[table["split"] != "train", "user_id"].unique()
    usable = set(n_input.index[n_input >= 1]) & set(n_target.index[n_target >= 1])
    bad = set(eval_users) - usable
    return table[~table["user_id"].isin(bad)].copy(), len(bad)
