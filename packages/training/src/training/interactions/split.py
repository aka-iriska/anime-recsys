from __future__ import annotations

import numpy as np
import pandas as pd

from core import config


MIN_POSITIVES_PER_USER = 5  # пользователь с меньшим числом позитивов всегда в train
VAL_USER_FRACTION = 0.1  # доля оцениваемых пользователей в val
TEST_USER_FRACTION = 0.1  # доля оцениваемых пользователей в test
TARGET_FRACTION = 0.2  # доля позитивов val/test-пользователя, уходящая в target
SPLITS = ("train", "val_input", "val_target", "test_input", "test_target")
TARGET_RULE = "no anime whose direct sequel the user has also completed"  # правило pick_targets для manifest
VISIBLE_SPLITS = ("train", "val_input", "test_input")  # строки, известные модели
TARGET_SPLITS = ("val_target", "test_target")


def assign_user_groups(table: pd.DataFrame) -> pd.Series:
    """Возвращает группу каждого пользователя: 'train', 'val' или 'test'.

    Выбирает пользователей с >= MIN_POSITIVES_PER_USER позитивов, делит их на train, val (10%) и test(10%).

    :param table: с user_id и is_positive.
    :returns Series: индекс — user_id.
    """

    positives = table.loc[table["is_positive"]].groupby("user_id").size()
    groups = pd.Series("train", index=np.sort(table["user_id"].unique()), name="user_group")

    eligible = np.sort(positives[positives >= MIN_POSITIVES_PER_USER].index.to_numpy())
    shuffled = np.random.default_rng(config.SEED).permutation(eligible)
    n_test = round(TEST_USER_FRACTION * len(eligible))
    n_val = round(VAL_USER_FRACTION * len(eligible))

    groups.loc[shuffled[:n_test]] = "test"
    groups.loc[shuffled[n_test:n_test + n_val]] = "val"
    return groups


def pick_targets(watched: np.ndarray, sequels: dict[int, set[int]], rng: np.random.Generator,
                 fraction: float = TARGET_FRACTION) -> np.ndarray:
    """Какие из досмотренных аниме спрятать: fraction от них, но без «дыр» во франшизах.

    Прятать можно только аниме, продолжение которого пользователь не досмотрел. Иначе в
    истории остаётся, например, 1-й и 3-й сезон, а спрятан 2-й: такую «дыру» легко угадать
    по связям франшизы, а в продукте её почти не бывает — сезоны смотрят по порядку. Задача
    проверки — угадать, что человек посмотрит дальше, а не что пропущено в середине.

    :param watched: anime_id досмотренных пользователем аниме (позитивы).
    :param sequels: Результат franchise.load_sequels: anime_id → его прямые продолжения.
    :param rng: Генератор случайных чисел (для воспроизводимости).
    :param fraction: Доля досмотренного, которую прячем (не меньше одного аниме).
    :return: anime_id спрятанных аниме: fraction от всего досмотренного, но не больше, чем
        можно спрятать. Предел срабатывает редко (на interactions_v2 — у 2 из 1731
        пользователей): обычно можно спрятать около 70% истории, а прячется 20%. Пустой
        массив, если спрятать нечего.
    """
    watched_set = set(watched.tolist())
    # можно прятать: ни одно прямое продолжение аниме не досмотрено
    hideable = [a for a in watched.tolist() if not (sequels.get(a, set()) & watched_set)]
    # сколько прятать: доля от всего досмотренного, но не больше, чем можно
    n_target = max(1, round(fraction * len(watched)))
    n_target = min(n_target, len(hideable))
    return rng.choice(np.array(hideable, dtype=watched.dtype), size=n_target, replace=False)


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
    rng = np.random.default_rng(config.SEED)
    for _, rows in candidates.groupby("user_id"):
        # 20% позитивов пользователя — в target, но не аниме с досмотренным продолжением
        hidden = pick_targets(rows["anime_id"].to_numpy(), sequels, rng)
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
