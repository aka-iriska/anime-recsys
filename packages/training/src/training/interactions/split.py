from __future__ import annotations

import numpy as np
import pandas as pd

from core import config


MIN_POSITIVES_PER_USER = 5  # пользователь с меньшим числом позитивов всегда в train
VAL_USER_FRACTION = 0.1  # доля оцениваемых пользователей в val
TEST_USER_FRACTION = 0.1  # доля оцениваемых пользователей в test
TARGET_FRACTION = 0.2  # доля позитивов val/test-пользователя, уходящая в target
SPLITS = ("train", "val_input", "val_target", "test_input", "test_target")
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


def split_rows(table: pd.DataFrame, groups: pd.Series) -> pd.Series:
    """Возвращает split для каждой строки: значения из SPLITS.

    :param table: (user_id, anime_id, is_positive)
    :param groups: Группы пользователей на train, val и test.
    :return: Строки train с пометками датасета строк.
    """

    group = table["user_id"].map(groups)
    split = pd.Series("train", index=table.index)

    is_eval = group.isin(["val", "test"])
    split[is_eval] = group[is_eval] + "_input"

    # кандидаты: с позитивной оценкой и для test/val
    candidates = table[table["is_positive"] & is_eval].sort_values(["user_id", "anime_id"])
    rng = np.random.default_rng(config.SEED)
    for _, rows in candidates.groupby("user_id"): # у этих кандидатов берем 20% данных для предсказания
        k = max(1, round(TARGET_FRACTION * len(rows)))
        chosen = rng.choice(rows.index.to_numpy(), size=k, replace=False)
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
