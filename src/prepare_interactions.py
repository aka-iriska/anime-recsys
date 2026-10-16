"""Подготовка таблицы взаимодействий для офлайн-оценки (протокол v0).

Запуск из корня проекта: uv run python src/prepare_interactions.py
Решения и их статус: docs/system_design.md, раздел «Протокол оценки v0».

Схема разбиения: пользователи делятся на train / val / test (по группам).
У val- и test-пользователей позитивы делятся на input (модель видит их при
выдаче) и target (скрыты, по ним считаем метрики). Каталог и признаки
строятся только по train-пользователям.

Вход:  data/processed/shikimori_v1 (user_anime_rates.csv, anime.csv).
Выход: data/processed/interactions_v0/
    interactions.csv  одна строка на пару (user, anime) внутри каталога:
                      флаги позитивов, is_positive и split
    catalog.csv       anime_id разрешённого каталога
    manifest.json     параметры и число строк
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "processed" / "shikimori_v1"
OUTPUT = ROOT / "data" / "processed" / "interactions_v0"

# Protocol v0 parameters; see the status column in docs/system_design.md.
KEPT_KINDS = ("tv", "movie", "ova", "ona", "special", "tv_special")
MIN_COMPLETED_PER_ANIME = 5  # аниме с меньшим числом completed в каталог не входят
MAIN_POSITIVE = "p_completed"  # proposal; the other flags are control slices
MIN_POSITIVES_PER_USER = 5  # пользователь с меньшим числом позитивов всегда в train
MIN_EXPLICIT_FOR_MEAN = 5  # минимум явных оценок, чтобы считать p_above_mean

VAL_USER_FRACTION = 0.1  # доля оцениваемых пользователей в val
TEST_USER_FRACTION = 0.1  # доля оцениваемых пользователей в test
TARGET_FRACTION = 0.2  # доля позитивов val/test-пользователя, уходящая в target
SEED = 42

FLAG_COLUMNS = ["p_completed", "p_score8", "p_above_mean", "n_dropped"]
SPLITS = ("train", "val_input", "val_target", "test_input", "test_target")
VISIBLE_SPLITS = ("train", "val_input", "test_input")  # строки, известные модели
TARGET_SPLITS = ("val_target", "test_target")


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает (rates, anime)."""

    rates = pd.read_csv(
        SOURCE / "user_anime_rates.csv",
        usecols=["user_id", "anime_id", "score", "status", "episodes"],
    )
    anime = pd.read_csv(SOURCE / "anime.csv", usecols=["anime_id", "kind"])
    return rates, anime


def build_catalog(anime: pd.DataFrame, rates: pd.DataFrame) -> set[int]:
    """Возвращает множество уникальных anime_id,
    которые можно использовать в качестве кандидатов для рекомендательной системы.

    :param anime: (anime_id, kind).
    :param rates: Оценки, по которым считается порог (в пайплайне — train-строки).
    :return: Множество уникальных anime_id
    """

    mask = anime["kind"].isin(KEPT_KINDS)
    good_kind = anime.loc[mask, "anime_id"]

    mask = rates["status"] == "completed"
    completed = rates.loc[mask]
    completed_counts = completed.groupby("anime_id").size()  # completed на аниме
    popular = completed_counts[completed_counts >= MIN_COMPLETED_PER_ANIME].index

    return set(good_kind.tolist()) & set(popular.tolist())


def mark_positives(rates: pd.DataFrame) -> pd.DataFrame:
    """Возвращает копию rates с флагами позитивов и колонкой `is_positive`.

    Флаги зависят только от самой строки (status, score), поэтому их можно
    считать до разбиения. `p_above_mean` зависит от других строк пользователя,
    его добавляет add_above_mean после разбиения.

    :param rates: Оценки (user_id, anime_id, score, status, ...).
    :return: Копия rates с колонками p_score8, p_completed, n_dropped, is_positive.
    """

    rates_c = rates.copy()

    rates_c["p_score8"] = rates_c.score >= 8
    rates_c["p_completed"] = rates_c.status.eq("completed")
    rates_c["n_dropped"] = rates_c.status.eq("dropped")
    rates_c["is_positive"] = rates_c[MAIN_POSITIVE]

    return rates_c


def assign_user_groups(table: pd.DataFrame) -> pd.Series:
    """Возвращает группу каждого пользователя: 'train', 'val' или 'test'.

    Выбирает пользователей с >= MIN_POSITIVES_PER_USER позитивов, делит их на train, val (10%) и test(10%).

    :param table: с user_id и is_positive.
    :returns Series: индекс — user_id.
    """

    positives = table.loc[table["is_positive"]].groupby("user_id").size()
    groups = pd.Series("train", index=np.sort(table["user_id"].unique()), name="user_group")

    eligible = np.sort(positives[positives >= MIN_POSITIVES_PER_USER].index.to_numpy())
    shuffled = np.random.default_rng(SEED).permutation(eligible)
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
    rng = np.random.default_rng(SEED)
    for _, rows in candidates.groupby("user_id"): # у этих кандидатов берем 20% данных для предсказания
        k = max(1, round(TARGET_FRACTION * len(rows)))
        chosen = rng.choice(rows.index.to_numpy(), size=k, replace=False)
        split.loc[chosen] = group.loc[chosen] + "_target"
    return split


def restrict_to_catalog(rates: pd.DataFrame, catalog: set[int]) -> pd.DataFrame:
    """Возвращает оценки только на разрешенные аниме."""
    return rates[rates["anime_id"].isin(catalog)].copy()


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


def add_above_mean(table: pd.DataFrame, visible: pd.Series) -> pd.DataFrame:
    """Добавляет `p_above_mean` — оценка выше средней оценки пользователя.

    Среднее считается только по строкам с visible=True (train, input), поэтому
    скрытые target-оценки не влияют на флаг. Флаг применяется ко всем строкам.
    Пользователи с менее чем MIN_EXPLICIT_FOR_MEAN явными оценками получают False.

    :param table: Таблица с user_id и score.
    :param visible: Маска строк, известных модели (split из VISIBLE_SPLITS).
    :return: Копия table с колонкой p_above_mean.
    """

    table = table.copy()
    explicit = table.score.where(table.score > 0)  # где поставили оценки
    seen = explicit.where(visible)  # только известные модели оценки
    user_mean = seen.groupby(table.user_id).transform("mean")
    user_n = seen.groupby(table.user_id).transform("count")

    table["p_above_mean"] = (explicit > user_mean) & (user_n >= MIN_EXPLICIT_FOR_MEAN)
    return table


def require(condition: bool, message: str) -> None:
    """Бросает ValueError с сообщением message, если condition ложно."""
    if not condition:
        raise ValueError(message)


def check_protocol(table: pd.DataFrame, catalog: set[int]) -> dict[str, int]:
    """Валидация данных и протокола.

    Проверяет: нет повторов пары пользователь–аниме, все аниме в каталоге,
    пользователь целиком в одной группе, target состоит из позитивов, каталог
    построен по train-пользователям. При нарушении бросает ValueError.

    :param table: Итоговая таблица с колонкой split.
    :param catalog: Множество anime_id разрешённого каталога.
    :return: Сводка: число строк, пользователей и позитивов по каждому split.
    """
    require(not table.duplicated(["user_id", "anime_id"]).any(), "Repeated user-anime pair")
    require(table["anime_id"].isin(catalog).all(), "Row outside the catalog")
    require(set(table["split"].unique()) <= set(SPLITS), "Unknown split value")

    # каждый пользователь целиком в одной группе
    allowed = {
        frozenset({"train"}),
        frozenset({"val_input", "val_target"}),
        frozenset({"test_input", "test_target"}),
    }
    per_user = table.groupby("user_id")["split"].agg(frozenset)
    require(per_user.isin(allowed).all(), "User is split across groups or lacks input/target")

    targets = table[table["split"].isin(TARGET_SPLITS)]
    require(targets["is_positive"].all(), "Target contains non-positive rows")

    # каталог построен только по train-пользователям
    train_completed = (
        table[(table["split"] == "train") & (table["status"] == "completed")]
        .groupby("anime_id").size()
        .reindex(list(catalog), fill_value=0)
    )
    require((train_completed >= MIN_COMPLETED_PER_ANIME).all(), "Catalog anime below the train threshold")

    positives = table[table["is_positive"]]
    summary = {
        "catalog_anime": len(catalog),
        "rows": len(table),
        "users": int(table["user_id"].nunique()),
        "positives": int(positives.shape[0]),
    }
    for name in SPLITS:
        rows = table[table["split"] == name]
        summary[f"{name}_rows"] = len(rows)
        summary[f"{name}_users"] = int(rows["user_id"].nunique())
        summary[f"{name}_positives"] = int(rows["is_positive"].sum())
    return summary


def write_outputs(table: pd.DataFrame, catalog: set[int], summary: dict[str, int]) -> None:
    """Записывает interactions.csv, catalog.csv и manifest.json в OUTPUT.

    :param table: Итоговая таблица взаимодействий.
    :param catalog: Множество anime_id разрешённого каталога.
    :param summary: Сводка из check_protocol, попадает в manifest.json.
    """
    table.to_csv(OUTPUT / "interactions.csv", index=False)
    pd.Series(sorted(catalog), name="anime_id").to_csv(OUTPUT / "catalog.csv", index=False)
    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE.relative_to(ROOT)),
        "config": {
            "kept_kinds": list(KEPT_KINDS),
            "min_completed_per_anime": MIN_COMPLETED_PER_ANIME,
            "main_positive": MAIN_POSITIVE,
            "min_positives_per_user": MIN_POSITIVES_PER_USER,
            "min_explicit_for_mean": MIN_EXPLICIT_FOR_MEAN,
            "val_user_fraction": VAL_USER_FRACTION,
            "test_user_fraction": TEST_USER_FRACTION,
            "target_fraction": TARGET_FRACTION,
            "seed": SEED,
        },
        "summary": summary,
    }
    with (OUTPUT / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    """Собирает interactions_v0: типы и флаги, разбиение, каталог по train, проверки.

    Не перезаписывает существующий OUTPUT: при его наличии бросает FileExistsError.
    """
    if OUTPUT.exists():
        raise FileExistsError(f"Output already exists; nothing was overwritten: {OUTPUT}")

    rates, anime = load_tables()

    # 1. только допустимые типы; флаги по строке; группы пользователей и split
    kind_ids = set(anime.loc[anime["kind"].isin(KEPT_KINDS), "anime_id"])
    table = mark_positives(restrict_to_catalog(rates, kind_ids))
    groups = assign_user_groups(table)
    table["split"] = split_rows(table, groups)

    # 2. каталог только по train-пользователям, затем ограничение каталогом
    catalog = build_catalog(anime, table[table["split"] == "train"])
    table = restrict_to_catalog(table, catalog)
    table, dropped_users = drop_unusable_users(table)

    # 3. признаки, зависящие от других строк пользователя, — по известной части
    table = add_above_mean(table, table["split"].isin(VISIBLE_SPLITS))

    summary = check_protocol(table, catalog)
    summary["dropped_eval_users"] = dropped_users
    OUTPUT.mkdir(parents=True)
    write_outputs(table, catalog, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Built {OUTPUT}")


if __name__ == "__main__":
    main()
