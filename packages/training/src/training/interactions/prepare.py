from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pandas as pd

from core import config, logs
from core.catalog.rules import KEPT_KINDS, MIN_COMPLETED_PER_ANIME, build_catalog, restrict_to_catalog
from training.interactions.labels import MAIN_POSITIVE, MIN_EXPLICIT_FOR_MEAN, add_above_mean, mark_positives
from training.interactions.split import (
    MIN_POSITIVES_PER_USER, SPLITS, TARGET_FRACTION, TARGET_SPLITS, TEST_USER_FRACTION,
    VAL_USER_FRACTION, VISIBLE_SPLITS, assign_user_groups, drop_unusable_users, split_rows,
)
from core.validation import require

# подробности по шагам; включить: core.logs.enable()
logger = logging.getLogger(__name__)


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает (rates, anime)."""

    rates = pd.read_csv(
        config.SOURCE / "user_anime_rates.csv",
        usecols=["user_id", "anime_id", "score", "status", "episodes"],
    )
    anime = pd.read_csv(config.SOURCE / "anime.csv", usecols=["anime_id", "kind"])
    return rates, anime


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
    """Записывает interactions.csv, catalog.csv и manifest.json в config.INTERACTIONS.

    :param table: Итоговая таблица взаимодействий.
    :param catalog: Множество anime_id разрешённого каталога.
    :param summary: Сводка из check_protocol, попадает в manifest.json.
    """
    table.to_csv(config.INTERACTIONS / "interactions.csv", index=False)
    pd.Series(sorted(catalog), name="anime_id").to_csv(config.INTERACTIONS / "catalog.csv", index=False)
    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(config.SOURCE.relative_to(config.ROOT)),
        "config": {
            "kept_kinds": list(KEPT_KINDS),
            "min_completed_per_anime": MIN_COMPLETED_PER_ANIME,
            "main_positive": MAIN_POSITIVE,
            "min_positives_per_user": MIN_POSITIVES_PER_USER,
            "min_explicit_for_mean": MIN_EXPLICIT_FOR_MEAN,
            "val_user_fraction": VAL_USER_FRACTION,
            "test_user_fraction": TEST_USER_FRACTION,
            "target_fraction": TARGET_FRACTION,
            "seed": config.SEED,
        },
        "summary": summary,
    }
    with (config.INTERACTIONS / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    """Собирает interactions_v0: типы и флаги, разбиение, каталог по train, проверки.

    Не перезаписывает существующий config.INTERACTIONS: при его наличии бросает FileExistsError.
    """
    if config.INTERACTIONS.exists():
        raise FileExistsError(f"Output already exists; nothing was overwritten: {config.INTERACTIONS}")

    rates, anime = load_tables()

    # 1. только допустимые типы; флаги по строке; группы пользователей и split
    kind_ids = set(anime.loc[anime["kind"].isin(KEPT_KINDS), "anime_id"])
    table = mark_positives(restrict_to_catalog(rates, kind_ids))
    groups = assign_user_groups(table)
    table["split"] = split_rows(table, groups)
    logs.counts(logger, "строки по частям разбиения", table["split"])

    # 2. каталог только по train-пользователям, затем ограничение каталогом
    catalog = build_catalog(anime, table[table["split"] == "train"])
    table = restrict_to_catalog(table, catalog)
    table, dropped_users = drop_unusable_users(table)
    logs.table(logger, "таблица взаимодействий после ограничения каталогом", table)

    # 3. признаки, зависящие от других строк пользователя, — по известной части
    table = add_above_mean(table, table["split"].isin(VISIBLE_SPLITS))

    summary = check_protocol(table, catalog)
    summary["dropped_eval_users"] = dropped_users
    config.INTERACTIONS.mkdir(parents=True)
    write_outputs(table, catalog, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Built {config.INTERACTIONS}")


if __name__ == "__main__":
    # подробные логи по шагам; уровень — переменная окружения RECSYS_LOG_LEVEL (по умолчанию DEBUG)
    logs.enable()
    main()
