from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pandas as pd

from core import config, logs
from core.catalog import franchise
from core.catalog.rules import build_catalog, restrict_to_catalog
from core.settings import PROTOCOL, as_manifest
from training.interactions.labels import add_above_mean, mark_positives
from training.interactions.split import (
    SPLITS, TARGET_SPLITS, VISIBLE_SPLITS, assign_user_groups, drop_unusable_users, split_rows,
)
from core.validation import require

# подробности по шагам; включить: core.logs.enable("DEBUG")
logger = logging.getLogger(__name__)


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает (rates, anime)."""

    rates = pd.read_csv(
        config.SOURCE / "user_anime_rates.csv",
        usecols=["user_id", "anime_id", "score", "status", "episodes"],
    )
    anime = pd.read_csv(config.SOURCE / "anime.csv", usecols=["anime_id", "kind"])
    return rates, anime


def check_protocol(table: pd.DataFrame, catalog: set[int], sequels: dict[int, set[int]]) -> dict[str, int]:
    """Валидация данных и протокола.

    Проверяет: нет повторов пары пользователь–аниме, все аниме в каталоге,
    пользователь целиком в одной группе, target состоит из позитивов, в target нет
    «дыр» (аниме, продолжение которого есть в input), каталог построен по
    train-пользователям. При нарушении бросает ValueError.

    :param table: Итоговая таблица с колонкой split.
    :param catalog: Множество anime_id разрешённого каталога.
    :param sequels: Результат franchise.load_sequels.
    :return: Сводка: число строк, пользователей и позитивов по каждому split.
    """
    # нет повторов пары, всё в каталоге, известные значения split
    has_repeats = table.duplicated(["user_id", "anime_id"]).any()
    require(not has_repeats, "Repeated user-anime pair")
    in_catalog = table["anime_id"].isin(catalog)
    require(in_catalog.all(), "Row outside the catalog")
    split_values = set(table["split"].unique())
    require(split_values <= set(SPLITS), "Unknown split value")

    # каждый пользователь целиком в одной группе
    allowed = {
        frozenset({"train"}),
        frozenset({"val_input", "val_target"}),
        frozenset({"test_input", "test_target"}),
    }
    per_user = table.groupby("user_id")["split"].agg(frozenset)
    in_one_group = per_user.isin(allowed)
    require(in_one_group.all(), "User is split across groups or lacks input/target")

    targets = table[table["split"].isin(TARGET_SPLITS)]
    require(targets["is_positive"].all(), "Target contains non-positive rows")

    # «дыр» нет: ни у одного спрятанного аниме продолжение не лежит в input-позитивах
    inputs = table[table["split"].isin(["val_input", "test_input"]) & table["is_positive"]]
    input_pairs = set(zip(inputs["user_id"], inputs["anime_id"]))
    holes = 0
    for user_id, anime_id in zip(targets["user_id"], targets["anime_id"]):
        for sequel in sequels.get(anime_id, set()):
            holes += (user_id, sequel) in input_pairs
    require(holes == 0, f"{holes} target rows are prequels of input positives")

    # каталог построен только по train-пользователям
    train_completed = (
        table[(table["split"] == "train") & (table["status"] == "completed")]
        .groupby("anime_id").size()
        .reindex(list(catalog), fill_value=0)
    )
    above_threshold = train_completed >= PROTOCOL.min_completed_per_anime
    require(above_threshold.all(), "Catalog anime below the train threshold")

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
    catalog_ids = pd.Series(sorted(catalog), name="anime_id")
    catalog_ids.to_csv(config.INTERACTIONS / "catalog.csv", index=False)
    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(config.SOURCE.relative_to(config.ROOT)),
        "config": as_manifest(protocol=PROTOCOL),
        "summary": summary,
    }
    with (config.INTERACTIONS / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    """Собирает config.INTERACTIONS: типы и флаги, разбиение, каталог по train, проверки.

    Не перезаписывает существующий config.INTERACTIONS: при его наличии бросает FileExistsError.
    """
    if config.INTERACTIONS.exists():
        raise FileExistsError(f"Output already exists; nothing was overwritten: {config.INTERACTIONS}")

    rates, anime = load_tables()
    sequels = franchise.load_sequels()

    # 1. только допустимые типы; флаги по строке; группы пользователей и split
    is_kept_kind = anime["kind"].isin(PROTOCOL.kept_kinds)
    kind_ids = set(anime.loc[is_kept_kind, "anime_id"])
    kept_rates = restrict_to_catalog(rates, kind_ids)
    table = mark_positives(kept_rates)
    groups = assign_user_groups(table)

    # 2. каталог только по train-пользователям и ограничение каталогом – до выбора target,
    # чтобы правило «без дыр» у val/test судило по тем же аниме, что у train-частей ranking
    train_rows = table[table["user_id"].map(groups) == "train"]
    catalog = build_catalog(anime, train_rows)
    table = restrict_to_catalog(table, catalog)
    table["split"] = split_rows(table, groups, sequels)
    logs.counts(logger, "строки по частям разбиения", table["split"])
    table, dropped_users = drop_unusable_users(table)
    logs.table(logger, "таблица взаимодействий после ограничения каталогом", table)

    # 3. признаки, зависящие от других строк пользователя, – по известной части
    is_visible = table["split"].isin(VISIBLE_SPLITS)
    table = add_above_mean(table, is_visible)

    summary = check_protocol(table, catalog, sequels)
    summary["dropped_eval_users"] = dropped_users
    config.INTERACTIONS.mkdir(parents=True)
    write_outputs(table, catalog, summary)
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(summary_text)
    print(f"Built {config.INTERACTIONS}")


if __name__ == "__main__":
    # подробные логи по шагам; уровень – переменная окружения RECSYS_LOG_LEVEL (по умолчанию INFO, таблицы – DEBUG)
    logs.enable()
    main()
