"""Сборка проверяемых нормализованных таблиц из сырых выгрузок Shikimori.

Запуск из корня проекта: uv run python src/build_dataset.py
Скрипт создаёт новый каталог data/processed/shikimori_v1 и никогда не меняет и
не удаляет файлы в data/raw.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OLD = RAW / "shikimori_all_status_history_2026-09-24"
EXPANDED = RAW / "shikimori_expanded_users_2000_2026-09-24"
CATALOG = RAW / "shikimori_catalog_2026-09-24"
OUTPUT = ROOT / "data" / "processed" / "shikimori_v1"

csv.field_size_limit(2**31 - 1)


def rows(path: Path):
    """Построчно читает CSV (utf-8-sig) и отдаёт строки как словари.

    :param path: Путь к CSV-файлу.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def writer(path: Path, columns: list[str]):
    """Открывает CSV для записи и пишет заголовок.

    :param path: Путь к создаваемому файлу.
    :param columns: Названия колонок в порядке записи.
    :return: (файл, DictWriter); закрывает файл вызывающий код.
    """
    handle = path.open("w", encoding="utf-8", newline="")
    table = csv.DictWriter(handle, fieldnames=columns)
    table.writeheader()
    return handle, table


def copied(source: Path, name: str) -> None:
    """Копирует исходный файл в каталог OUTPUT под именем name.

    :param source: Путь к исходному файлу.
    :param name: Имя файла в OUTPUT.
    """
    shutil.copyfile(source, OUTPUT / name)


def access(path: Path) -> dict[str, dict[str, str]]:
    """Читает таблицу доступности данных пользователей.

    :param path: Путь к *_access.csv.
    :return: {user_id: строка таблицы}; ValueError, если пользователь повторяется.
    """
    result = {}
    for row in rows(path):
        user_id = row["user_id"]
        if user_id in result:
            raise ValueError(f"Repeated access record: {path}, {user_id}")
        result[user_id] = row
    return result


def build_users() -> int:
    """Строит users.csv: когорта из cohort_manifest.json с доступностью данных.

    Для каждого пользователя сохраняет группу (original/expanded) и доступность
    rates, history и favourites. Доступная, но пустая история отличается от
    недоступной.

    :return: Число пользователей.
    """
    with (EXPANDED / "cohort_manifest.json").open(encoding="utf-8") as handle:
        cohort = json.load(handle)
    user_ids = [str(value) for value in cohort["user_ids"]]
    original_count = int(cohort["original_users"])
    if len(user_ids) != len(set(user_ids)) or len(user_ids) != cohort["target_users"]:
        raise ValueError("Cohort user IDs are missing or repeated")
    old_users = set(user_ids[:original_count])
    new_users = set(user_ids[original_count:])

    rates = access(EXPANDED / "rates_access.csv")
    favourites = access(EXPANDED / "favourites_access.csv")
    old_history = access(OLD / "history_access.csv")
    new_history = access(EXPANDED / "history_access.csv")
    if set(rates) != set(user_ids) or set(favourites) != set(user_ids):
        raise ValueError("Rates or favourites access does not cover the cohort")
    if set(old_history) != old_users or set(new_history) != new_users:
        raise ValueError("History access does not match the two cohort groups")

    columns = [
        "user_id", "cohort", "rates_access", "rates_http_status",
        "history_access", "history_http_status", "favourites_access",
        "favourites_http_status",
    ]
    handle, table = writer(OUTPUT / "users.csv", columns)
    with handle:
        for user_id in user_ids:
            history = old_history[user_id] if user_id in old_users else new_history[user_id]
            table.writerow({
                "user_id": user_id,
                "cohort": "original" if user_id in old_users else "expanded",
                "rates_access": rates[user_id]["access"],
                "rates_http_status": rates[user_id]["http_status"],
                "history_access": history["access"],
                "history_http_status": history["http_status"],
                "favourites_access": favourites[user_id]["access"],
                "favourites_http_status": favourites[user_id]["http_status"],
            })
    return len(user_ids)


def build_anime() -> tuple[int, set[str], dict[str, str]]:
    """Строит anime.csv: переименовывает id в anime_id и проверяет его уникальность.

    :return: (число аниме, множество anime_id, {anime_id: franchise}).
    """
    source = CATALOG / "anime.csv"
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = ["anime_id" if name == "id" else name for name in reader.fieldnames]
        out_handle, table = writer(OUTPUT / "anime.csv", columns)
        anime_ids = set()
        franchises = {}
        count = 0
        with out_handle:
            for row in reader:
                anime_id = row.pop("id")
                if not anime_id or anime_id in anime_ids:
                    raise ValueError(f"Missing or repeated anime ID: {anime_id}")
                anime_ids.add(anime_id)
                franchises[anime_id] = row["franchise"]
                table.writerow({"anime_id": anime_id, **row})
                count += 1
    return count, anime_ids, franchises


def build_rates(users: set[str], anime_ids: set[str]) -> int:
    """Строит user_anime_rates.csv из текущего списка расширенной выгрузки.

    Проверяет уникальность rate_id и пары (пользователь, аниме), что пользователь
    и аниме известны, target_type равен Anime и score лежит в диапазоне 0–10.
    Пропуски и score=0 не меняются.

    :param users: Множество user_id когорты.
    :param anime_ids: Множество anime_id каталога.
    :return: Число строк.
    """
    source = EXPANDED / "users_rates.csv"
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        names = {"id": "rate_id", "target_id": "anime_id"}
        columns = [names.get(name, name) for name in reader.fieldnames]
        out_handle, table = writer(OUTPUT / "user_anime_rates.csv", columns)
        rate_ids = set()
        pairs = set()
        count = 0
        with out_handle:
            for row in reader:
                record = {names.get(key, key): value for key, value in row.items()}
                rate_id = record["rate_id"]
                pair = (record["user_id"], record["anime_id"])
                if not rate_id or rate_id in rate_ids or pair in pairs:
                    raise ValueError(f"Repeated/missing rate ID or user-anime pair: {rate_id}")
                if pair[0] not in users or pair[1] not in anime_ids:
                    raise ValueError(f"Unknown user or anime in rate: {pair}")
                if record["target_type"] != "Anime" or not 0 <= int(record["score"]) <= 10:
                    raise ValueError(f"Unexpected rate target or score: {rate_id}")
                rate_ids.add(rate_id)
                pairs.add(pair)
                table.writerow(record)
                count += 1
    return count


def build_history(users: set[str], anime_ids: set[str]) -> int:
    """Строит user_anime_history.csv: объединяет истории двух групп пользователей.

    Группы не пересекаются. id переименовывается в event_id, created_at — в
    event_at. Для одной пары пользователь–аниме допустимо много событий.

    :param users: Множество user_id когорты.
    :param anime_ids: Множество anime_id каталога.
    :return: Число событий.
    """
    sources = [OLD / "users_history.csv", EXPANDED / "users_history.csv"]
    columns = ["event_id", "user_id", "event_at", "description", "anime_id"]
    out_handle, table = writer(OUTPUT / "user_anime_history.csv", columns)
    event_ids = set()
    count = 0
    with out_handle:
        for source in sources:
            for row in rows(source):
                event_id = row["id"]
                if not event_id or event_id in event_ids:
                    raise ValueError(f"Missing or repeated event ID: {event_id}")
                if row["user_id"] not in users or row["anime_id"] not in anime_ids:
                    raise ValueError(f"Unknown user or anime in history: {event_id}")
                event_ids.add(event_id)
                table.writerow({
                    "event_id": event_id,
                    "user_id": row["user_id"],
                    "event_at": row["created_at"],
                    "description": row["description"],
                    "anime_id": row["anime_id"],
                })
                count += 1
    return count


def build_genres() -> tuple[int, set[str]]:
    """Строит genres.csv: переименовывает id в genre_id и проверяет его уникальность.

    :return: (число жанров, множество genre_id).
    """
    source = CATALOG / "genres.csv"
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = ["genre_id" if name == "id" else name for name in reader.fieldnames]
        out_handle, table = writer(OUTPUT / "genres.csv", columns)
        genre_ids = set()
        with out_handle:
            for row in reader:
                genre_id = row.pop("id")
                if not genre_id or genre_id in genre_ids:
                    raise ValueError(f"Missing or repeated genre ID: {genre_id}")
                genre_ids.add(genre_id)
                table.writerow({"genre_id": genre_id, **row})
    return len(genre_ids), genre_ids


def build_studios(anime_ids: set[str]) -> tuple[int, int]:
    """Строит anime_studios.csv (связь) и studios.csv (справочник).

    Справочник собирается из уникальных пар (studio_id, studio_name); для одного
    studio_id допустимо ровно одно название.

    :param anime_ids: Множество anime_id каталога.
    :return: (число студий, число связей аниме–студия).
    """
    names = {}
    count = 0
    out_handle, table = writer(OUTPUT / "anime_studios.csv", ["anime_id", "studio_id"])
    with out_handle:
        for row in rows(CATALOG / "anime_studios.csv"):
            anime_id, studio_id, name = row["anime_id"], row["studio_id"], row["studio_name"]
            if anime_id not in anime_ids or not studio_id:
                raise ValueError(f"Unknown anime or missing studio ID: {anime_id}, {studio_id}")
            if studio_id in names and names[studio_id] != name:
                raise ValueError(f"Different names for studio ID {studio_id}")
            names[studio_id] = name
            table.writerow({"anime_id": anime_id, "studio_id": studio_id})
            count += 1
    out_handle, table = writer(OUTPUT / "studios.csv", ["studio_id", "studio_name"])
    with out_handle:
        for studio_id in sorted(names, key=int):
            table.writerow({"studio_id": studio_id, "studio_name": names[studio_id]})
    return len(names), count


def build_chronology(franchises: dict[str, str]) -> int:
    """Строит anime_chronology.csv.

    Поле franchise не дублируется, но проверяется на совпадение с
    anime.franchise у seed_anime_id.

    :param franchises: {anime_id: franchise} из каталога.
    :return: Число строк.
    """
    out_handle, table = writer(
        OUTPUT / "anime_chronology.csv", ["seed_anime_id", "position", "anime_id"]
    )
    count = 0
    with out_handle:
        for row in rows(CATALOG / "anime_chronology.csv"):
            seed = row["seed_anime_id"]
            if seed not in franchises or row["franchise"] != franchises[seed]:
                raise ValueError(f"Chronology franchise differs from anime: {seed}")
            table.writerow({key: row[key] for key in table.fieldnames})
            count += 1
    return count


def count_rows(path: Path) -> int:
    """Считает строки CSV без заголовка.

    :param path: Путь к CSV-файлу.
    """
    return sum(1 for _ in rows(path))


def sha256(path: Path) -> str:
    """Считает SHA-256 файла, читая его блоками по 1 МБ.

    :param path: Путь к файлу.
    :return: Шестнадцатеричная контрольная сумма.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Собирает все таблицы shikimori_v1 и manifest.json.

    Не перезаписывает существующий OUTPUT: при его наличии бросает FileExistsError.
    """
    if OUTPUT.exists():
        raise FileExistsError(f"Output already exists; no files were overwritten: {OUTPUT}")
    OUTPUT.mkdir(parents=True)

    user_count = build_users()
    users = {row["user_id"] for row in rows(OUTPUT / "users.csv")}
    anime_count, anime_ids, franchises = build_anime()
    genre_count, genre_ids = build_genres()
    counts = {
        "users.csv": user_count,
        "anime.csv": anime_count,
        "genres.csv": genre_count,
        "user_anime_rates.csv": build_rates(users, anime_ids),
        "user_anime_history.csv": build_history(users, anime_ids),
    }

    copied(EXPANDED / "user_favourites.csv", "user_favourites.csv")
    counts["user_favourites.csv"] = count_rows(OUTPUT / "user_favourites.csv")
    for row in rows(OUTPUT / "user_favourites.csv"):
        if row["user_id"] not in users:
            raise ValueError(f"Unknown user in favourites: {row['user_id']}")
        if row["category"] == "animes" and row["target_id"] not in anime_ids:
            raise ValueError(f"Unknown favourite anime: {row['target_id']}")

    copied(CATALOG / "anime_genres.csv", "anime_genres.csv")
    counts["anime_genres.csv"] = 0
    for row in rows(OUTPUT / "anime_genres.csv"):
        if row["anime_id"] not in anime_ids or row["genre_id"] not in genre_ids:
            raise ValueError(f"Unknown anime/genre link: {row}")
        counts["anime_genres.csv"] += 1

    studio_count, link_count = build_studios(anime_ids)
    counts["studios.csv"] = studio_count
    counts["anime_studios.csv"] = link_count

    copied(CATALOG / "anime_relations.csv", "anime_relations.csv")
    counts["anime_relations.csv"] = 0
    for row in rows(OUTPUT / "anime_relations.csv"):
        if row["anime_id"] not in anime_ids:
            raise ValueError(f"Unknown source anime in relation: {row}")
        counts["anime_relations.csv"] += 1

    counts["anime_chronology.csv"] = build_chronology(franchises)

    sources = [
        EXPANDED / "cohort_manifest.json", EXPANDED / "rates_access.csv",
        EXPANDED / "favourites_access.csv", EXPANDED / "history_access.csv",
        OLD / "history_access.csv", EXPANDED / "users_rates.csv",
        OLD / "users_history.csv", EXPANDED / "users_history.csv",
        EXPANDED / "user_favourites.csv", CATALOG / "anime.csv",
        CATALOG / "genres.csv", CATALOG / "anime_genres.csv",
        CATALOG / "anime_studios.csv", CATALOG / "anime_relations.csv",
        CATALOG / "anime_chronology.csv",
    ]
    manifest = {
        "schema_version": 1,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Structured source facts for audit; no imputation, label creation or deduplication",
        "tables": counts,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "notes": [
            "Original users_rates.csv is an older overlapping snapshot and is not appended.",
            "History files cover disjoint user cohorts and are concatenated.",
            "All original raw files remain unchanged.",
            "This internal version contains site user IDs and user text; do not publish it directly.",
        ],
    }
    with (OUTPUT / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    print(f"Built {OUTPUT}")


if __name__ == "__main__":
    main()
