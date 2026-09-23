from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "kdduha/shikimori-recsys"
REVISION = "033630726d82cd9d99817a735dced800984c3aa8"
FILES = ("anime.csv", "genres.csv", "users_rates.csv")


def download_dataset(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        local_dir=output_dir,
        allow_patterns=list(FILES),
    )

    paths = {name: output_dir / name for name in FILES}
    missing = [name for name, path in paths.items() if not path.is_file()]

    if missing:
        raise RuntimeError(f"Не скачались файлы: {missing}")

    return paths


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    downloaded = download_dataset(project_root / "data" / "raw")

    for name, path in downloaded.items():
        print(f"{name}: {path.stat().st_size:,} bytes")