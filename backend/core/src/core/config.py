import os
from pathlib import Path


# корень репозитория: backend/core/src/core/config.py; в контейнере пакет установлен в
# site-backend, поэтому корень задаётся переменной окружения RECSYS_ROOT
REPOSITORY = Path(__file__).resolve().parents[4]
ROOT = Path(os.environ.get("RECSYS_ROOT", REPOSITORY))
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
SOURCE = PROCESSED / "shikimori_v1"  # нормализованные таблицы Shikimori
INTERACTIONS = PROCESSED / "interactions_v2"  # протокол оценки: каталог и разбиение пользователей
RANKING_DATA = PROCESSED / "ranking_v2"  # кандидаты для обучения ранжирования
POSTERS = DATA / "posters" / "posters.csv"  # ссылки на постеры (shikimori-parse, examples/parse_posters.py)
MODELS = Path(os.environ.get("RECSYS_MODELS", ROOT / "models"))  # артефакты итоговых моделей для выдачи

SEED = 42
