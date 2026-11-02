from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]  # корень проекта: src/shikimori_rec/config.py -> ../..
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
SOURCE = PROCESSED / "shikimori_v1"  # нормализованные таблицы Shikimori
INTERACTIONS = PROCESSED / "interactions_v0"  # протокол оценки: каталог и разбиение пользователей
RANKING_DATA = PROCESSED / "ranking_v0"  # кандидаты для обучения ранжирования
MODELS = ROOT / "models"  # артефакты итоговых моделей для выдачи

SEED = 42
