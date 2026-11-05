# Частые команды проекта. Запуск из корня репозитория: make <цель>; список целей – make help.
# Python-команды идут через uv-проект backend/ (uv run --project backend), поэтому cd не нужен.

UV := uv run --project backend
TRAIN := $(UV) python -m training
PARSER ?= ../../shikimori-parse  # репозиторий shikimori-parse (для постеров)

.DEFAULT_GOAL := help
.PHONY: help install data ranking-data retrieval ranking rerank final-test simulate export posters \
        test lint api web up down

help: ## список целей
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-14s %s\n", $$1, $$2}'

# ---------- окружение
install: ## окружение backend/.venv: пакеты core, serving, training и инструменты ноутбуков
	uv sync --project backend

# ---------- данные (результат – backend/data/processed)
data: ## сырые выгрузки -> таблицы shikimori_v1 -> протокол interactions_v2 (каталог, разбиение)
	$(TRAIN).ingestion.build_dataset
	$(TRAIN).interactions.prepare

ranking-data: ## датасет для CatBoost ranking_v2: 100 кандидатов iALS на пользователя, 5 частей
	$(TRAIN).ranking.dataset

# ---------- эксперименты на val (E01–E08)
retrieval: ## сравнение retrieval: популярность, item-kNN, PureSVD, iALS (recall@100)
	$(TRAIN).retrieval.als

ranking: ## CatBoost на кандидатах iALS против порядка iALS (ndcg@10)
	$(TRAIN).ranking.model

rerank: ## правила ленты: сетка штрафа за популярность и лимита франшизы
	$(TRAIN).reranking.evaluate

# ---------- итоговые проверки
final-test: ## один раз: все стадии на test-пользователях (E09)
	$(TRAIN).pipelines.final_test

simulate: ## симуляция обратной связи и A/B против популярности (E10), ~4 минуты
	$(TRAIN).simulation.feedback

# ---------- выдача
export: ## итоговые iALS и CatBoost и данные правил -> backend/models
	$(TRAIN).pipelines.export_models

posters: ## ссылки на постеры каталога -> backend/data/posters (нужен PARSER=путь к shikimori-parse)
	cd $(strip $(PARSER))/examples && uv run python parse_posters.py --ids "$(CURDIR)/backend/data/processed/interactions_v2/catalog.csv" --output "$(CURDIR)/backend/data/posters"

api: ## API на http://localhost:8000 с перезапуском при правках кода
	$(UV) uvicorn serving.api:app --port 8000 --reload

web: ## страница ленты на http://localhost:3000 (API должен быть запущен: make api)
	$(UV) python -m http.server 3000 --directory frontend

up: ## API и страница в Docker: http://localhost:8000 и http://localhost:3000
	docker compose up --build

down: ## остановить контейнеры
	docker compose down

# ---------- проверки кода
test: ## тесты serving: правила ленты и API на артефактах backend/models
	$(UV) pytest backend -q

lint: ## неиспользуемые импорты и имена (pyflakes)
	uvx pyflakes backend/core backend/serving backend/training
