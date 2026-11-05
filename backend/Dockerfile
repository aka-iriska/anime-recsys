# Бэкенд ленты: только пакет serving (+ core). Кода и библиотек обучения в образе нет.
# Контекст сборки – папка backend/: docker compose build backend (из корня репозитория)

# 1. сборка окружения: uv ставит serving и его зависимости по lock-файлу
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# описания workspace нужны uv целиком, а исходники – только core и serving
COPY pyproject.toml uv.lock ./
COPY core/pyproject.toml core/pyproject.toml
COPY serving/pyproject.toml serving/pyproject.toml
COPY training/pyproject.toml training/pyproject.toml
RUN uv sync --frozen --package serving --no-dev --no-editable --no-install-workspace
COPY core core
COPY serving serving
RUN uv sync --frozen --package serving --no-dev --no-editable

# 2. итоговый образ: только готовое окружение, без uv и без исходников
FROM python:3.12-slim-bookworm
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    RECSYS_ROOT=/app \
    RECSYS_MODELS=/app/models \
    PYTHONUNBUFFERED=1

# артефакты модели монтируются томом (их создаёт: uv run python -m training.pipelines.export_models)
EXPOSE 8000
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
