from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    """Настройки сервера выдачи из переменных окружения с префиксом RECSYS_.

    Пути к данным и моделям – в core.config (RECSYS_ROOT, RECSYS_MODELS), уровень логов –
    RECSYS_LOG_LEVEL (core.logs.enable), решения о модели –
    в артефактах (params.json, settings.json); здесь только то, что касается самого сервера.

    :param cors_origins: Адреса фронтенда, которым можно обращаться к API
        (RECSYS_CORS_ORIGINS='["http://localhost:3000"]'); "*" – любым.
    :param search_limit: Наибольшее число результатов поиска.
    """

    model_config = SettingsConfigDict(env_prefix="RECSYS_")

    cors_origins: list[str] = ["*"]
    search_limit: int = 20
