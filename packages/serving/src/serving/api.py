from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from serving.recommender import Recommender


class FeedRequest(BaseModel):
    """Запрос ленты.

    :param history: anime_id досмотренных аниме: по ним строится вектор пользователя.
    :param exclude: anime_id, которые нельзя показывать (брошенные, в планах).
    :param k: Длина ленты.
    """

    history: list[int] = Field(default_factory=list)
    exclude: list[int] = Field(default_factory=list)
    k: int = Field(default=10, ge=1, le=100)


class FeedItem(BaseModel):
    """Одна позиция ленты.

    :param anime_id: Идентификатор аниме на Shikimori.
    :param name: Название латиницей.
    :param russian: Русское название, если есть.
    :param score: Балл модели.
    """

    anime_id: int
    name: str
    russian: str | None
    score: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружает артефакты модели один раз при старте сервиса.

    :param app: Приложение FastAPI.
    """
    app.state.recommender = Recommender.load()
    yield


app = FastAPI(title="Shikimori anime recommender", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])  # фронтенд на другом порту


@app.get("/health")
def health() -> dict[str, str]:
    """Проверка, что сервис жив и модель загружена.

    :return: {"status": "ok"}.
    """
    return {"status": "ok"}


@app.post("/feed")
def feed(request: FeedRequest) -> list[FeedItem]:
    """Лента для пользователя по его истории.

    :param request: История, исключения и длина ленты.
    :return: Позиции ленты по убыванию балла.
    """
    items = app.state.recommender.recommend(request.history, k=request.k, exclude=request.exclude)
    return [FeedItem(anime_id=i.anime_id, name=i.name, russian=i.russian, score=i.score) for i in items]
