from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core import logs
from serving.directory import AnimeCard
from serving.recommender import Recommender
from serving.settings import ServerSettings

settings = ServerSettings()


class FeedRequest(BaseModel):
    """Запрос ленты: что пользователь посмотрел на Shikimori.

    :param history: anime_id досмотренного (completed): по ним строятся вектор и признаки.
    :param exclude: anime_id, которые нельзя показывать (смотрю, в планах, брошено, отложено).
    :param favourites: anime_id избранного (учитывается то, что есть в history).
    :param k: Длина ленты; не задана – принятая длина из настроек правил.
    """

    history: list[int] = Field(default_factory=list, max_length=5000)
    exclude: list[int] = Field(default_factory=list, max_length=5000)
    favourites: list[int] = Field(default_factory=list, max_length=500)
    k: int | None = Field(default=None, ge=1, le=100)


class AnimeOut(BaseModel):
    """Аниме для показа: название, постер, жанры.

    :param anime_id: Идентификатор аниме на Shikimori.
    :param title: Название для показа (русское, иначе латиницей).
    :param name: Название латиницей.
    :param russian: Русское название, если есть.
    :param kind: Тип: tv, movie, ova, ...
    :param year: Год начала показа.
    :param episodes: Число серий.
    :param genres: Жанры через запятую.
    :param poster_url: Маленький постер.
    :param poster_main_url: Крупный постер.
    :param url: Страница аниме на Shikimori.
    """

    anime_id: int
    title: str
    name: str
    russian: str | None
    kind: str | None
    year: int | None
    episodes: int | None
    genres: str | None
    poster_url: str | None
    poster_main_url: str | None
    url: str


class FeedItemOut(BaseModel):
    """Позиция ленты.

    :param anime: Аниме.
    :param score: Балл модели (у популярной ленты – популярность).
    :param reason: Почему аниме в ленте.
    :param reason_anime_id: Аниме из истории, на которое ссылается причина.
    """

    anime: AnimeOut
    score: float
    reason: str
    reason_anime_id: int | None


class FeedResponse(BaseModel):
    """Лента.

    :param items: Позиции по порядку показа.
    :param strategy: "personal" (iALS → CatBoost → правила) или "popular" (истории в каталоге нет).
    :param history_used: Сколько аниме истории нашлось в каталоге.
    """

    items: list[FeedItemOut]
    strategy: str
    history_used: int


def anime_out(card: AnimeCard) -> AnimeOut:
    """Карточка аниме в формате ответа API.

    :param card: Карточка из справочника.
    :return: AnimeOut со ссылкой на страницу Shikimori.
    """
    url = f"https://shikimori.io/animes/{card.anime_id}"
    return AnimeOut(anime_id=card.anime_id, title=card.title, name=card.name, russian=card.russian,
                    kind=card.kind, year=card.year, episodes=card.episodes, genres=card.genres,
                    poster_url=card.poster_url, poster_main_url=card.poster_main_url, url=url)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружает артефакты моделей один раз при старте сервиса.

    :param app: Приложение FastAPI.
    """
    logs.enable()
    app.state.recommender = Recommender.load()
    yield


app = FastAPI(title="Shikimori anime recommender", lifespan=lifespan)
# фронтенд открыт на другом адресе (порт 3000), поэтому браузеру нужно разрешение CORS
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/health")
def health() -> dict[str, str]:
    """Проверка, что сервис жив и модели загружены.

    :return: {"status": "ok"}.
    """
    return {"status": "ok"}


@app.get("/info")
def info() -> dict:
    """Версии моделей и данных: manifest.json каждой папки артефактов и настройки ленты.

    :return: {"artifacts": {папка: manifest}, "rerank": настройки правил}.
    """
    recommender = app.state.recommender
    return {"artifacts": recommender.manifests, "rerank": recommender.rules.settings}


@app.post("/feed")
def feed(request: FeedRequest) -> FeedResponse:
    """Лента по истории пользователя: iALS → CatBoost → правила ленты, с причинами показа.

    :param request: История, исключения, избранное и длина ленты.
    :return: FeedResponse.
    """
    result = app.state.recommender.feed(request.history, exclude=request.exclude,
                                        favourites=request.favourites, k=request.k)
    items = []
    for item in result.items:
        out = FeedItemOut(anime=anime_out(item.card), score=item.score, reason=item.reason,
                          reason_anime_id=item.reason_anime_id)
        items.append(out)
    return FeedResponse(items=items, strategy=result.strategy, history_used=result.history_used)


# /anime/search объявлен раньше /anime/{anime_id}, иначе "search" разбирался бы как id
@app.get("/anime/search")
def search(q: str = Query(min_length=1, max_length=100), limit: int = Query(default=10, ge=1)) -> list[AnimeOut]:
    """Поиск аниме каталога по названию – чтобы собрать историю на фронтенде.

    :param q: Часть названия латиницей или по-русски.
    :param limit: Сколько результатов вернуть (не больше ServerSettings.search_limit).
    :return: Карточки: сначала названия, начинающиеся с запроса, затем по популярности.
    """
    limit = min(limit, settings.search_limit)
    cards = app.state.recommender.directory.search(q, limit)
    return [anime_out(card) for card in cards]


@app.get("/anime/{anime_id}")
def anime(anime_id: int) -> AnimeOut:
    """Карточка аниме каталога.

    :param anime_id: Идентификатор аниме на Shikimori.
    :return: AnimeOut; 404, если аниме нет в каталоге.
    """
    card = app.state.recommender.directory.card(anime_id)
    if card is None:
        raise HTTPException(status_code=404, detail=f"Аниме {anime_id} нет в каталоге рекомендаций")
    return anime_out(card)
