from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AnimeCard:
    """Сведения об аниме для ленты, карточки и поиска.

    :param anime_id: Идентификатор аниме на Shikimori.
    :param name: Название латиницей.
    :param russian: Русское название, если есть.
    :param kind: Тип: tv, movie, ova, ...
    :param year: Год начала показа, если известен.
    :param episodes: Число серий, если известно.
    :param genres: Жанры через запятую.
    :param poster_url: Маленький постер (для ленты), если есть.
    :param poster_main_url: Крупный постер (для карточки), если есть.
    """

    anime_id: int
    name: str
    russian: str | None
    kind: str | None
    year: int | None
    episodes: int | None
    genres: str | None
    poster_url: str | None
    poster_main_url: str | None

    @property
    def title(self) -> str:
        """Название для показа: русское, если есть, иначе латиницей."""
        return self.russian or self.name


def text_or_none(value) -> str | None:
    """Значение ячейки как строка; пропуск (NaN) – None.

    :param value: Значение из DataFrame.
    :return: Строка или None.
    """
    if pd.isna(value):
        return None
    return str(value)


def int_or_none(value) -> int | None:
    """Значение ячейки как целое; пропуск и 0 («нет данных» в источнике) – None.

    :param value: Значение из DataFrame.
    :return: Целое или None.
    """
    if pd.isna(value) or int(value) == 0:
        return None
    return int(value)


class AnimeDirectory:
    """Справочник аниме каталога: карточка по id и поиск по названию.

    Поиск – подстрока в латинском или русском названии без учёта регистра; первыми идут
    названия, которые начинаются с запроса, затем – более популярные.
    """

    def __init__(self, anime: pd.DataFrame, popularity: pd.Series):
        """Строит карточки и строки для поиска.

        :param anime: Справочник из models/als_v0/anime.csv (anime_id, name, russian, kind,
            aired_on, episodes, genres, poster_url, poster_main_url).
        :param popularity: anime_id → популярность (для порядка результатов поиска).
        """
        self.cards = {}
        for row in anime.itertuples(index=False):
            card = self.make_card(row)
            self.cards[card.anime_id] = card
        # строки поиска: оба названия в нижнем регистре
        self.ids = anime["anime_id"].to_numpy()
        self.names = anime["name"].fillna("").str.lower().reset_index(drop=True)
        self.russian = anime["russian"].fillna("").str.lower().reset_index(drop=True)
        self.popularity = popularity.reindex(self.ids, fill_value=0.0).to_numpy()

    @staticmethod
    def make_card(row) -> AnimeCard:
        """Карточка из строки справочника.

        :param row: Строка anime.itertuples() с полями справочника.
        :return: AnimeCard.
        """
        aired_on = text_or_none(getattr(row, "aired_on", None))
        year = int(aired_on[:4]) if aired_on and aired_on[:4].isdigit() else None
        return AnimeCard(
            anime_id=int(row.anime_id),
            name=str(row.name),
            russian=text_or_none(row.russian),
            kind=text_or_none(getattr(row, "kind", None)),
            year=year,
            episodes=int_or_none(getattr(row, "episodes", None)),
            genres=text_or_none(getattr(row, "genres", None)),
            poster_url=text_or_none(getattr(row, "poster_url", None)),
            poster_main_url=text_or_none(getattr(row, "poster_main_url", None)),
        )

    def card(self, anime_id: int) -> AnimeCard | None:
        """Карточка аниме каталога.

        :param anime_id: Идентификатор аниме.
        :return: AnimeCard или None, если аниме нет в каталоге.
        """
        return self.cards.get(int(anime_id))

    def search(self, query: str, limit: int = 20) -> list[AnimeCard]:
        """Аниме каталога, в названии которых есть запрос.

        :param query: Часть названия (латиницей или по-русски).
        :param limit: Сколько результатов вернуть.
        :return: Карточки: сначала названия, начинающиеся с запроса, затем по популярности.
        """
        text = query.strip().lower()
        if not text:
            return []
        # запрос где-то в названии и запрос в начале названия
        contains = self.names.str.contains(text, regex=False) | self.russian.str.contains(text, regex=False)
        starts = self.names.str.startswith(text) | self.russian.str.startswith(text)
        found = contains.to_numpy().nonzero()[0]
        # порядок: сначала совпадения с начала, внутри – по убыванию популярности
        order = pd.DataFrame({"starts": starts.to_numpy()[found], "popularity": self.popularity[found]})
        order = order.sort_values(["starts", "popularity"], ascending=False, kind="stable")
        positions = order.index.to_numpy()[:limit]
        top = found[positions]
        return [self.cards[int(self.ids[i])] for i in top]
