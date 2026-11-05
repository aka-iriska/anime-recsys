from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core import logs
from core.catalog import franchise
from core.settings import EVALUATION, RERANK
from serving.rerank import allowed_by_prequel, rerank
from training.evaluation import protocol, stats
from training.interactions import store
from training.ranking import model as rk
from training.retrieval import popularity

# подробности по шагам; включить: core.logs.enable("DEBUG")
logger = logging.getLogger(__name__)

K = RERANK.feed_length  # длина ленты
FRANCHISE_LIMITS = (None, 3, 2, 1)  # сколько аниме одной франшизы разрешаем в ленте
POPULARITY_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)  # сила штрафа за популярность


@dataclass(frozen=True)
class RerankContext:
    """Всё, что нужно правилам ленты и их оценке на val.

    :param table: Таблица взаимодействий из config.INTERACTIONS.
    :param catalog: anime_id каталога.
    :param completed: {user_id: досмотренное из input} – по нему проверяется правило о приквелах.
    :param prequels: anime_id → прямые приквелы из каталога.
    :param franchise_of: anime_id → франшиза.
    :param cases: Спрятанные ответы val: {user_id: (seen, targets)}.
    """

    table: pd.DataFrame
    catalog: np.ndarray
    completed: dict[int, set[int]]
    prequels: dict[int, set[int]]
    franchise_of: pd.Series
    cases: dict[int, tuple[set[int], set[int]]]


@dataclass(frozen=True)
class Criterion:
    """Пороги критерия итогового списка (E01): точность и покрытие относительно популярности.

    :param ndcg_min: Наименьший допустимый ndcg@10.
    :param coverage_min: Наименьшее допустимое coverage@10.
    """

    ndcg_min: float
    coverage_min: float

    def met(self, ndcg: float, coverage: float) -> bool:
        """Выполнен ли критерий для ленты с такими точностью и покрытием.

        :param ndcg: Средний ndcg@10.
        :param coverage: coverage@10.
        :return: True, если оба порога пройдены.
        """
        accurate = ndcg >= self.ndcg_min
        diverse = coverage >= self.coverage_min
        return accurate and diverse


@dataclass(frozen=True)
class FeedSummary:
    """Свойства выдачи в целом (не по отдельному пользователю).

    :param coverage: Доля каталога, попавшая хотя бы в одну ленту.
    :param franchise_repeats: Доля мест ленты, занятых повтором франшизы.
    :param max_franchise: Среднее по пользователям наибольшее число аниме одной франшизы в ленте.
    :param no_prequel: Доля мест с продолжением, приквел которого пользователь не досмотрел.
    """

    coverage: float
    franchise_repeats: float
    max_franchise: float
    no_prequel: float

    def as_row(self) -> dict[str, float]:
        """Поля для строки таблицы; покрытие подписано длиной ленты, как в остальных таблицах.

        :return: {"coverage@10", "franchise_repeats", "max_franchise", "no_prequel"}.
        """
        return {f"coverage@{K}": self.coverage, "franchise_repeats": self.franchise_repeats,
                "max_franchise": self.max_franchise, "no_prequel": self.no_prequel}


def load_context(group: str = "val") -> RerankContext:
    """Всё, что нужно правилам и оценке: история, связи, франшизы, ответы.

    :param group: "val" (по умолчанию) или "test" – только для финальной проверки.
    :return: RerankContext.
    """
    table, catalog = store.load_interactions()
    # что val-пользователь досмотрел в input: по этому правило «сначала приквел» проверяет приквелы
    completed_ids = protocol.completed_input(table, group)
    completed = {}
    for user_id, anime_ids in completed_ids.items():
        completed[user_id] = set(anime_ids.tolist())
    # приквелы только из каталога: досмотреть аниме вне каталога в наших данных нельзя
    catalog_ids = set(catalog.tolist())
    prequels = franchise.load_prequels(catalog_ids)
    franchise_of = franchise.load_franchise_names()
    # спрятанные ответы – из уже прочитанной таблицы
    cases = protocol.build_cases(table, group)
    return RerankContext(table=table, catalog=catalog, completed=completed, prequels=prequels,
                         franchise_of=franchise_of, cases=cases)


def criterion_thresholds(context: RerankContext) -> Criterion:
    """Пороги критерия итогового списка от популярности на тех же val-пользователях.

    :param context: Результат load_context.
    :return: Criterion: доля EVALUATION.accuracy_share от ndcg@10 популярности и
        EVALUATION.coverage_factor × её coverage@10.
    """
    # метрики популярности на тех же пользователях
    scores = popularity.popularity_scores(context.table)
    ranker = popularity.make_popularity_ranker(scores)
    baseline = protocol.evaluate(ranker, context.cases, context.catalog)
    # пороги относительно популярности
    ndcg_min = EVALUATION.accuracy_share * baseline[f"ndcg@{K}"]
    coverage_min = EVALUATION.coverage_factor * baseline[f"coverage@{K}"]
    return Criterion(ndcg_min=ndcg_min, coverage_min=coverage_min)


def score_val(data: pd.DataFrame, columns: list[str], params: dict | None = None) -> pd.DataFrame:
    """Val-кандидаты с баллом CatBoost, обученного на всех train-кандидатах.

    :param data: Результат rk.prepare.
    :param columns: Колонки-признаки (все группы).
    :param params: Параметры CatBoost; None – по умолчанию.
    :return: Строки val (user_id, anime_id, popularity, label, ...) с колонкой score.
    """
    # обучение на train-кандидатах
    train = data[data["group"] == "train"]
    train_pool = rk.make_pool(train, columns)
    model = rk.train_ranker(train_pool, params)
    # балл модели для каждого val-кандидата – тем же шагом, что в оценке ranking
    return rk.score_split(model, data, columns, "val")


def build_feeds(scored: pd.DataFrame, context: RerankContext, max_per_franchise: int | None = None,
                popularity_weight: float = 0.0, require_prequel: bool = True) -> dict[int, np.ndarray]:
    """Ленты всех val-пользователей по одним и тем же правилам.

    По умолчанию правила, кроме «сначала приквел», выключены: так сетка сравнивает их явно.

    :param scored: Результат score_val.
    :param context: Результат load_context.
    :param max_per_franchise: Лимит аниме одной франшизы; None – без лимита.
    :param popularity_weight: Сила штрафа за популярность.
    :param require_prequel: Применять ли правило «сначала приквел».
    :return: {user_id: anime_id ленты по порядку}.
    """
    feeds = {}
    for user_id, rows in scored.groupby("user_id"):
        # кандидаты пользователя, их баллы и популярность
        anime_ids = rows["anime_id"].to_numpy()
        scores = rows["score"].to_numpy()
        item_popularity = rows["popularity"].to_numpy()
        completed = context.completed.get(user_id, set())
        # лента по правилам
        feeds[int(user_id)] = rerank(
            anime_ids, scores, item_popularity, context.franchise_of, completed, context.prequels,
            k=K, max_per_franchise=max_per_franchise, popularity_weight=popularity_weight,
            require_prequel=require_prequel,
        )
    return feeds


def accuracy_per_user(feeds: dict[int, np.ndarray], cases: dict) -> pd.DataFrame:
    """Точность каждой ленты по спрятанным ответам пользователя.

    :param feeds: Результат build_feeds.
    :param cases: Результат protocol.build_cases.
    :return: DataFrame с индексом user_id: recall@K, hit_rate@K, ndcg@K.
    """
    per_user = protocol.per_user_metrics(feeds, cases, (K,))
    return per_user.sort_index()


def coverage(feeds: dict[int, np.ndarray], catalog: np.ndarray) -> float:
    """Доля каталога, попавшая хотя бы в одну ленту.

    :param feeds: Результат build_feeds.
    :param catalog: anime_id каталога.
    :return: coverage@K.
    """
    shown = set()
    for feed in feeds.values():
        shown.update(feed.tolist())
    return len(shown) / len(catalog)


def franchise_stats(feeds: dict[int, np.ndarray], franchise_of: pd.Series) -> tuple[float, float]:
    """Насколько ленты забиты одной франшизой.

    :param feeds: Результат build_feeds.
    :param franchise_of: anime_id → франшиза.
    :return: (доля мест ленты, занятых повтором франшизы; среднее по пользователям наибольшее
        число аниме одной франшизы в ленте).
    """
    repeats = 0
    slots = 0
    max_franchise = []
    for feed in feeds.values():
        names = franchise_of.reindex(feed)
        # повторы: мест в ленте минус число разных франшиз
        repeats += len(feed) - names.nunique()
        slots += len(feed)
        # сколько аниме у самой частой франшизы ленты
        counts = names.value_counts()
        max_franchise.append(counts.max())
    return repeats / slots, float(np.mean(max_franchise))


def prequel_violations(feeds: dict[int, np.ndarray], completed: dict[int, set[int]],
                       prequels: dict[int, set[int]]) -> float:
    """Доля мест ленты с продолжением, приквел которого пользователь не досмотрел.

    :param feeds: Результат build_feeds.
    :param completed: {user_id: досмотренное из input}.
    :param prequels: anime_id → прямые приквелы.
    :return: Доля таких мест среди всех мест всех лент.
    """
    violations = 0
    slots = 0
    for user_id, feed in feeds.items():
        user_completed = completed.get(user_id, set())
        allowed = allowed_by_prequel(feed, user_completed, prequels)
        violations += int((~allowed).sum())
        slots += len(feed)
    return violations / slots


def feed_metrics(feeds: dict[int, np.ndarray], context: RerankContext) -> tuple[pd.DataFrame, FeedSummary]:
    """Точность лент по каждому пользователю и свойства выдачи в целом.

    :param feeds: Результат build_feeds.
    :param context: Результат load_context.
    :return: (per_user, summary): per_user – accuracy_per_user; summary – FeedSummary.
    """
    per_user = accuracy_per_user(feeds, context.cases)
    # свойства выдачи по отдельности
    shown_share = coverage(feeds, context.catalog)
    repeats, max_franchise = franchise_stats(feeds, context.franchise_of)
    violations = prequel_violations(feeds, context.completed, context.prequels)
    summary = FeedSummary(coverage=shown_share, franchise_repeats=repeats,
                          max_franchise=max_franchise, no_prequel=violations)
    return per_user, summary


def grid_variants(limits, weights) -> list[tuple[bool, int | None, float]]:
    """Варианты сетки: CatBoost без правил, только «сначала приквел», затем лимит × штраф.

    :param limits: Значения лимита франшизы.
    :param weights: Значения силы штрафа за популярность.
    :return: Список (require_prequel, max_per_franchise, popularity_weight).
    """
    variants = [(False, None, 0.0), (True, None, 0.0)]
    for limit in limits:
        for weight in weights:
            # «только приквел» уже есть второй строкой
            if (limit, weight) != (None, 0.0):
                variants.append((True, limit, weight))
    return variants


def sweep(scored: pd.DataFrame, context: RerankContext, criterion: Criterion,
          limits=FRANCHISE_LIMITS, weights=POPULARITY_WEIGHTS) -> pd.DataFrame:
    """Сравнение правил: лимит франшизы × штраф за популярность против порядка CatBoost.

    Первая строка – порядок CatBoost без правил, вторая – только «сначала приквел»; дальше
    сетка, во всех её точках правило «сначала приквел» включено.

    :param scored: Результат score_val.
    :param context: Результат load_context.
    :param criterion: Результат criterion_thresholds.
    :param limits: Значения лимита франшизы.
    :param weights: Значения силы штрафа за популярность.
    :return: DataFrame: строка – вариант правил, колонки – ndcg@10, парная разница с CatBoost
        и её интервал, свойства выдачи (FeedSummary) и выполнение критерия.
    """
    rows = []
    reference = None
    for require_prequel, limit, weight in grid_variants(limits, weights):
        # ленты по правилам и их метрики
        feeds = build_feeds(scored, context, limit, weight, require_prequel)
        per_user, summary = feed_metrics(feeds, context)
        ndcg = per_user[f"ndcg@{K}"]
        # порядок CatBoost без правил (первый вариант) – опорная линия для парных разниц
        if reference is None:
            reference = ndcg
        compared = stats.compare(ndcg, reference)
        row = {
            "сначала приквел": "да" if require_prequel else "нет",
            "лимит франшизы": "нет" if limit is None else limit,
            "штраф за популярность": weight,
            f"ndcg@{K}": compared.value.mean,
            "diff": compared.diff.mean,
            "diff_low": compared.diff.low,
            "diff_high": compared.diff.high,
            **summary.as_row(),
            "критерий": criterion.met(compared.value.mean, summary.coverage),
        }
        rows.append(row)
    result = pd.DataFrame(rows)
    logs.table(logger, "сетка правил re-ranking", result, rows=len(result))
    return result


def main() -> None:
    """Обучает CatBoost, проходит сетку правил на val и печатает таблицу."""
    data, columns = rk.prepare()
    context = load_context()
    criterion = criterion_thresholds(context)
    scored = score_val(data, columns)
    table = sweep(scored, context, criterion)
    rounded = table.round(4)
    print(f"критерий: ndcg@{K} >= {criterion.ndcg_min:.3f}, coverage@{K} >= {criterion.coverage_min:.3f}")
    print(rounded.to_string(index=False))


if __name__ == "__main__":
    # подробные логи по шагам; уровень – переменная окружения RECSYS_LOG_LEVEL (по умолчанию INFO, таблицы – DEBUG)
    logs.enable()
    main()
