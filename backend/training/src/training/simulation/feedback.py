from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from core import logs
from core.settings import RERANK, RETRIEVAL
from serving.artifacts import ALS_ARTIFACTS, RANKER_ARTIFACTS, RERANK_ARTIFACTS, load_als, load_ranker, load_rerank
from serving.recommender import Recommender
from training.evaluation import protocol, stats
from training.interactions import store
from training.retrieval import als
from training.retrieval import item_knn as knn

# подробности по шагам; включить: core.logs.enable("DEBUG")
logger = logging.getLogger(__name__)

GROUP = "test"  # пользователи, на которых ничего не подбиралось; параметры симуляции заданы заранее
ROUNDS = 10  # сколько раз пользователь открывает ленту
K = RERANK.feed_length  # длина ленты
PATIENCE = 2  # показали столько раз и не досмотрел – аниме больше не показываем
SEED = 42
ARMS = ("популярность", "система")  # A – популярное с правилами ленты, B – iALS → CatBoost → правила


@dataclass(frozen=True)
class SimUser:
    """Пользователь симуляции: что он уже знает и что досмотрел бы, если увидит.

    :param user_id: Пользователь.
    :param history: Досмотренное из input (аниме каталога) – с этим он приходит в сервис.
    :param seen: Все аниме input (любой статус): их не показываем.
    :param targets: Спрятанные ответы – аниме, которые он на самом деле досмотрел.
    """

    user_id: int
    history: tuple[int, ...]
    seen: frozenset[int]
    targets: frozenset[int]


@dataclass(frozen=True)
class ArmResult:
    """Итог одного варианта ленты в симуляции.

    :param per_user: Индекс user_id; колонки watched (досмотрено из ленты), targets (сколько
        было спрятанных ответов), first_round (раунд первого досмотра, NaN – не было).
    :param per_round: По строке на раунд: watched (среднее накопленное число досмотров на
        пользователя), coverage (доля каталога, показанная хотя бы раз к этому раунду),
        mean_popularity (средняя популярность показанного в раунде).
    :param impressions: anime_id → сколько раз аниме показали за всю симуляцию.
    """

    per_user: pd.DataFrame
    per_round: pd.DataFrame
    impressions: pd.Series


def examination_probability(k: int) -> np.ndarray:
    """Вероятность, что пользователь вообще посмотрит на позицию ленты (модель внимания).

    1 / log2(позиция + 1): первая позиция – 1, десятая – 0,29. То же убывание, что в NDCG.

    :param k: Длина ленты.
    :return: Массив формы (k,).
    """
    positions = np.arange(1, k + 1)
    return 1.0 / np.log2(positions + 1)


def load_users(table: pd.DataFrame, item_ids: np.ndarray, group: str = GROUP) -> list[SimUser]:
    """Пользователи группы со своей историей, просмотренным и спрятанными ответами.

    :param table: Таблица взаимодействий из config.INTERACTIONS.
    :param item_ids: anime_id каталога.
    :param group: "val" или "test".
    :return: Список SimUser в порядке user_id.
    """
    cases = protocol.build_cases(table, group)
    completed = protocol.completed_input(table, group)
    catalog = set(item_ids.tolist())
    users = []
    for user_id, (seen, targets) in sorted(cases.items()):
        own = completed.get(user_id, np.array([], dtype=int))
        history = tuple(int(a) for a in own if int(a) in catalog)
        user = SimUser(user_id=user_id, history=history, seen=frozenset(seen), targets=frozenset(targets))
        users.append(user)
    return users


def simulation_recommender(table: pd.DataFrame, catalog: np.ndarray) -> tuple[Recommender, pd.Series]:
    """Сервис выдачи без утечки: те же классы, что в продукте, но модели не видели пользователей группы.

    Итоговый iALS из models/als_v0 обучен на всех пользователях (в том числе на спрятанных
    ответах test), поэтому здесь iALS переобучается на train-пользователях, как в E09.
    CatBoost из models/ranker_v0 и так обучен только на train-кандидатах. Популярность в
    правилах ленты – тоже по train-пользователям.

    :param table: Таблица взаимодействий из config.INTERACTIONS.
    :param catalog: Массив anime_id каталога.
    :return: (Recommender, число train-зрителей каждого аниме каталога).
    """
    # iALS по train-пользователям
    matrix, _, item_ids = knn.build_train_matrix(table, catalog)
    confidence = als.build_confidence_matrix(matrix)
    model = als.fit_als(confidence, RETRIEVAL.factors)

    # справочник и параметры iALS, CatBoost, правила – из артефактов выдачи
    _, exported_ids, anime, params = load_als(ALS_ARTIFACTS)
    if not np.array_equal(exported_ids, item_ids):
        raise ValueError("Каталог артефактов не совпадает с каталогом протокола: пересоберите export_models")
    ranker = load_ranker(RANKER_ARTIFACTS)
    rules = load_rerank(RERANK_ARTIFACTS)

    # популярность – число train-зрителей (как признак CatBoost), а не всех пользователей
    is_train_completed = (table["split"] == "train") & (table["status"] == "completed")
    viewers = table[is_train_completed].groupby("anime_id")["user_id"].nunique()
    viewers = viewers.reindex(item_ids, fill_value=0)
    rules = replace(rules, popularity=np.log1p(viewers).astype(float))
    recommender = Recommender(model.item_factors, item_ids, anime, params, ranker, rules)
    return recommender, viewers


def show_feed(recommender: Recommender, arm: str, history: list[int], exclude: set[int]) -> list[int]:
    """Лента варианта для пользователя в текущем состоянии.

    :param recommender: Сервис выдачи.
    :param arm: Вариант из ARMS.
    :param history: Досмотренное к этому моменту.
    :param exclude: Что нельзя показывать (input и надоевшее).
    :return: anime_id ленты по порядку.
    """
    if arm == "популярность":
        banned = set(history) | exclude
        feed = recommender.popular_feed(banned, K, completed=history)
    else:
        feed = recommender.feed(history, exclude=exclude, k=K)
    return [item.card.anime_id for item in feed.items]


def run_arm(recommender: Recommender, viewers: pd.Series, users: list[SimUser], arm: str,
            rounds: int = ROUNDS, seed: int = SEED) -> ArmResult:
    """Симуляция одного варианта ленты: раунды «показали ленту – пользователь что-то досмотрел».

    Модель пользователя: позицию i он рассматривает с вероятностью examination_probability;
    рассмотренное аниме досматривает, только если оно из его спрятанных ответов (то, что он
    действительно досмотрел), иначе пропускает. Досмотренное попадает в историю – следующая
    лента строится уже по ней (обратная связь пользователя). Аниме, показанное PATIENCE раз
    без досмотра, больше не показываем. Досмотры раунда добавляются к популярности – так
    популярное становится ещё популярнее (обратная связь системы).

    Случайность одинакова для обоих вариантов (зерно от seed, пользователя и раунда): разница
    вариантов – от лент, а не от удачи броска.

    :param recommender: Результат simulation_recommender (его популярность меняется по ходу).
    :param viewers: Число train-зрителей каждого аниме – начальная популярность.
    :param users: Результат load_users.
    :param arm: Вариант из ARMS.
    :param rounds: Число раундов.
    :param seed: Зерно генератора.
    :return: ArmResult.
    """
    start_rules = recommender.rules
    counts = viewers.copy()
    looked = examination_probability(K)
    catalog_size = len(counts)

    # состояние каждого пользователя
    history = {u.user_id: list(u.history) for u in users}
    remaining = {u.user_id: set(u.targets) for u in users}
    tired = {u.user_id: set() for u in users}
    shown_times = {u.user_id: {} for u in users}
    watched = {u.user_id: 0 for u in users}
    first_round = {}
    impressions = {}
    shown_ever = set()
    rounds_rows = []

    for round_number in range(1, rounds + 1):
        clicks = []
        shown_popularity = []
        for user in users:
            user_id = user.user_id
            # лента по текущей истории; input и надоевшее не показываем
            exclude = set(user.seen) | tired[user_id]
            feed = show_feed(recommender, arm, history[user_id], exclude)
            # какие позиции пользователь рассмотрел
            rng = np.random.default_rng([seed, user_id, round_number])
            draws = rng.random(K)[:len(feed)]
            is_looked = draws < looked[:len(feed)]
            for position, anime_id in enumerate(feed):
                impressions[anime_id] = impressions.get(anime_id, 0) + 1
                shown_ever.add(anime_id)
                shown_popularity.append(float(recommender.rules.popularity.get(anime_id, 0.0)))
                # рассмотрел и это его аниме – досматривает, оно уходит в историю
                if is_looked[position] and anime_id in remaining[user_id]:
                    history[user_id].append(anime_id)
                    remaining[user_id].discard(anime_id)
                    watched[user_id] += 1
                    first_round.setdefault(user_id, round_number)
                    clicks.append(anime_id)
                    continue
                # не досмотрел: после PATIENCE показов аниме надоедает
                times = shown_times[user_id].get(anime_id, 0) + 1
                shown_times[user_id][anime_id] = times
                if times >= PATIENCE:
                    tired[user_id].add(anime_id)

        # обратная связь системы: досмотры раунда увеличивают популярность
        for anime_id in clicks:
            counts[anime_id] += 1
        popularity = np.log1p(counts).astype(float)
        recommender.rules = replace(recommender.rules, popularity=popularity)

        mean_watched = float(np.mean(list(watched.values())))
        row = {"round": round_number, "watched": mean_watched, "coverage": len(shown_ever) / catalog_size,
               "mean_popularity": float(np.mean(shown_popularity))}
        rounds_rows.append(row)
        logger.info("%s, раунд %d: досмотров на пользователя %.3f, покрытие %.3f",
                    arm, round_number, row["watched"], row["coverage"])

    # исходная популярность – для следующего варианта
    recommender.rules = start_rules
    per_user = pd.DataFrame({
        "watched": pd.Series(watched),
        "targets": pd.Series({u.user_id: len(u.targets) for u in users}),
        "first_round": pd.Series(first_round, dtype=float),
    })
    per_user = per_user.rename_axis("user_id").sort_index()
    per_round = pd.DataFrame(rounds_rows).set_index("round")
    impressions_series = pd.Series(impressions, name="impressions").sort_values(ascending=False)
    logs.table(logger, f"{arm}: итог по пользователям", per_user)
    return ArmResult(per_user=per_user, per_round=per_round, impressions=impressions_series)


def gini(values: np.ndarray) -> float:
    """Коэффициент Джини показов: 0 – все показанные аниме показаны поровну, 1 – всё ушло одному.

    :param values: Число показов каждого показанного аниме.
    :return: Число от 0 до 1.
    """
    ordered = np.sort(np.asarray(values, dtype=float))
    n = len(ordered)
    ranks = np.arange(1, n + 1)
    return float((2 * ranks - n - 1) @ ordered / (n * ordered.sum()))


def arm_summary(results: dict[str, ArmResult]) -> pd.DataFrame:
    """Сводка вариантов: досмотры с интервалами и свойства показов.

    :param results: {вариант: ArmResult} на одних и тех же пользователях.
    :return: DataFrame: строка – вариант.
    """
    rows = {}
    for arm, result in results.items():
        watched = result.per_user["watched"]
        estimate = stats.summarize(watched)
        recall = (watched / result.per_user["targets"]).mean()
        rows[arm] = {
            "досмотров на пользователя": estimate.mean,
            "low": estimate.low,
            "high": estimate.high,
            "доля с досмотром": float((watched > 0).mean()),
            "доля найденных ответов": float(recall),
            "покрытие каталога": float(result.per_round["coverage"].iloc[-1]),
            "джини показов": gini(result.impressions.to_numpy()),
            "популярность показанного": float(result.per_round["mean_popularity"].mean()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def ab_test(results: dict[str, ArmResult], seed: int = SEED) -> pd.DataFrame:
    """Симулированный A/B: пользователи случайно делятся пополам, каждая половина видит свой вариант.

    Для сравнения рядом – парная разница на всех пользователях: в симуляции можно показать
    обоим вариантам одних и тех же людей, в настоящем A/B – нельзя.

    :param results: {вариант: ArmResult} из run_arm для ARMS.
    :param seed: Зерно разбиения пользователей.
    :return: DataFrame: строка – способ сравнения, колонки – разница «система − популярность»,
        её интервал и число пользователей в каждой группе.
    """
    control = results[ARMS[0]].per_user["watched"]
    treatment = results[ARMS[1]].per_user["watched"]
    # случайное разбиение пользователей: половина – A, половина – B
    rng = np.random.default_rng(seed)
    user_ids = control.index.to_numpy()
    in_b = rng.permutation(len(user_ids)) < len(user_ids) // 2
    a_values = control[user_ids[~in_b]]
    b_values = treatment[user_ids[in_b]]
    split = stats.independent_bootstrap_diff(b_values, a_values)
    # те же варианты на одних и тех же пользователях
    mean, low, high = stats.paired_bootstrap_diff(treatment, control)
    rows = {
        "A/B: разные половины пользователей": {"diff": split.mean, "low": split.low, "high": split.high,
                                                "users_a": len(a_values), "users_b": len(b_values)},
        "парно: все пользователи в обоих вариантах": {"diff": mean, "low": low, "high": high,
                                                       "users_a": len(control), "users_b": len(treatment)},
    }
    return pd.DataFrame.from_dict(rows, orient="index")


def simulate(group: str = GROUP, rounds: int = ROUNDS) -> dict[str, ArmResult]:
    """Обе ленты на одних и тех же пользователях группы.

    :param group: "val" или "test".
    :param rounds: Число раундов.
    :return: {вариант: ArmResult} в порядке ARMS.
    """
    table, catalog = store.load_interactions()
    recommender, viewers = simulation_recommender(table, catalog)
    users = load_users(table, recommender.item_ids, group)
    logger.info("симуляция: %d пользователей %s, %d раундов по %d аниме", len(users), group, rounds, K)
    results = {}
    for arm in ARMS:
        results[arm] = run_arm(recommender, viewers, users, arm, rounds)
    return results


def main() -> None:
    """Симуляция обратной связи и A/B «система против популярности»; печатает сводки."""
    results = simulate()
    summary = arm_summary(results)
    ab = ab_test(results)
    print(summary.round(4).to_string())
    print(ab.round(4).to_string())


if __name__ == "__main__":
    # подробные логи по шагам; уровень – переменная окружения RECSYS_LOG_LEVEL (по умолчанию INFO, таблицы – DEBUG)
    logs.enable()
    main()
