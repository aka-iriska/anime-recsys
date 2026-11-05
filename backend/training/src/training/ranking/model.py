from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostRanker, Pool

from training.evaluation import metrics, protocol, stats
from training.interactions import store
from core import config, logs
from core.catalog import anime, franchise, history, pairs
from core.catalog.pairs import BASE_GROUPS, FEATURE_GROUPS, group_columns
from core.settings import PROTOCOL, RANKING
from core.validation import require
from training.ranking import dataset as rd
from training.retrieval.popularity import item_popularity


# подробности по шагам; включить: core.logs.enable("DEBUG")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RankingInputs:
    """Данные, из которых собираются признаки ranking.

    :param table: Таблица взаимодействий из config.INTERACTIONS.
    :param item_ids: anime_id каталога по порядку.
    :param candidates: Кандидаты из config.RANKING_DATA.
    :param users: Результат rd.split_train_users (те же части, что при сборке датасета).
    """

    table: pd.DataFrame
    item_ids: np.ndarray
    candidates: pd.DataFrame
    users: dict[int, rd.UserSplit]


@dataclass(frozen=True)
class ValEvaluation:
    """Метрики итогового списка на val по каждому пользователю.

    :param baseline: Порядок iALS (опорная линия).
    :param catboost: Порядок модели.
    """

    baseline: pd.DataFrame
    catboost: pd.DataFrame


@dataclass(frozen=True)
class RunResult:
    """Результат run: модель, её данные, метрики на val и важность признаков.

    :param model: Обученная модель.
    :param data: Результат prepare.
    :param columns: Колонки-признаки.
    :param evaluation: Метрики на val (evaluate_on_val).
    :param importance: Важность признаков по убыванию.
    """

    model: CatBoostRanker
    data: pd.DataFrame
    columns: list[str]
    evaluation: ValEvaluation
    importance: pd.Series


def popularity_by_fold(table: pd.DataFrame, users: dict[int, rd.UserSplit]) -> pd.DataFrame:
    """log(1 + число train-зрителей аниме) отдельно для каждой части без её пользователей.

    Для train-пользователей части f популярность считается по всем train-пользователям, кроме
    части f (их спрятанные просмотры иначе попали бы в признак). Для val (fold = -1) – по всем
    train-пользователям.

    :param table: Таблица взаимодействий из config.INTERACTIONS.
    :param users: Результат rd.split_train_users (у кого какая часть).
    :return: DataFrame с колонками fold, anime_id, popularity.
    """
    # все досмотренные train-пользователями аниме
    is_train_completed = (table["split"] == "train") & (table["status"] == "completed")
    completed = table[is_train_completed]

    # часть каждого просмотра: часть его пользователя, -1 – пользователь не попал в части
    fold_of_user = {user_id: user.fold for user_id, user in users.items()}
    user_fold = completed["user_id"].map(fold_of_user).fillna(-1).astype(int)

    frames = []
    for fold in [-1, *range(RANKING.n_folds)]:
        # для val считаем по всем, для части f – без её пользователей
        if fold == -1:
            rows = completed
        else:
            rows = completed[user_fold != fold]
        # популярность – то же определение, что у baseline и штрафа в выдаче
        popularity = item_popularity(rows)
        frame = pd.DataFrame({"fold": fold, "anime_id": popularity.index, "popularity": popularity.to_numpy()})
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    logs.table(logger, "популярность по частям: log(1 + зрителей), fold = -1 – для val", result)
    return result


def build_features(candidates: pd.DataFrame, item_features: pd.DataFrame, popularity: pd.DataFrame) -> pd.DataFrame:
    """Таблица признаков: кандидаты + популярность (по своей части) + признаки аниме.

    :param candidates: Кандидаты из config.RANKING_DATA (group, fold, user_id, anime_id, retrieval_score, ...).
    :param item_features: Результат anime_features, индекс anime_id.
    :param popularity: Результат popularity_by_fold.
    :return: Кандидаты с добавленными колонками-признаками, тот же порядок строк.
    """
    # популярность берём из своей части: для train – без своих пользователей, для val – по всем
    out = candidates.merge(popularity, on=["fold", "anime_id"], how="left")
    # аниме без train-зрителей в своей части: популярность 0
    out["popularity"] = out["popularity"].fillna(0.0)
    # признаки аниме одинаковы для всех пользователей
    out = out.merge(item_features, left_on="anime_id", right_index=True, how="left")
    return out


def load_favourites() -> pd.DataFrame:
    """Избранные аниме пользователей из снимка данных.

    :return: DataFrame с колонками user_id, anime_id (в таблице есть и персонажи, люди, манга –
        они отброшены).
    """
    favourites = pd.read_csv(config.SOURCE / "user_favourites.csv")
    favourites = favourites[favourites["category"] == "animes"]
    favourites = favourites.rename(columns={"target_id": "anime_id"})
    return favourites[["user_id", "anime_id"]]


def make_pool(data: pd.DataFrame, columns: list[str], partial: bool = False) -> Pool:
    """Pool CatBoost: строки одного пользователя идут подряд, group_id – пользователь.

    По умолчанию требует полный набор признаков из FEATURE_GROUPS: так CatBoost нельзя
    случайно обучить или применить без признаков пары (франшиза, жанры, профиль, избранное).

    :param data: Таблица признаков с колонками group, user_id, retrieval_rank и label.
    :param columns: Колонки-признаки, обычно все: group_columns(list(FEATURE_GROUPS)).
    :param partial: True – разрешить неполный набор; только для сравнений (ablation, шаг 2).
    :return: Pool для обучения CatBoostRanker.
    """
    # каких признаков из полного набора не хватает; без явного partial=True это ошибка
    full = group_columns(list(FEATURE_GROUPS))
    missing = [c for c in full if c not in columns]
    if missing and not partial:
        raise ValueError(f"CatBoost без признаков {missing}; для сравнения урезанных наборов передайте partial=True")

    # CatBoost требует, чтобы строки одного списка (пользователя) шли подряд
    data = data.sort_values(["group", "user_id", "retrieval_rank"])
    features = anime.categorical_as_text(data[columns])
    # идентификатор списка: группа и пользователь (train и val не смешиваются)
    group_id = data["group"] + "_" + data["user_id"].astype(str)
    cat_features = [c for c in columns if c in anime.CATEGORICAL_FEATURES]
    n_rows = logs.num(len(data))
    n_lists = logs.num(group_id.nunique())
    logger.debug("Pool: %s строк, %s списков (пользователей), %d признаков, из них категориальных %d",
                 n_rows, n_lists, len(columns), len(cat_features))
    return Pool(features, label=data["label"], group_id=group_id, cat_features=cat_features)


def train_ranker(train_pool: Pool, params: dict | None = None, eval_pool: Pool | None = None) -> CatBoostRanker:
    """Обучает CatBoostRanker на кандидатах train-пользователей.

    :param train_pool: Pool из make_pool по строкам group == "train".
    :param params: Параметры CatBoost (iterations, learning_rate, depth, l2_leaf_reg,
        random_seed, ...); None – значения по умолчанию из констант модуля.
    :param eval_pool: Pool для ранней остановки по NDCG@10; None – без ранней остановки.
    :return: Обученная модель.
    """
    # значения по умолчанию, поверх них – переданные параметры
    defaults = {"iterations": RANKING.iterations, "learning_rate": RANKING.learning_rate, "depth": RANKING.depth, "random_seed": PROTOCOL.seed}
    overrides = params or {}
    params = {**defaults, **overrides}

    model = CatBoostRanker(
        loss_function=RANKING.loss,
        eval_metric="NDCG:top=10",
        verbose=False,
        allow_writing_files=False,  # не создавать папку catboost_info с логами
        **params,
    )
    logger.info("обучение CatBoost: %s", params)
    if eval_pool is None:
        model.fit(train_pool)
    else:
        # остановка, если NDCG@10 на проверочной части 50 итераций не растёт; берём лучшую итерацию
        model.fit(train_pool, eval_set=eval_pool, use_best_model=True, early_stopping_rounds=50)
    return model


class SeedEnsemble:
    """Несколько CatBoost, обученных одинаково, но с разными seed; балл – среднее их баллов.

    Обучение CatBoost случайно (какие пары кандидатов YetiRank сравнивает, случайность при
    выборе разбиений), поэтому модели с разными seed ошибаются по-разному. Среднее их баллов
    гасит эту случайную часть. Интерфейс как у модели: predict(features).
    """

    def __init__(self, models: list[CatBoostRanker]):
        """Собирает ансамбль из обученных моделей.

        :param models: Модели из train_ranker на одних и тех же признаках.
        """
        self.models = models

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Средний балл моделей ансамбля для каждой строки.

        :param features: Признаки в том же виде, что для одной модели (после categorical_as_text).
        :return: Вектор формы (len(features),).
        """
        # баллы каждой модели: матрица (n_models, n_rows)
        per_model = []
        for model in self.models:
            per_model.append(model.predict(features))
        scores = np.stack(per_model)
        # среднее по моделям
        return scores.mean(axis=0)


def train_seed_ensemble(train_pool: Pool, seeds: list[int], params: dict | None = None) -> SeedEnsemble:
    """Обучает по одной модели CatBoost на каждый seed и объединяет их в ансамбль.

    Стоимость: len(seeds) обычных обучений.

    :param train_pool: Pool из make_pool по строкам group == "train".
    :param seeds: Значения random_seed, по одному на модель.
    :param params: Общие параметры CatBoost; None – по умолчанию.
    :return: SeedEnsemble из len(seeds) моделей.
    """
    models = []
    for seed in seeds:
        # те же параметры, другой seed
        common = params or {}
        seed_params = {**common, "random_seed": seed}
        model = train_ranker(train_pool, seed_params)
        models.append(model)
    return SeedEnsemble(models)


def list_metrics(scored: pd.DataFrame, score_column: str, cases: dict, ks=(10, 20)) -> pd.DataFrame:
    """Метрики итогового списка по каждому пользователю при порядке кандидатов по score_column.

    Спрятанные ответы берутся целиком (из cases), включая не попавшие в кандидаты, поэтому
    значения сравнимы с метриками retrieval.

    :param scored: Кандидаты одной группы (обычно val) с колонкой score_column.
    :param score_column: Колонка, по убыванию которой упорядочиваем кандидатов.
    :param cases: Результат protocol.build_cases: {user_id: (seen, targets)}.
    :param ks: Значения k для метрик.
    :return: DataFrame: индекс user_id, колонки recall@k, hit_rate@k, ndcg@k.
    """
    rows = {}
    for user_id, group in scored.groupby("user_id"):
        # лента пользователя: кандидаты по убыванию балла
        ordered = group.sort_values(score_column, ascending=False, kind="stable")
        ranked = ordered["anime_id"].to_numpy()
        # метрики по спрятанным ответам пользователя при каждом k
        hidden = cases[user_id][1]
        rows[user_id] = metrics.ranked_metrics(ranked, hidden, ks)
    return pd.DataFrame.from_dict(rows, orient="index")


def load_ranking_inputs() -> RankingInputs:
    """Данные, из которых собираются признаки: протокол, кандидаты и части train-пользователей.

    :return: RankingInputs.
    """
    table, catalog = store.load_interactions()
    candidates = pd.read_csv(rd.OUTPUT / "candidates.csv")
    # train-пользователи в порядке строк матрицы (как в knn.build_train_matrix)
    is_train_completed = (table["split"] == "train") & (table["status"] == "completed")
    train_user_ids = table.loc[is_train_completed, "user_id"].unique()
    matrix_users = np.sort(train_user_ids)
    item_ids = np.sort(catalog)
    # те же части train-пользователей, что при сборке датасета
    users = rd.split_train_users(table, item_ids, matrix_users)
    return RankingInputs(table=table, item_ids=item_ids, candidates=candidates, users=users)


def assemble_features(candidates: pd.DataFrame, table: pd.DataFrame, item_ids: np.ndarray,
                      users: dict[int, rd.UserSplit], group: str = "val") -> pd.DataFrame:
    """Кандидаты со всеми признаками: популярность по своей части, признаки аниме и пары.

    :param candidates: Кандидаты (group, fold, user_id, anime_id, retrieval_score, ...).
    :param table: Таблица взаимодействий из config.INTERACTIONS.
    :param item_ids: anime_id каталога по порядку.
    :param users: Результат rd.split_train_users.
    :param group: Проверочная группа, для которой берутся input-истории: "val" или "test".
    :return: Кандидаты с колонками всех групп FEATURE_GROUPS.
    """
    # популярность без пользователей своей части (иначе утечка)
    popularity = popularity_by_fold(table, users)

    # признаки аниме только для каталога
    raw_anime = anime.load_anime()
    item_features = anime.anime_features(raw_anime)
    item_features = item_features[item_features.index.isin(item_ids)]

    # общая таблица: каждый кандидат с признаками аниме и популярностью
    data = build_features(candidates, item_features, popularity)

    # признаки пары по тому, что модель «знает» о пользователе (input без спрятанных ответов)
    histories = rd.input_histories(table, item_ids, users, group)
    favourites = load_favourites()
    links = franchise.load_anime_links()
    genres = history.load_anime_genres()
    by_pair = pairs.pair_features(candidates, histories, favourites, item_features, links, genres)
    data = data.merge(by_pair, on=["user_id", "anime_id"], how="left")

    # признаки франшизы посчитаны для каждой пары (нет связей – 0); NaN значит, что пара не нашлась
    missing = data[franchise.FEATURES].isna().any(axis=1)
    n_missing = int(missing.sum())
    require(n_missing == 0, f"У {n_missing} кандидатов не посчитаны признаки пары")
    return data


def prepare() -> tuple[pd.DataFrame, list[str]]:
    """Кандидаты из config.RANKING_DATA со всеми признаками: iALS, популярность, аниме и признаки пары.

    :return: (data, columns): таблица всех кандидатов (train и val) с признаками и список
        всех колонок-признаков (все группы FEATURE_GROUPS).
    """
    inputs = load_ranking_inputs()
    data = assemble_features(inputs.candidates, inputs.table, inputs.item_ids, inputs.users)
    logs.table(logger, "итоговая таблица для CatBoost: кандидат + все признаки + label", data)
    logs.counts(logger, "строки по группам", data["group"])

    # все признаки всех групп: модель по умолчанию учится только на полном наборе
    columns = group_columns(list(FEATURE_GROUPS))
    return data, columns


def val_cases(group: str = "val") -> dict:
    """Спрятанные ответы проверочных пользователей (protocol.build_cases по данным протокола).

    :param group: "val" или "test".
    :return: {user_id: (seen, targets)}.
    """
    table, _ = store.load_interactions()
    return protocol.build_cases(table, group)


def baseline_metrics(data: pd.DataFrame, cases: dict, group: str = "val") -> pd.DataFrame:
    """Метрики опорной линии: порядок кандидатов у iALS.

    :param data: Результат prepare (или признаки test-кандидатов).
    :param cases: Результат val_cases для той же группы.
    :param group: "val" или "test".
    :return: Метрики по каждому пользователю группы (как list_metrics).
    """
    val = data[data["group"] == group].copy()
    # чем меньше место у iALS, тем выше балл
    val["baseline_score"] = -val["retrieval_rank"]
    return list_metrics(val, "baseline_score", cases)


def score_split(model, data: pd.DataFrame, columns: list[str], group: str = "val") -> pd.DataFrame:
    """Кандидаты одной группы с баллом модели – общий шаг для оценки ranking и re-ranking.

    :param model: Обученная модель (CatBoostRanker или SeedEnsemble): у неё есть predict.
    :param data: Результат prepare.
    :param columns: Колонки-признаки, на которых обучена модель.
    :param group: "train" или "val".
    :return: Строки группы (копия) с колонкой score.
    """
    rows = data[data["group"] == group].copy()
    features = anime.categorical_as_text(rows[columns])
    rows["score"] = model.predict(features)
    return rows


def evaluate_on_val(model: CatBoostRanker, data: pd.DataFrame, columns: list[str],
                    cases: dict | None = None, group: str = "val") -> ValEvaluation:
    """Метрики итогового списка на проверочной группе: CatBoost против порядка iALS.

    :param model: Обученная модель.
    :param data: Результат prepare (или признаки test-кандидатов).
    :param columns: Колонки-признаки.
    :param cases: Результат val_cases той же группы; None – прочитать данные протокола (это
        несколько секунд, поэтому при многих вызовах подряд лучше передать готовые).
    :param group: "val" (по умолчанию) или "test" – только для финальной проверки.
    :return: ValEvaluation – метрики по каждому пользователю группы для iALS и модели.
    """
    # спрятанные ответы пользователей группы
    if cases is None:
        cases = val_cases(group)

    # кандидаты группы и баллы модели для них
    val = score_split(model, data, columns, group)
    # как CatBoost переставил кандидатов первого val-пользователя (только для отладки)
    if logger.isEnabledFor(logging.DEBUG):
        first_id = val["user_id"].iloc[0]
        first_user = val[val["user_id"] == first_id]
        first_user = first_user.sort_values("score", ascending=False)
        shown = first_user[["user_id", "anime_id", "retrieval_rank", "score", "label"]]
        logs.table(logger, "val: кандидаты одного пользователя по баллу CatBoost", shown, rows=10)

    baseline = baseline_metrics(data, cases, group)
    catboost = list_metrics(val, "score", cases)
    return ValEvaluation(baseline=baseline, catboost=catboost)


def tune_ranker(data: pd.DataFrame, columns: list[str], n_trials: int = 20, timeout: int = 900,
                valid_fold: int = 0, partial: bool = False) -> optuna.Study:
    """Подбор параметров CatBoost через Optuna на частях train, без val.

    Обучаемся на частях train, кроме valid_fold, и смотрим NDCG@10 на valid_fold (с ранней
    остановкой). Val в подборе не участвует, поэтому итоговая оценка на val остаётся честной.

    :param data: Результат prepare.
    :param columns: Колонки-признаки.
    :param n_trials: Сколько наборов параметров попробовать.
    :param timeout: Ограничение по времени в секундах.
    :param valid_fold: Часть train, на которой сравниваются наборы параметров.
    :param partial: True – разрешить неполный набор признаков (см. make_pool).
    :return: Исследование Optuna; лучшие параметры – study.best_params.
    """
    # обучение на частях train, кроме valid_fold; сравнение наборов – на valid_fold
    train = data[data["group"] == "train"]
    fit_pool = make_pool(train[train["fold"] != valid_fold], columns, partial)
    valid_pool = make_pool(train[train["fold"] == valid_fold], columns, partial)

    def objective(trial: optuna.Trial) -> float:
        # пространство поиска: темп обучения, глубина деревьев, регуляризация листьев
        params = {
            "iterations": 600,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        }
        model = train_ranker(fit_pool, params, eval_pool=valid_pool)

        # NDCG@10 на проверочной части в лучшей итерации (get_best_score у ранжирования пуст)
        best = model.get_best_iteration()
        evals = model.get_evals_result()
        ndcg_by_iteration = evals["validation"]["NDCG:top=10;type=Base"]
        trial.set_user_attr("best_iteration", best)
        return float(ndcg_by_iteration[best])

    # TPE: следующий набор выбирается с учётом того, какие наборы уже оказались лучше
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=PROTOCOL.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study


def run(params: dict | None = None) -> RunResult:
    """Собирает признаки, обучает модель на всех train-пользователях и считает метрики на val.

    :param params: Параметры CatBoost (например, study.best_params); None – по умолчанию.
    :return: RunResult.
    """
    # признаки всех кандидатов
    data, columns = prepare()

    # обучение на всех train-кандидатах
    train = data[data["group"] == "train"]
    train_pool = make_pool(train, columns)
    model = train_ranker(train_pool, params)

    # оценка на val против порядка iALS
    evaluation = evaluate_on_val(model, data, columns)

    # важность признаков: для ранжирующей модели считается по изменению функции потерь на train
    values = model.get_feature_importance(data=train_pool)
    importance = pd.Series(values, index=columns)
    importance = importance.sort_values(ascending=False)
    return RunResult(model=model, data=data, columns=columns, evaluation=evaluation, importance=importance)


def ablation(data: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Сколько даёт каждая группа признаков: модель со всеми группами против модели без одной.

    Обучает одну модель на всех группах, по одной модели без каждой группы признаков пары
    и модель шага 2 (без признаков пары). Каждая оценивается на val.

    :param data: Результат prepare.
    :param params: Параметры CatBoost; None – по умолчанию.
    :return: DataFrame: строки – варианты, колонки ndcg@10 и парная разница с моделью на всех
        группах (diff, diff_low, diff_high), а также разница с порядком iALS.
    """
    train = data[data["group"] == "train"]
    all_groups = list(FEATURE_GROUPS)
    pair_groups = [g for g in all_groups if g not in BASE_GROUPS]

    # варианты: все группы, без каждой группы пары, только группы шага 2
    variants = {"все группы": all_groups}
    for name in pair_groups:
        variants[f"без {name}"] = [g for g in all_groups if g != name]
    variants["шаг 2 (без признаков пары)"] = list(BASE_GROUPS)

    # спрятанные ответы и опорная линия (порядок iALS) – один раз для всех вариантов
    cases = val_cases()
    baseline_table = baseline_metrics(data, cases)
    baseline = baseline_table["ndcg@10"]

    per_user = {}
    for name, groups in variants.items():
        # обучение на train и метрики по каждому val-пользователю
        columns = group_columns(groups)
        train_pool = make_pool(train, columns, partial=True)
        model = train_ranker(train_pool, params)
        evaluation = evaluate_on_val(model, data, columns, cases)
        per_user[name] = evaluation.catboost["ndcg@10"]

    rows = {}
    reference = per_user["все группы"]
    for name, values in per_user.items():
        # парная разница с моделью на всех группах и с порядком iALS на одних и тех же пользователях
        to_all = stats.compare(values, reference)
        to_ials = stats.compare(values, baseline)
        rows[name] = {"ndcg@10": to_all.value.mean,
                      "diff": to_all.diff.mean, "diff_low": to_all.diff.low, "diff_high": to_all.diff.high,
                      "vs_ials": to_ials.diff.mean, "vs_ials_low": to_ials.diff.low, "vs_ials_high": to_ials.diff.high}
    return pd.DataFrame(rows).T


def report(evaluation: ValEvaluation) -> pd.DataFrame:
    """Средние, 95% интервалы и парная разница CatBoost с опорной линией.

    :param evaluation: Результат evaluate_on_val.
    :return: DataFrame: строки – метрики, колонки – средние, интервалы и разница.
    """
    rows = {}
    for metric in ("ndcg@10", "hit_rate@10", "recall@10"):
        baseline = evaluation.baseline[metric]
        catboost = evaluation.catboost[metric]
        # среднее и интервал опорной линии
        base = stats.summarize(baseline)
        # среднее и интервал CatBoost и его парная разница с опорной линией
        model = stats.compare(catboost, baseline)
        rows[metric] = {"baseline": base.mean, "baseline_low": base.low, "baseline_high": base.high,
                        "catboost": model.value.mean, "catboost_low": model.value.low, "catboost_high": model.value.high,
                        "diff": model.diff.mean, "diff_low": model.diff.low, "diff_high": model.diff.high}
    return pd.DataFrame(rows).T


def main() -> None:
    """Обучает CatBoost с параметрами по умолчанию и печатает метрики на val и важность признаков."""
    result = run()
    table = report(result.evaluation)
    rounded = table.round(4)
    print(rounded.to_string())
    print()
    print("важность признаков:")
    importance = result.importance.round(2)
    print(importance.to_string())


if __name__ == "__main__":
    # подробные логи по шагам; уровень – переменная окружения RECSYS_LOG_LEVEL (по умолчанию INFO, таблицы – DEBUG)
    logs.enable()
    main()
