from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core import logs
from core.settings import RERANK, RETRIEVAL
from training.evaluation import protocol, stats
from training.ranking import dataset as rd
from training.ranking import model as rk
from training.reranking import evaluate as rerank_eval
from training.retrieval import als, popularity
from training.retrieval import item_knn as knn

# подробности по шагам; включить: core.logs.enable("DEBUG")
logger = logging.getLogger(__name__)

GROUP = "test"
K = RERANK.feed_length  # длина ленты
STAGES = ["популярность", "iALS", "CatBoost", "CatBoost + re-ranking"]


@dataclass(frozen=True)
class StageResult:
    """Лента одной стадии на test: метрики по каждому пользователю и свойства выдачи.

    :param per_user: ndcg@K, recall@K, hit_rate@K (и recall@100 у retrieval) с индексом user_id.
    :param summary: Покрытие, повторы франшиз и продолжения без приквела (FeedSummary).
    """

    per_user: pd.DataFrame
    summary: rerank_eval.FeedSummary


def top_k(rankings: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Первые K аниме каждого списка – лента, которую увидел бы пользователь.

    :param rankings: {user_id: anime_id по убыванию полезности}.
    :return: {user_id: первые K}.
    """
    feeds = {}
    for user_id, ranked in rankings.items():
        feeds[user_id] = ranked[:K]
    return feeds


def retrieval_stage(ranker: protocol.Ranker, context: rerank_eval.RerankContext) -> StageResult:
    """Лента модели retrieval по всему каталогу (без ranking): популярность или iALS.

    :param ranker: Функция ranker(user_id, candidates).
    :param context: rerank_eval.load_context("test").
    :return: StageResult; в per_user есть и recall@100 – метрика стадии retrieval.
    """
    rankings = protocol.rank_users(ranker, context.cases, context.catalog)
    ks = (K, RETRIEVAL.n_candidates)
    per_user = protocol.per_user_metrics(rankings, context.cases, ks)
    feeds = top_k(rankings)
    _, summary = rerank_eval.feed_metrics(feeds, context)
    return StageResult(per_user=per_user.sort_index(), summary=summary)


def ranking_stages(context: rerank_eval.RerankContext) -> tuple[StageResult, StageResult]:
    """CatBoost на 100 кандидатах iALS и он же с правилами ленты.

    CatBoost обучается на train-кандидатах (как на val), признаки test-кандидатов считаются
    по их input-истории.

    :param context: rerank_eval.load_context("test").
    :return: (лента CatBoost, лента CatBoost + re-ranking с настройками RERANK).
    """
    # модель – как на val: все train-кандидаты, все признаки
    data, columns = rk.prepare()
    train = data[data["group"] == "train"]
    train_pool = rk.make_pool(train, columns)
    model = rk.train_ranker(train_pool)

    # кандидаты iALS для test-пользователей и их признаки
    inputs = rk.load_ranking_inputs()
    matrix, _, item_ids = knn.build_train_matrix(inputs.table, context.catalog)
    candidates, _ = rd.build_eval_candidates(inputs.table, context.catalog, matrix, item_ids, GROUP)
    features = rk.assemble_features(candidates, inputs.table, item_ids, inputs.users, GROUP)
    scored = rk.score_split(model, features, columns, GROUP)

    # лента CatBoost как есть и с правилами
    plain_feeds = rerank_eval.build_feeds(scored, context, require_prequel=False)
    plain_per_user, plain_summary = rerank_eval.feed_metrics(plain_feeds, context)
    reranked_feeds = rerank_eval.build_feeds(scored, context, RERANK.max_per_franchise, RERANK.popularity_weight)
    reranked_per_user, reranked_summary = rerank_eval.feed_metrics(reranked_feeds, context)
    plain = StageResult(per_user=plain_per_user, summary=plain_summary)
    reranked = StageResult(per_user=reranked_per_user, summary=reranked_summary)
    return plain, reranked


def evaluate_stages() -> dict[str, StageResult]:
    """Все стадии пайплайна на test-пользователях: популярность, iALS, CatBoost, re-ranking.

    :return: {название стадии: StageResult} в порядке STAGES.
    """
    context = rerank_eval.load_context(GROUP)

    # retrieval: популярность и iALS по train-пользователям
    scores = popularity.popularity_scores(context.table)
    popularity_ranker = popularity.make_popularity_ranker(scores)
    matrix, _, item_ids = knn.build_train_matrix(context.table, context.catalog)
    histories = knn.build_histories(context.table, GROUP, item_ids)
    confidence = als.build_confidence_matrix(matrix)
    als_model = als.fit_als(confidence, RETRIEVAL.factors)
    als_ranker = als.make_als_ranker(als_model, item_ids, histories)

    # ranking и re-ranking
    catboost, reranked = ranking_stages(context)
    return {
        "популярность": retrieval_stage(popularity_ranker, context),
        "iALS": retrieval_stage(als_ranker, context),
        "CatBoost": catboost,
        "CatBoost + re-ranking": reranked,
    }


def stage_table(results: dict[str, StageResult], criterion: rerank_eval.Criterion) -> pd.DataFrame:
    """Итоговая таблица: метрики каждой стадии с интервалами и разница с предыдущей стадией.

    :param results: Результат evaluate_stages.
    :param criterion: Пороги критерия итогового списка на тех же пользователях.
    :return: DataFrame: строка – стадия.
    """
    rows = {}
    previous = None
    for name in STAGES:
        result = results[name]
        ndcg = result.per_user[f"ndcg@{K}"]
        # ndcg@10 с интервалом и парная разница с предыдущей стадией
        reference = ndcg if previous is None else previous
        compared = stats.compare(ndcg, reference)
        row = {
            f"ndcg@{K}": compared.value.mean,
            "ndcg_low": compared.value.low,
            "ndcg_high": compared.value.high,
            "diff_to_previous": compared.diff.mean,
            "diff_low": compared.diff.low,
            "diff_high": compared.diff.high,
            f"hit_rate@{K}": result.per_user[f"hit_rate@{K}"].mean(),
            **result.summary.as_row(),
            "критерий": criterion.met(compared.value.mean, result.summary.coverage),
        }
        # recall@100 есть только у моделей retrieval (у ranking – 100 кандидатов на входе)
        recall_column = f"recall@{RETRIEVAL.n_candidates}"
        if recall_column in result.per_user:
            row[recall_column] = result.per_user[recall_column].mean()
        rows[name] = row
        previous = ndcg
    # строки – стадии; у каждой колонки свой тип (числа и признак критерия)
    return pd.DataFrame.from_dict(rows, orient="index")


def main() -> None:
    """Один раз считает все стадии на test и печатает итоговую таблицу."""
    results = evaluate_stages()
    context = rerank_eval.load_context(GROUP)
    criterion = rerank_eval.criterion_thresholds(context)
    table = stage_table(results, criterion)
    rounded = table.round(4)
    print(f"test: пользователей {len(context.cases)}; критерий ndcg@{K} >= {criterion.ndcg_min:.3f}, coverage@{K} >= {criterion.coverage_min:.3f}")
    print(rounded.to_string())


if __name__ == "__main__":
    # подробные логи по шагам; уровень – переменная окружения RECSYS_LOG_LEVEL (по умолчанию INFO, таблицы – DEBUG)
    logs.enable()
    main()
