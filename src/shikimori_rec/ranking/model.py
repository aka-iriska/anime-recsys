from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

from shikimori_rec.evaluation import metrics, protocol, stats
from shikimori_rec.interactions import store
from shikimori_rec.catalog import anime
from shikimori_rec.ranking import dataset as rd


ITERATIONS = 500
LEARNING_RATE = 0.05
DEPTH = 6
LOSS = "YetiRank"  # функция потерь CatBoost для ранжирования внутри списка пользователя


def popularity_by_fold(table: pd.DataFrame, users: dict[int, dict]) -> pd.DataFrame:
    """log(1 + число train-зрителей аниме) отдельно для каждой части без её пользователей.

    Для train-пользователей части f популярность считается по всем train-пользователям, кроме
    части f (их спрятанные просмотры иначе попали бы в признак). Для val (fold = -1) — по всем
    train-пользователям.

    :param table: Таблица взаимодействий из interactions_v0.
    :param users: Результат rd.split_train_users (у кого какая часть).
    :return: DataFrame с колонками fold, anime_id, popularity.
    """
    completed = table[(table["split"] == "train") & (table["status"] == "completed")]
    fold_of_user = {u: info["fold"] for u, info in users.items()}
    user_fold = completed["user_id"].map(fold_of_user).fillna(-1).astype(int)  # -1: не в частях

    frames = []
    for fold in [-1, *range(rd.N_FOLDS)]:
        rows = completed if fold == -1 else completed[user_fold != fold]
        counts = rows.groupby("anime_id")["user_id"].nunique()
        frames.append(pd.DataFrame({"fold": fold, "anime_id": counts.index, "popularity": np.log1p(counts.to_numpy())}))
    return pd.concat(frames, ignore_index=True)


def build_features(candidates: pd.DataFrame, item_features: pd.DataFrame, popularity: pd.DataFrame) -> pd.DataFrame:
    """Таблица признаков: кандидаты + популярность (по своей части) + признаки аниме.

    :param candidates: Кандидаты из ranking_v0 (group, fold, user_id, anime_id, retrieval_score, ...).
    :param item_features: Результат anime_features, индекс anime_id.
    :param popularity: Результат popularity_by_fold.
    :return: Кандидаты с добавленными колонками-признаками, тот же порядок строк.
    """
    out = candidates.merge(popularity, on=["fold", "anime_id"], how="left")
    out["popularity"] = out["popularity"].fillna(0.0)  # аниме без train-зрителей в своей части
    out = out.merge(item_features, left_on="anime_id", right_index=True, how="left")
    return out


def feature_columns(item_features: pd.DataFrame) -> list[str]:
    """Колонки, которые подаются в модель (без идентификаторов и метки).

    :param item_features: Результат anime_features.
    :return: Список названий колонок-признаков.
    """
    return ["retrieval_score", "retrieval_rank", "popularity", *item_features.columns]


def categorical_as_text(features: pd.DataFrame) -> pd.DataFrame:
    """Категориальные колонки — обычные строки, пропуск — "unknown".

    CatBoost не принимает NaN в категориальных признаках.

    :param features: Таблица признаков.
    :return: Копия таблицы, где колонки из CATEGORICAL_FEATURES приведены к строкам.
    """
    features = features.copy()
    for column in features.columns.intersection(anime.CATEGORICAL_FEATURES):
        values = features[column].astype("object")
        features[column] = values.where(values.notna(), "unknown").astype(str).to_numpy(dtype=object)
    return features


def make_pool(data: pd.DataFrame, columns: list[str]) -> Pool:
    """Pool CatBoost: строки одного пользователя идут подряд, group_id — пользователь.

    :param data: Таблица признаков с колонками group, user_id, retrieval_rank и label.
    :param columns: Колонки-признаки из feature_columns.
    :return: Pool для обучения CatBoostRanker.
    """
    data = data.sort_values(["group", "user_id", "retrieval_rank"])
    features = categorical_as_text(data[columns])
    return Pool(
        features,
        label=data["label"],
        group_id=data["group"] + "_" + data["user_id"].astype(str),
        cat_features=[c for c in columns if c in anime.CATEGORICAL_FEATURES],
    )


def train_ranker(train_pool: Pool) -> CatBoostRanker:
    """Обучает CatBoostRanker на кандидатах train-пользователей.

    :param train_pool: Pool из make_pool по строкам group == "train".
    :return: Обученная модель.
    """
    model = CatBoostRanker(
        loss_function=LOSS,
        iterations=ITERATIONS,
        learning_rate=LEARNING_RATE,
        depth=DEPTH,
        random_seed=rd.SEED,
        verbose=False,
        allow_writing_files=False,  # не создавать папку catboost_info с логами
    )
    model.fit(train_pool)
    return model


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
        ranked = group.sort_values(score_column, ascending=False, kind="stable")["anime_id"].to_numpy()
        targets = cases[user_id][1]
        row = {}
        for k in ks:
            row[f"recall@{k}"] = metrics.recall_at_k(ranked, targets, k)
            row[f"hit_rate@{k}"] = metrics.hit_rate_at_k(ranked, targets, k)
            row[f"ndcg@{k}"] = metrics.ndcg_at_k(ranked, targets, k)
        rows[user_id] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def run() -> dict:
    """Собирает признаки, обучает модель и считает метрики на val.

    :return: Словарь: model, data (все признаки), val (кандидаты val с баллами),
        per_user (метрики по пользователям для baseline и catboost), importance.
    """
    table, catalog = store.load_interactions()
    candidates = pd.read_csv(rd.OUTPUT / "candidates.csv")

    matrix_users = np.sort(table.loc[(table["split"] == "train") & (table["status"] == "completed"), "user_id"].unique())
    item_ids = np.sort(catalog)
    users = rd.split_train_users(table, item_ids, matrix_users)  # те же части, что при сборке датасета

    item_features = anime.anime_features(anime.load_anime()).loc[lambda d: d.index.isin(item_ids)]
    data = build_features(candidates, item_features, popularity_by_fold(table, users))
    columns = feature_columns(item_features)

    train_pool = make_pool(data[data["group"] == "train"], columns)
    model = train_ranker(train_pool)
    val = data[data["group"] == "val"].copy()
    val["model_score"] = model.predict(categorical_as_text(val[columns]))
    val["baseline_score"] = -val["retrieval_rank"]  # опорная линия: порядок iALS

    cases = protocol.build_cases(table, "val")
    per_user = {
        "baseline": list_metrics(val, "baseline_score", cases),
        "catboost": list_metrics(val, "model_score", cases),
    }
    # для ранжирующей модели важность считается по изменению функции потерь на обучающих данных
    importance = pd.Series(model.get_feature_importance(data=train_pool), index=columns).sort_values(ascending=False)
    return {"model": model, "data": data, "val": val, "per_user": per_user, "importance": importance}


def main() -> None:
    """Печатает метрики на val со 95% интервалами, разницу с опорной линией и важность признаков."""
    result = run()
    per_user = result["per_user"]
    print("val, метрики итогового списка (среднее [95% интервал]):")
    for metric in ("ndcg@10", "hit_rate@10", "recall@10"):
        line = []
        for name, df in per_user.items():
            low, high = stats.bootstrap_ci(df[metric].to_numpy())
            line.append(f"{name} {df[metric].mean():.4f} [{low:.4f}; {high:.4f}]")
        diff, low, high = stats.paired_bootstrap_diff(per_user["catboost"][metric], per_user["baseline"][metric])
        print(f"  {metric}: " + " | ".join(line) + f" | разница {diff:+.4f} [{low:+.4f}; {high:+.4f}]")
    print("\nважность признаков:")
    print(result["importance"].round(2).to_string())


if __name__ == "__main__":
    main()
