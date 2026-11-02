from __future__ import annotations

import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostRanker, Pool

from training.evaluation import metrics, protocol, stats
from training.interactions import store
from core.catalog import anime
from training.ranking import dataset as rd


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


def train_ranker(train_pool: Pool, params: dict | None = None, eval_pool: Pool | None = None) -> CatBoostRanker:
    """Обучает CatBoostRanker на кандидатах train-пользователей.

    :param train_pool: Pool из make_pool по строкам group == "train".
    :param params: Параметры CatBoost (iterations, learning_rate, depth, l2_leaf_reg, ...);
        None — значения по умолчанию из констант модуля.
    :param eval_pool: Pool для ранней остановки по NDCG@10; None — без ранней остановки.
    :return: Обученная модель.
    """
    params = {"iterations": ITERATIONS, "learning_rate": LEARNING_RATE, "depth": DEPTH, **(params or {})}
    model = CatBoostRanker(
        loss_function=LOSS,
        eval_metric="NDCG:top=10",
        random_seed=rd.SEED,
        verbose=False,
        allow_writing_files=False,  # не создавать папку catboost_info с логами
        **params,
    )
    if eval_pool is None:
        model.fit(train_pool)
    else:
        model.fit(train_pool, eval_set=eval_pool, use_best_model=True, early_stopping_rounds=50)
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


def prepare() -> tuple[pd.DataFrame, list[str]]:
    """Кандидаты ranking_v0 с признаками: баллы iALS, популярность по своей части, признаки аниме.

    :return: (data, columns): таблица всех кандидатов (train и val) с признаками и список
        колонок-признаков для модели.
    """
    # данные протокола и кандидаты от iALS
    table, catalog = store.load_interactions()
    candidates = pd.read_csv(rd.OUTPUT / "candidates.csv")

    # те же части train-пользователей, что при сборке датасета (нужны для популярности без утечки)
    matrix_users = np.sort(table.loc[(table["split"] == "train") & (table["status"] == "completed"), "user_id"].unique())
    item_ids = np.sort(catalog)
    users = rd.split_train_users(table, item_ids, matrix_users)

    # признаки аниме только для каталога и сборка общей таблицы
    item_features = anime.anime_features(anime.load_anime()).loc[lambda d: d.index.isin(item_ids)]
    data = build_features(candidates, item_features, popularity_by_fold(table, users))
    return data, feature_columns(item_features)


def evaluate_on_val(model: CatBoostRanker, data: pd.DataFrame, columns: list[str]) -> dict[str, pd.DataFrame]:
    """Метрики итогового списка на val: CatBoost против порядка iALS.

    :param model: Обученная модель.
    :param data: Результат prepare.
    :param columns: Колонки-признаки.
    :return: {"baseline": ..., "catboost": ...} — метрики по каждому val-пользователю.
    """
    table, _ = store.load_interactions()
    cases = protocol.build_cases(table, "val")
    val = data[data["group"] == "val"].copy()
    # баллы модели и опорная линия: порядок iALS
    val["model_score"] = model.predict(categorical_as_text(val[columns]))
    val["baseline_score"] = -val["retrieval_rank"]
    return {"baseline": list_metrics(val, "baseline_score", cases), "catboost": list_metrics(val, "model_score", cases)}


def tune_ranker(data: pd.DataFrame, columns: list[str], n_trials: int = 20, timeout: int = 900,
                valid_fold: int = 0) -> optuna.Study:
    """Подбор параметров CatBoost через Optuna на частях train, без val.

    Обучаемся на частях train, кроме valid_fold, и смотрим NDCG@10 на valid_fold (с ранней
    остановкой). Val в подборе не участвует, поэтому итоговая оценка на val остаётся честной.

    :param data: Результат prepare.
    :param columns: Колонки-признаки.
    :param n_trials: Сколько наборов параметров попробовать.
    :param timeout: Ограничение по времени в секундах.
    :param valid_fold: Часть train, на которой сравниваются наборы параметров.
    :return: Исследование Optuna; лучшие параметры — study.best_params.
    """
    train = data[data["group"] == "train"]
    fit_pool = make_pool(train[train["fold"] != valid_fold], columns)
    valid_pool = make_pool(train[train["fold"] == valid_fold], columns)

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
        best = int(model.get_best_iteration())
        ndcg = model.get_evals_result()["validation"]["NDCG:top=10;type=Base"][best]
        trial.set_user_attr("best_iteration", best)
        return float(ndcg)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=rd.SEED))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study


def run(params: dict | None = None) -> dict:
    """Собирает признаки, обучает модель на всех train-пользователях и считает метрики на val.

    :param params: Параметры CatBoost (например, study.best_params); None — по умолчанию.
    :return: Словарь: model, data, columns, per_user (метрики val для baseline и catboost),
        importance (важность признаков).
    """
    data, columns = prepare()
    train_pool = make_pool(data[data["group"] == "train"], columns)
    model = train_ranker(train_pool, params)
    per_user = evaluate_on_val(model, data, columns)
    # для ранжирующей модели важность считается по изменению функции потерь на обучающих данных
    importance = pd.Series(model.get_feature_importance(data=train_pool), index=columns).sort_values(ascending=False)
    return {"model": model, "data": data, "columns": columns, "per_user": per_user, "importance": importance}


def report(per_user: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Средние, 95% интервалы и парная разница CatBoost с опорной линией.

    :param per_user: Результат evaluate_on_val.
    :return: DataFrame: строки — метрики, колонки — средние, интервалы и разница.
    """
    rows = {}
    for metric in ("ndcg@10", "hit_rate@10", "recall@10"):
        row = {}
        for name, df in per_user.items():
            low, high = stats.bootstrap_ci(df[metric].to_numpy())
            row[f"{name}"], row[f"{name}_low"], row[f"{name}_high"] = df[metric].mean(), low, high
        diff, low, high = stats.paired_bootstrap_diff(per_user["catboost"][metric], per_user["baseline"][metric])
        row["diff"], row["diff_low"], row["diff_high"] = diff, low, high
        rows[metric] = row
    return pd.DataFrame(rows).T


def main() -> None:
    """Обучает CatBoost с параметрами по умолчанию и печатает метрики на val и важность признаков."""
    result = run()
    print(report(result["per_user"]).round(4).to_string())
    print()
    print("важность признаков:")
    print(result["importance"].round(2).to_string())


if __name__ == "__main__":
    main()
