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
    # все досмотренные train-пользователями аниме
    is_train_completed = (table["split"] == "train") & (table["status"] == "completed")
    completed = table[is_train_completed]

    # часть каждого просмотра: часть его пользователя, -1 — пользователь не попал в части
    fold_of_user = {u: info["fold"] for u, info in users.items()}
    user_fold = completed["user_id"].map(fold_of_user).fillna(-1).astype(int)

    frames = []
    for fold in [-1, *range(rd.N_FOLDS)]:
        # для val считаем по всем, для части f — без её пользователей
        if fold == -1:
            rows = completed
        else:
            rows = completed[user_fold != fold]
        # число разных зрителей каждого аниме, в логарифме
        counts = rows.groupby("anime_id")["user_id"].nunique()
        popularity = np.log1p(counts.to_numpy())
        frames.append(pd.DataFrame({"fold": fold, "anime_id": counts.index, "popularity": popularity}))
    return pd.concat(frames, ignore_index=True)


def build_features(candidates: pd.DataFrame, item_features: pd.DataFrame, popularity: pd.DataFrame) -> pd.DataFrame:
    """Таблица признаков: кандидаты + популярность (по своей части) + признаки аниме.

    :param candidates: Кандидаты из ranking_v0 (group, fold, user_id, anime_id, retrieval_score, ...).
    :param item_features: Результат anime_features, индекс anime_id.
    :param popularity: Результат popularity_by_fold.
    :return: Кандидаты с добавленными колонками-признаками, тот же порядок строк.
    """
    # популярность берём из своей части: для train — без своих пользователей, для val — по всем
    out = candidates.merge(popularity, on=["fold", "anime_id"], how="left")
    # аниме без train-зрителей в своей части: популярность 0
    out["popularity"] = out["popularity"].fillna(0.0)
    # признаки аниме одинаковы для всех пользователей
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
    categorical = features.columns.intersection(anime.CATEGORICAL_FEATURES)
    for column in categorical:
        # pandas хранит пропуски в строковых колонках особым значением: превращаем их в "unknown"
        values = features[column].astype("object")
        values = values.where(values.notna(), "unknown")
        features[column] = values.astype(str).to_numpy(dtype=object)
    return features


def make_pool(data: pd.DataFrame, columns: list[str]) -> Pool:
    """Pool CatBoost: строки одного пользователя идут подряд, group_id — пользователь.

    :param data: Таблица признаков с колонками group, user_id, retrieval_rank и label.
    :param columns: Колонки-признаки из feature_columns.
    :return: Pool для обучения CatBoostRanker.
    """
    # CatBoost требует, чтобы строки одного списка (пользователя) шли подряд
    data = data.sort_values(["group", "user_id", "retrieval_rank"])
    features = categorical_as_text(data[columns])
    # идентификатор списка: группа и пользователь (train и val не смешиваются)
    group_id = data["group"] + "_" + data["user_id"].astype(str)
    cat_features = [c for c in columns if c in anime.CATEGORICAL_FEATURES]
    return Pool(features, label=data["label"], group_id=group_id, cat_features=cat_features)


def train_ranker(train_pool: Pool, params: dict | None = None, eval_pool: Pool | None = None) -> CatBoostRanker:
    """Обучает CatBoostRanker на кандидатах train-пользователей.

    :param train_pool: Pool из make_pool по строкам group == "train".
    :param params: Параметры CatBoost (iterations, learning_rate, depth, l2_leaf_reg, ...);
        None — значения по умолчанию из констант модуля.
    :param eval_pool: Pool для ранней остановки по NDCG@10; None — без ранней остановки.
    :return: Обученная модель.
    """
    # значения по умолчанию, поверх них — переданные параметры
    defaults = {"iterations": ITERATIONS, "learning_rate": LEARNING_RATE, "depth": DEPTH}
    params = {**defaults, **(params or {})}

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
        # остановка, если NDCG@10 на проверочной части 50 итераций не растёт; берём лучшую итерацию
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
        # лента пользователя: кандидаты по убыванию балла
        ordered = group.sort_values(score_column, ascending=False, kind="stable")
        ranked = ordered["anime_id"].to_numpy()
        # спрятанные ответы пользователя
        targets = cases[user_id][1]
        # метрики при каждом k
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
    is_train_completed = (table["split"] == "train") & (table["status"] == "completed")
    matrix_users = np.sort(table.loc[is_train_completed, "user_id"].unique())
    item_ids = np.sort(catalog)
    users = rd.split_train_users(table, item_ids, matrix_users)
    popularity = popularity_by_fold(table, users)

    # признаки аниме только для каталога
    raw_anime = anime.load_anime()
    item_features = anime.anime_features(raw_anime)
    item_features = item_features[item_features.index.isin(item_ids)]

    # общая таблица: каждый кандидат со всеми признаками
    data = build_features(candidates, item_features, popularity)
    columns = feature_columns(item_features)
    return data, columns


def evaluate_on_val(model: CatBoostRanker, data: pd.DataFrame, columns: list[str]) -> dict[str, pd.DataFrame]:
    """Метрики итогового списка на val: CatBoost против порядка iALS.

    :param model: Обученная модель.
    :param data: Результат prepare.
    :param columns: Колонки-признаки.
    :return: {"baseline": ..., "catboost": ...} — метрики по каждому val-пользователю.
    """
    # спрятанные ответы val-пользователей
    table, _ = store.load_interactions()
    cases = protocol.build_cases(table, "val")

    # кандидаты val и баллы модели для них
    val = data[data["group"] == "val"].copy()
    features = categorical_as_text(val[columns])
    val["model_score"] = model.predict(features)
    # опорная линия: порядок iALS (чем меньше место, тем лучше)
    val["baseline_score"] = -val["retrieval_rank"]

    baseline = list_metrics(val, "baseline_score", cases)
    catboost = list_metrics(val, "model_score", cases)
    return {"baseline": baseline, "catboost": catboost}


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
    # обучение на частях train, кроме valid_fold; сравнение наборов — на valid_fold
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
        history = model.get_evals_result()["validation"]["NDCG:top=10;type=Base"]
        trial.set_user_attr("best_iteration", best)
        return float(history[best])

    # TPE: следующий набор выбирается с учётом того, какие наборы уже оказались лучше
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=rd.SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study


def run(params: dict | None = None) -> dict:
    """Собирает признаки, обучает модель на всех train-пользователях и считает метрики на val.

    :param params: Параметры CatBoost (например, study.best_params); None — по умолчанию.
    :return: Словарь: model, data, columns, per_user (метрики val для baseline и catboost),
        importance (важность признаков).
    """
    # признаки всех кандидатов
    data, columns = prepare()

    # обучение на всех train-кандидатах
    train = data[data["group"] == "train"]
    train_pool = make_pool(train, columns)
    model = train_ranker(train_pool, params)

    # оценка на val против порядка iALS
    per_user = evaluate_on_val(model, data, columns)

    # важность признаков: для ранжирующей модели считается по изменению функции потерь на train
    values = model.get_feature_importance(data=train_pool)
    importance = pd.Series(values, index=columns).sort_values(ascending=False)
    return {"model": model, "data": data, "columns": columns, "per_user": per_user, "importance": importance}


def report(per_user: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Средние, 95% интервалы и парная разница CatBoost с опорной линией.

    :param per_user: Результат evaluate_on_val.
    :return: DataFrame: строки — метрики, колонки — средние, интервалы и разница.
    """
    rows = {}
    for metric in ("ndcg@10", "hit_rate@10", "recall@10"):
        row = {}
        # среднее и интервал для каждой модели
        for name, df in per_user.items():
            values = df[metric].to_numpy()
            low, high = stats.bootstrap_ci(values)
            row[name] = values.mean()
            row[f"{name}_low"] = low
            row[f"{name}_high"] = high
        # парная разница CatBoost минус опорная линия на одних и тех же пользователях
        diff, low, high = stats.paired_bootstrap_diff(per_user["catboost"][metric], per_user["baseline"][metric])
        row["diff"] = diff
        row["diff_low"] = low
        row["diff_high"] = high
        rows[metric] = row
    return pd.DataFrame(rows).T


def main() -> None:
    """Обучает CatBoost с параметрами по умолчанию и печатает метрики на val и важность признаков."""
    result = run()
    table = report(result["per_user"])
    print(table.round(4).to_string())
    print()
    print("важность признаков:")
    print(result["importance"].round(2).to_string())


if __name__ == "__main__":
    main()
