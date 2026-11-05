from __future__ import annotations

from dataclasses import asdict, dataclass

from core import config


@dataclass(frozen=True)
class ProtocolSettings:
    """Протокол оценки: каталог, позитив и разбиение пользователей.

    Любое изменение меняет данные на диске: после него нужна новая версия
    config.INTERACTIONS (interactions_vN) и пересборка, иначе код и данные разойдутся.

    :param kept_kinds: Типы аниме, которые могут попасть в каталог.
    :param min_completed_per_anime: Аниме попадает в каталог, если его досмотрели не меньше
        стольких train-пользователей.
    :param main_positive: Колонка-флаг основного позитива (остальные – контрольные срезы).
    :param min_explicit_for_mean: Минимум явных оценок пользователя для флага p_above_mean.
    :param min_positives_per_user: Пользователь с меньшим числом позитивов всегда в train.
    :param val_user_fraction: Доля оцениваемых пользователей в val.
    :param test_user_fraction: Доля оцениваемых пользователей в test.
    :param target_fraction: Доля позитивов val/test-пользователя, которая прячется в target.
    :param target_rule: Какие аниме нельзя прятать (см. split.pick_targets); пишется в manifest.
    :param seed: Зерно случайности разбиения.
    """

    kept_kinds: tuple[str, ...] = ("tv", "movie", "ova", "ona", "special", "tv_special")
    min_completed_per_anime: int = 5
    main_positive: str = "p_completed"
    min_explicit_for_mean: int = 5
    min_positives_per_user: int = 5
    val_user_fraction: float = 0.1
    test_user_fraction: float = 0.1
    target_fraction: float = 0.2
    target_rule: str = "no anime whose direct sequel the user has also completed"
    seed: int = config.SEED


@dataclass(frozen=True)
class RetrievalSettings:
    """Retrieval для ленты: iALS (E05) и число кандидатов для ranking.

    :param factors: Длина векторов iALS.
    :param alpha: Уверенность просмотра (implicit умножает значение клетки на alpha).
    :param regularization: Штраф за размер векторов.
    :param iterations: Число чередований «пользователи / аниме».
    :param n_candidates: Сколько кандидатов retrieval передаёт в ranking.
    """

    factors: int = 128
    alpha: float = 5.0
    regularization: float = 0.01
    iterations: int = 15
    n_candidates: int = 100


@dataclass(frozen=True)
class RankingSettings:
    """Ranking: датасет кандидатов и CatBoost (E06–E07).

    :param n_folds: На сколько частей делим train-пользователей (iALS каждой части – без неё).
    :param min_completed_per_user: Train-пользователь даёт примеры, если досмотрел не меньше.
    :param iterations: Число деревьев CatBoost.
    :param learning_rate: Темп обучения CatBoost.
    :param depth: Глубина деревьев CatBoost.
    :param loss: Функция потерь CatBoost для ранжирования внутри списка пользователя.
    """

    n_folds: int = 5
    min_completed_per_user: int = 5
    iterations: int = 500
    learning_rate: float = 0.05
    depth: int = 6
    loss: str = "YetiRank"


@dataclass(frozen=True)
class RerankSettings:
    """Правила ленты (E08, serving.rerank): наименьший штраф, при котором с одним аниме на
    франшизу выполняется критерий покрытия итогового списка.

    :param feed_length: Длина ленты.
    :param max_per_franchise: Не больше стольких аниме одной франшизы; None – без лимита.
    :param popularity_weight: Сила штрафа за популярность; 0 – без штрафа.
    """

    feed_length: int = 10
    max_per_franchise: int | None = 1
    popularity_weight: float = 0.7


@dataclass(frozen=True)
class EvaluationSettings:
    """Оценка: длины списков и критерий успеха итогового списка (E01).

    :param ks: Значения k для метрик; 100 – метрика стадии retrieval.
    :param accuracy_share: ndcg@10 модели не ниже этой доли от популярности.
    :param coverage_factor: coverage@10 модели не меньше, чем во столько раз больше популярности.
    """

    ks: tuple[int, ...] = (10, 20, 100)
    accuracy_share: float = 0.9
    coverage_factor: float = 10.0


# принятые решения; обучение читает их отсюда, экспорт пишет их в артефакты выдачи
PROTOCOL = ProtocolSettings()
RETRIEVAL = RetrievalSettings()
RANKING = RankingSettings()
RERANK = RerankSettings()
EVALUATION = EvaluationSettings()


def as_manifest(**groups) -> dict:
    """Настройки в виде словаря для manifest.json, по группе на ключ.

    :param groups: Группы настроек по именам, например protocol=PROTOCOL, retrieval=RETRIEVAL.
    :return: {имя группы: {поле: значение}}; кортежи превращены в списки (как в JSON).
    """
    result = {}
    for name, group in groups.items():
        fields = asdict(group)
        result[name] = {key: list(value) if isinstance(value, tuple) else value for key, value in fields.items()}
    return result
