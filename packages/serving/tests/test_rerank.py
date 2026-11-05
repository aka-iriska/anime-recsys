import numpy as np
import pandas as pd

from serving.rerank import allowed_by_prequel, rerank


# пять кандидатов: 1, 2, 3 – одна франшиза, 4 и 5 – сами по себе; баллы по убыванию
ANIME = np.array([1, 2, 3, 4, 5])
SCORES = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
FLAT_POPULARITY = np.zeros(5)
FRANCHISE = pd.Series(["a", "a", "a", "anime_4", "anime_5"], index=ANIME)


def feed(**kwargs) -> list[int]:
    """Лента по умолчанию без правила приквела и без штрафа; правила задаются явно."""
    params = {"completed": set(), "prequels": {}, "k": 3, "max_per_franchise": None,
              "popularity_weight": 0.0, "require_prequel": False}
    params.update(kwargs)
    return rerank(ANIME, SCORES, FLAT_POPULARITY, FRANCHISE, **params).tolist()


def test_without_rules_keeps_ranking_order():
    assert feed() == [1, 2, 3]


def test_franchise_limit_lets_next_franchises_up():
    assert feed(max_per_franchise=1) == [1, 4, 5]
    assert feed(max_per_franchise=2) == [1, 2, 4]


def test_franchise_limit_backfills_when_not_enough_franchises():
    # разных франшиз всего три, а лента длиной 4: добираем лучшим отложенным
    assert feed(max_per_franchise=1, k=4) == [1, 4, 5, 2]


def test_prequel_rule_hides_sequel_until_prequel_completed():
    prequels = {2: {1}}
    assert allowed_by_prequel(ANIME, set(), prequels).tolist() == [True, False, True, True, True]
    assert allowed_by_prequel(ANIME, {1}, prequels).tolist() == [True, True, True, True, True]
    assert feed(prequels=prequels, require_prequel=True) == [1, 3, 4]


def test_any_completed_prequel_is_enough():
    prequels = {3: {1, 2}}
    assert allowed_by_prequel(np.array([3]), {2}, prequels).tolist() == [True]


def test_popularity_penalty_moves_popular_down():
    popularity = np.array([10.0, 0.0, 0.0, 0.0, 0.0])
    ranked = rerank(ANIME, SCORES, popularity, FRANCHISE, set(), {}, k=2,
                    max_per_franchise=None, popularity_weight=1.0, require_prequel=False)
    assert ranked.tolist() == [2, 3]


def test_ties_keep_input_order():
    ranked = rerank(ANIME, np.ones(5), FLAT_POPULARITY, FRANCHISE, set(), {}, k=5,
                    max_per_franchise=None, popularity_weight=0.0, require_prequel=False)
    assert ranked.tolist() == [1, 2, 3, 4, 5]


def test_everything_filtered_gives_empty_feed():
    prequels = {a: {99} for a in ANIME.tolist()}
    assert feed(prequels=prequels, require_prequel=True) == []
