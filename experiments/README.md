# Журнал экспериментов

Это индекс: данные и разбиение, что менялось, ключевые метрики и ссылка на разбор. Гипотезы, выводы и таблицы ведутся в ноутбуках. Протокол оценки: `docs/system_design.md`, раздел «Разбиение и оценка». Метрики усредняются по пользователям; оценка офлайн, это прокси, а не измерение продукта. Для retrieval главная метрика — `recall@100`, для итогового списка — `ndcg@10`.

## Данные и разбиение

Все эксперименты — на одном протоколе: `data/processed/interactions_v2` (`uv run python -m training.interactions.prepare`, `seed=42`). Каталог 6198 аниме (типы `tv`, `movie`, `ova`, `ona`, `special`, `tv_special`; не меньше 5 `completed` у train-пользователей). Позитив — `status == 'completed'`. Пользователи: train 1649, val 173, test 173. У val/test 20% позитивов спрятано в `target`, но никогда не аниме, продолжение которого пользователь тоже досмотрел (нет «дыр» во франшизах). Оценка на **val**, test не использовался. Ranking — на `data/processed/ranking_v2`: 100 кандидатов iALS для 1386 train-пользователей (5 частей, iALS каждой части обучен без неё) и 173 val-пользователей.

## Индекс

| ID | Стадия | Что | Ключевая метрика (val) | Где разбор |
| --- | --- | --- | --- | --- |
| E01 | retrieval | baseline'ы: random, popularity | `recall@100` 0,018 / 0,327 | `notebooks/02_baselines.ipynb` |
| E02 | retrieval | item-item по совместному просмотру, вес `1/n` (`1/sqrt(n)`) | `recall@100` 0,413 (0,417) | `notebooks/03_item_knn.ipynb` |
| E02c | retrieval | поправка на популярность в item-item | не лучше 0,416; критерий итогового списка не выполняется ни при одном `alpha` | `notebooks/03_item_knn.ipynb` |
| E04 | retrieval | PureSVD, `k=64`, вес `1/sqrt(n)` | `recall@100` 0,555 [0,522; 0,585] | `notebooks/04_retrieval_comparison.ipynb` |
| E04b | retrieval | PureSVD с нормировкой столбцов, `beta=0,25` | `recall@100` 0,551, не лучше E04 | `notebooks/04_retrieval_comparison.ipynb` |
| E05 | retrieval | iALS (`implicit`), бинарная уверенность, `alpha=5`, `factors=128` | `recall@100` 0,582 [0,547; 0,615], +0,027 [0,006; 0,048] к PureSVD | `notebooks/04_retrieval_comparison.ipynb` |
| E06 | ranking | CatBoost (`YetiRank`) на 100 кандидатах iALS: признаки iALS, популярность, признаки аниме | `ndcg@10` 0,508 против 0,484 у порядка iALS (+0,024 [0,011; 0,038]) | `notebooks/05_ranking.ipynb` |
| E07 | ranking | E06 + признаки пары по input-истории: франшиза, жанры, профиль, избранное; сравнение групп, Optuna, ансамбль из 5 seed | `ndcg@10` 0,590 (+0,106 [0,082; 0,129] к iALS); прирост почти весь от франшизы, Optuna и ансамбль не помогают | `notebooks/05_ranking.ipynb` |
