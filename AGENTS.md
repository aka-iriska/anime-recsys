# Project instructions

## Collaboration and learning workflow

- Communicate in Russian unless the user asks otherwise. Keep explanations concrete and introduce theory alongside the current implementation step.
- Work one project stage at a time. Before a substantial component, explain the problem it solves, why it is needed, where it fits, and the main trade-offs.
- Preserve the user's learning role: invite the user to implement the central ML logic, then review it before presenting a polished version. Small setup and plumbing tasks may be implemented directly when requested.
- Do not generate future stages or a complete repository ahead of the user's progress.
- Ask meaningful design questions only. Avoid trivial recall quizzes; prefer questions that require choosing and defending a modeling, data, or evaluation decision.

## Project layout

- Keep all Python code in the uv workspace under `packages/`: three installable packages, each organised by domain (DDD-style bounded contexts). Do not put Python files elsewhere:
  - `packages/core` → `core`: shared kernel used by both sides — `config` (paths, seed), `validation`, `catalog` (`rules`: what may be recommended; `anime`: anime attributes and features). Depends only on numpy/pandas.
  - `packages/serving` → `serving`: production inference from exported artifacts — `artifacts`, `recommender` (later the FastAPI app). Depends on `core` only; it must never import `training` or training-only libraries (`implicit`, `scipy`).
  - `packages/training` → `training`: offline work — `ingestion` (raw → tables), `interactions` (labels, user split, protocol data), `evaluation` (metrics, loop, statistics), `retrieval` (popularity, item-kNN, SVD, iALS), `ranking` (candidate dataset, CatBoost), `pipelines` (train final models, export artifacts). May import `core` and `serving` (artifact format).
- Deployment files live in `deploy/` (one folder per service, e.g. `deploy/backend/Dockerfile`) and `docker-compose.yml` at the root. The backend image installs only `serving` and mounts `models/` read-only.
- Declare dependencies in the package that needs them (`packages/*/pyproject.toml`); notebook tools stay in the root `dev` group. Production installs only the serving package: `uv sync --package serving --no-dev`.
- Keep research and project artifacts outside `packages/`: notebooks in `notebooks/`, raw and processed data in `data/`, design notes in `docs/`, and experiment records in `experiments/`.
- Use notebooks for exploration, visualizations, and written findings. Put reusable pipeline and model code in the package; notebooks import it directly (`from training.retrieval import als`).
- Prefer small functions first. Add classes only when a component has useful persistent state or a clear interface.
- Prefer flat, readable code: one step per line with a named intermediate variable instead of nested calls (write `model = fit_als(confidence, factors)` then `make_als_ranker(model, ...)`, not `make_als_ranker(fit_als(confidence, factors), ...)`). Put a one-line comment before each step of an algorithm saying what it does.
- Do not put documentation at the top of Python files (no module docstrings); explanations of a stage belong in the notebook or in `docs/`.
- Write function docstrings in Russian in this format: a one-line summary; optionally a short paragraph with details; then `:param name:` for every parameter and `:return:` for the result. Example:

  ```python
  def popularity_by_fold(table: pd.DataFrame, users: dict[int, dict]) -> pd.DataFrame:
      """log(1 + число train-зрителей аниме) отдельно для каждой части без её пользователей.

      Для train-пользователей части f популярность считается по всем train-пользователям,
      кроме части f. Для val (fold = -1) — по всем train-пользователям.

      :param table: Таблица взаимодействий из interactions_v0.
      :param users: Результат rd.split_train_users (у кого какая часть).
      :return: DataFrame с колонками fold, anime_id, popularity.
      """
  ```
- Manage Python dependencies through `uv` and `pyproject.toml`; do not add the local project itself as a third-party dependency.

## Data handling

- Treat the Hugging Face dataset `kdduha/shikimori-recsys` as the initial source. Pin its revision for reproducible downloads and keep original files unchanged in `data/raw/`.
- Keep downloaded datasets and generated model artifacts out of Git unless the user explicitly decides to publish them. Never commit access tokens or credentials.
- Validate schemas, missing values, duplicate user–anime pairs, unknown anime IDs, score ranges, and interaction counts before modeling.
- Interpret `score=0` as no explicit rating, not dislike. A missing user–anime row is unknown, not a negative label. Preserve viewing progress as a separate signal.
- Dataset audit found 67,071 rates from 1,151 users, 38,942 zero scores (58.06%), 1,616 rate rows (2.41%) without matching anime metadata, six duplicate user–anime pairs, and one duplicate anime ID. Re-check these facts if the source revision changes.
- Observed `createdAt` values cluster between 2025-11-07 and 2026-02-23; many users have nearly all rows on a single day. Do not use these fields for temporal train/test splitting unless their event-time meaning is validated.
- Prevent temporal and feature leakage: every training or ranking feature must be available at the simulated recommendation time.

## Modeling and evaluation

- Keep the product task as personalized Top-K anime recommendation. Rating prediction may be an internal signal, not the end product.
- State the positive-label policy explicitly before constructing training examples. Keep positive, explicit negative, and unknown interactions distinct.
- Start with simple baselines, including random and popularity ranking, before collaborative filtering or more complex models.
- Compare models on the same users, candidate catalog, split, and metrics. Report Recall@K or HitRate@K, NDCG@K, and catalog coverage where appropriate.
- Treat offline ranking metrics as proxies. Do not claim online product improvement without impression and outcome logs or a real experiment.
- Keep each experiment's hypothesis, conclusions, and analysis in its notebook; `experiments/README.md` is an index with data/split, the change, key metrics, and a link to the notebook. Update `docs/system_design.md` when assumptions change.

## Implementation and reporting

- Make changes only for the current stage and explain the input, output, data shapes, and relevant computational cost.
- Use `uv run` from the project root for commands; run package modules with `uv run python -m training.<context>.<module>`. State exactly what command to run and what output to expect.
- Do not claim data, code, or metrics have been checked unless they were actually inspected or run.
- Frontend work and a GitHub demo video are later product tasks; first establish a reproducible data and offline evaluation pipeline.
