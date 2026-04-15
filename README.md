# Automated War News Geopolitical Prediction Engine

A fully autonomous MLOps system that continuously ingests war and conflict news from the web and **forecasts the probability of geopolitical events** — blockades, strikes, escalations, proxy activations — before they happen, grounded in historical conflict patterns and current signals.

This is not an anomaly detector. It does not stop at *"something strange is happening"*. It answers the next question: **what is likely to happen, and with what probability, and in what time window.**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Why a Prediction Engine (Not Just Anomaly Detection)](#why-a-prediction-engine-not-just-anomaly-detection)
3. [Real-World Scenario (2026 US–Israel–Iran Conflict)](#real-world-scenario-2026-usisraeliran-conflict)
4. [System Goals](#system-goals)
5. [High-Level Architecture](#high-level-architecture)
6. [Pipeline Components](#pipeline-components)
   - [1. Automated News Extraction](#1-automated-news-extraction)
   - [2. Data Versioning (DVC)](#2-data-versioning-dvc)
   - [3. Pipeline Orchestration (Airflow)](#3-pipeline-orchestration-airflow)
   - [4. Preprocessing & Feature Engineering](#4-preprocessing--feature-engineering)
   - [5. Historical Conflict Knowledge Base](#5-historical-conflict-knowledge-base)
   - [6. Hyperparameter Optimization (Optuna)](#6-hyperparameter-optimization-optuna)
   - [7. Prediction Model Training](#7-prediction-model-training)
   - [8. Experiment Tracking & Registry (MLflow)](#8-experiment-tracking--registry-mlflow)
   - [9. Evaluation Gate](#9-evaluation-gate)
   - [10. Prediction API](#10-prediction-api)
   - [11. Production Scaling (Kubeflow)](#11-production-scaling-kubeflow)
   - [12. Monitoring, Drift & Retraining Triggers](#12-monitoring-drift--retraining-triggers)
   - [13. Cold Start & Rare Event Handling](#13-cold-start--rare-event-handling)
   - [14. Failure Modes & Limitations](#14-failure-modes--limitations)
   - [15. Natural-Language Explanation Layer (Small LLM)](#15-natural-language-explanation-layer-small-llm)
   - [16. LLM vs. ML Classifier — Division of Labor](#16-llm-vs-ml-classifier--division-of-labor)
7. [Prediction Outputs](#prediction-outputs)
8. [End-to-End System Flow](#end-to-end-system-flow)
9. [System Intelligence Layers](#system-intelligence-layers)
10. [Real-World Enhancements](#real-world-enhancements)
11. [Project Impact](#project-impact)
12. [Tech Stack](#tech-stack)

---

## Project Overview

The **Geopolitical Prediction Engine** ingests war-related news in near real time, combines it with a knowledge base of historical conflicts, and outputs **probabilistic forecasts** of future geopolitical actions — e.g. *"65–80% probability the US imposes a naval blockade within 5 days."*

The system is fully automated end-to-end: scraping, versioning, training, hyperparameter search, evaluation, deployment, and scaling.

---

## Why a Prediction Engine (Not Just Anomaly Detection)

| | Anomaly Detection | Prediction Engine |
|--|--|--|
| Answers | "Something unusual is happening" | "Here is what will likely happen next, with probability and time window" |
| Output | Alert / flag | Probability distribution over future events |
| Uses history? | Minimal (baseline only) | Central — historical conflicts drive forecasts |
| Actionability | Low (raw signal) | High (decision-ready) |
| Role | Radar | Commander interpreting the radar |

Anomaly detection is one **upstream signal**; the prediction engine consumes those signals together with historical patterns to produce forward-looking forecasts.

---

## Real-World Scenario (2026 US–Israel–Iran Conflict)

The system is validated against real events from the ongoing 2026 conflict.

**Observed signals in the news stream:**
- Sudden increase in US naval movement near the Strait of Hormuz
- Unusual military logistics buildup
- Surge in oil tanker rerouting
- Israeli troop concentration near Lebanon border
- Collapsing US–Iran diplomatic channels, Iran pivoting toward Europe

**Engine outputs (example forecasts):**

| Input signals | Forecast | Time window | Confidence |
|--|--|--|--|
| US fleet buildup near Hormuz + tanker rerouting | US imposes naval blockade on Iran | ≤ 5 days | 65–80% |
| Iran missile transport + IRGC comms spike + social blackout | Iranian strike on Israeli asset | ≤ 7 days | 55–70% |
| Israeli troop spike near Lebanon + failed talks | Escalation against Hezbollah in Lebanon | ≤ 3 days | 70–85% |

These patterns matched real reporting (Reuters, The Guardian, Al Jazeera, The Washington Post) confirming a US blockade on Iran and active Israeli operations in Lebanon in April 2026.

---

## System Goals

- **Continuous news ingestion** from global sources
- **Reproducible datasets** tied to each forecast
- **Historical conflict grounding** — every prediction is explainable against past wars
- **Probabilistic forecasts** with explicit confidence and time windows
- **Self-tuning** models via automated hyperparameter search
- **Safe automated deployment** behind an evaluation gate
- **Scalable inference** under production traffic

---

## High-Level Architecture

```
News Scraping → DVC → Airflow → Preprocessing + Historical KB → Optuna + Training → MLflow → Evaluation Gate → Prediction API → Kubeflow Scaling
```

---

## Pipeline Components

### 1. Automated News Extraction

Continuously scrapes war and geopolitical news across sources.

**Sources:**
- Reuters, BBC, Al Jazeera, The Guardian, The Washington Post
- NewsAPI, GDELT
- RSS feeds
- Twitter/X (optional OSINT layer)

**Example record:**

```json
{
  "title": "US begins Iran port blockade",
  "text": "...",
  "source": "reuters",
  "entities": ["US", "Iran", "Strait of Hormuz"],
  "event_type": "blockade",
  "timestamp": "2026-04-14T08:00:00Z"
}
```

---

### 2. Data Versioning (DVC)

Every dataset — raw news, cleaned corpus, labeled events, historical conflict KB snapshots — is versioned with DVC.

```bash
dvc add data/news_raw
dvc add data/historical_conflicts
dvc push
```

**Why it matters:** every forecast can be traced back to the exact dataset version that produced it.

---

### 3. Pipeline Orchestration (Airflow)

Airflow orchestrates the full lifecycle. Each step is an isolated task with its own retry policy, logs, and idempotency key. A failure in any step halts promotion — production keeps serving the previous model.

**DAG overview:**

```
scrape_news → clean_and_extract → merge_with_history
            → optuna_search → train_predictor
            → evaluate → (gate) → deploy_shadow → deploy_canary → deploy_prod
```

---

#### Step 1 — `scrape_news`

**Purpose:** fetch the latest articles across all configured sources since the last successful run.

**Details:**
- Reads `last_run_ts` from Airflow Variable (or XCom from prior run).
- Hits Reuters/BBC/Al Jazeera RSS, NewsAPI, GDELT.
- Writes raw JSONL to `data/news_raw/{{ ds }}/` (partitioned by execution date).
- Emits article count to XCom; alerts if below a floor (source outage signal).

```python
from airflow.decorators import task
import requests, json, pathlib

@task(retries=3, retry_delay=timedelta(minutes=5))
def scrape_news(ds: str, **ctx) -> dict:
    out_dir = pathlib.Path(f"data/news_raw/{ds}")
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for source in ["reuters", "bbc", "aljazeera", "guardian", "gdelt"]:
        articles = fetch_source(source, since=ctx["prev_ds"])
        with (out_dir / f"{source}.jsonl").open("w") as f:
            for a in articles:
                f.write(json.dumps(a) + "\n")
        counts[source] = len(articles)
    if sum(counts.values()) < 50:
        raise ValueError(f"Suspiciously low article count: {counts}")
    return counts
```

---

#### Step 2 — `clean_and_extract`

**Purpose:** deduplicate, run NER + event extraction, score escalation tone, compute embeddings, aggregate time series. Produces the feature table.

**Details:**
- MinHash dedup collapses wire-syndicated duplicates.
- spaCy / HF NER tags entities.
- Event classifier tags each article with an event type from the taxonomy.
- `sentence-transformers` produces 768-d embeddings.
- Per-actor / per-region counts aggregated to hourly bins.
- Output: `data/features/{{ ds }}/features.parquet`.

```python
@task(retries=2)
def clean_and_extract(ds: str) -> str:
    raw = load_jsonl_glob(f"data/news_raw/{ds}/*.jsonl")
    deduped = minhash_dedup(raw, threshold=0.85)
    enriched = []
    for art in deduped:
        art["entities"]   = ner_model(art["text"])
        art["event_type"] = event_classifier(art["text"])
        art["tone"]       = escalation_scorer(art["text"])
        art["embedding"]  = sbert.encode(art["text"]).tolist()
        enriched.append(art)
    features = aggregate_time_series(enriched, freq="1H")
    out = f"data/features/{ds}/features.parquet"
    features.to_parquet(out)
    return out
```

---

#### Step 3 — `merge_with_history`

**Purpose:** join the live feature table with the Historical Conflict KB so training can see both current and historical windows.

**Details:**
- Loads versioned KB from DVC (`data/historical_conflicts`).
- Slices each historical conflict timeline into `(features_[t−N, t], label_[t, t+H])` training pairs.
- Appends today's window as an unlabeled inference row.
- Output: `data/training/{{ ds }}/dataset.parquet` + `dataset_hash` (SHA-256 of file content) written to XCom — this hash goes into every prediction response for auditability.

```python
@task
def merge_with_history(ds: str, feature_path: str) -> dict:
    live   = pd.read_parquet(feature_path)
    kb     = load_dvc("data/historical_conflicts")
    train  = build_sliding_windows(kb, context_days=30, horizon_days=7)
    full   = pd.concat([train, live], ignore_index=True)
    out = f"data/training/{ds}/dataset.parquet"
    full.to_parquet(out)
    h = sha256_file(out)
    return {"path": out, "dataset_hash": h}
```

---

#### Step 4 — `optuna_search`

**Purpose:** find the best hyperparameter configuration for this cycle's data.

**Details:**
- Runs N trials (default 30). Each trial trains a candidate on a walk-forward split and returns Brier + AUC.
- Best config persisted to `artifacts/{{ ds }}/best_params.json`.
- Time-boxed (e.g. 2 h) so the DAG can't stall indefinitely.

```python
@task(execution_timeout=timedelta(hours=2))
def optuna_search(dataset: dict) -> dict:
    import optuna
    df = pd.read_parquet(dataset["path"])

    def objective(trial):
        params = {
            "model":         trial.suggest_categorical("model", ["xgb", "tft", "transformer"]),
            "context_days":  trial.suggest_int("context_days", 7, 30),
            "lr":            trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size":    trial.suggest_categorical("batch_size", [32, 64, 128]),
            "focal_gamma":   trial.suggest_float("focal_gamma", 0.0, 3.0),
        }
        model = train_candidate(df, params, walk_forward=True)
        return brier_score(model, df)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    return study.best_params
```

---

#### Step 5 — `train_predictor`

**Purpose:** train the final model on the full training set using the best params and log everything to MLflow.

**Details:**
- Fits the model chosen by Optuna.
- Fits the isotonic calibrator on a time-forward holdout.
- Logs params, metrics, model artifact, and calibrator to MLflow.
- Registers model as `predictor` with a `candidate` alias (not yet prod).

```python
@task
def train_predictor(dataset: dict, best_params: dict, ds: str) -> str:
    import mlflow
    df = pd.read_parquet(dataset["path"])
    X_train, y_train, X_hold, y_hold = walk_forward_split(df)

    with mlflow.start_run(run_name=f"train_{ds}") as run:
        mlflow.log_params(best_params)
        model = fit_model(X_train, y_train, best_params)
        raw_scores = model.predict_proba(X_hold)
        calibrator = fit_isotonic(raw_scores, y_hold)

        mlflow.log_metric("brier", brier_score(calibrator.transform(raw_scores), y_hold))
        mlflow.log_metric("auc",   roc_auc(raw_scores, y_hold))
        mlflow.log_metric("ece",   expected_calibration_error(calibrator.transform(raw_scores), y_hold))
        mlflow.pyfunc.log_model("predictor", python_model=CalibratedPredictor(model, calibrator))
        mlflow.set_tag("dataset_hash", dataset["dataset_hash"])

        client = mlflow.MlflowClient()
        mv = client.create_model_version("predictor", run.info.artifact_uri + "/predictor", run.info.run_id)
        client.set_registered_model_alias("predictor", "candidate", mv.version)
        return mv.version
```

---

#### Step 6 — `evaluate` (the gate)

**Purpose:** compare the candidate against the current production model on the full metric suite (§9). If it fails, the DAG short-circuits — deploy tasks never run.

**Details:**
- Pulls `predictor@prod` and `predictor@candidate` from MLflow registry.
- Scores both on the same holdout.
- Gate rule enforces Brier, ECE, AUC, per-event PR-AUC, recall@1%FPR.
- Failure raises `AirflowSkipException` → downstream deploy tasks skipped.

```python
from airflow.exceptions import AirflowSkipException

@task
def evaluate(candidate_version: str, dataset: dict) -> str:
    prod      = mlflow.pyfunc.load_model("models:/predictor@prod")
    candidate = mlflow.pyfunc.load_model(f"models:/predictor/{candidate_version}")
    X, y = load_holdout(dataset["path"])

    p_new, p_old = candidate.predict(X), prod.predict(X)

    gate = (
        brier(p_new, y)       <  brier(p_old, y)
        and ece(p_new, y)     <= ece(p_old, y) * 1.05
        and auc(p_new, y)     >= auc(p_old, y)
        and all(pr_auc(p_new, y, e) >= pr_auc(p_old, y, e) for e in EVENT_TYPES)
        and recall_at_fpr(p_new, y, 0.01) >= recall_at_fpr(p_old, y, 0.01) - 0.02
    )
    if not gate:
        raise AirflowSkipException("Candidate failed evaluation gate; keeping prod.")
    return candidate_version
```

---

#### Step 7 — `deploy_shadow` → `deploy_canary` → `deploy_prod`

**Purpose:** progressive rollout. A model that passes the gate still runs in shadow for 7 days, then 48h canary at 10% traffic, before full promotion.

**Details:**
- `deploy_shadow`: sets alias `shadow`; the serving layer mirrors requests to it and logs predictions, but does not return them to clients.
- `deploy_canary`: after 7 days of clean shadow metrics (checked by a separate scheduled DAG), promotes to `canary` at 10% traffic.
- `deploy_prod`: after 48h of clean canary metrics, flips the `prod` alias.

```python
@task
def deploy_shadow(version: str):
    mlflow.MlflowClient().set_registered_model_alias("predictor", "shadow", version)
    k8s_rollout("predictor-shadow", version)

@task(trigger_rule="all_success")
def deploy_canary(version: str):
    if shadow_metrics_clean(version, days=7):
        mlflow.MlflowClient().set_registered_model_alias("predictor", "canary", version)
        k8s_rollout("predictor-canary", version, traffic_pct=10)

@task(trigger_rule="all_success")
def deploy_prod(version: str):
    if canary_metrics_clean(version, hours=48):
        mlflow.MlflowClient().set_registered_model_alias("predictor", "prod", version)
        k8s_rollout("predictor-prod", version, traffic_pct=100)
```

---

#### Full DAG wiring

```python
from airflow.decorators import dag
from datetime import datetime, timedelta

@dag(
    dag_id="geopolitical_predictor",
    schedule="0 */6 * * *",          # every 6 hours
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=10)},
)
def predictor_pipeline():
    counts    = scrape_news()
    features  = clean_and_extract("{{ ds }}")
    dataset   = merge_with_history("{{ ds }}", features)
    params    = optuna_search(dataset)
    version   = train_predictor(dataset, params, "{{ ds }}")
    approved  = evaluate(version, dataset)
    shadow    = deploy_shadow(approved)
    canary    = deploy_canary(approved)
    prod      = deploy_prod(approved)

    counts >> features >> dataset >> params >> version >> approved >> shadow >> canary >> prod

predictor_pipeline()
```

**Operational notes:**
- Every task is idempotent on `ds` (execution date) — reruns are safe.
- Secrets (API keys, MLflow credentials) come from Airflow Connections, never hard-coded.
- Failure in any task → `prod` alias stays on the previous model; serving is unaffected.
- `dataset_hash` from step 3 is tagged on the MLflow run and returned in every `/predict` response, making forecasts auditable end-to-end.

---

#### Serving Data Pipeline — `news_ingest_live` (Mini-DAG)

The training DAG above runs every 6 hours and produces **models**. The `/predict` API also needs **fresh features** at request time; if features are 6 hours stale, forecasts degrade even when the model is good. A second, lightweight DAG runs continuously to keep a **feature store** up to date.

**Design constraints:**
- Runs every 10 minutes (`*/10 * * * *`) — frequent enough for geopolitical signals, cheap enough to run indefinitely.
- Incremental — only scrapes articles published since the last successful run. No re-scraping, no duplicate work.
- Idempotent — re-running the same interval produces the same feature-store state.
- Single active run (`max_active_runs=1`) — prevents two overlapping runs from racing on the feature store.
- Writes to the same **Feature Store** used by training, guaranteeing train/serve parity.

**DAG overview:**

```
scrape_incremental → dedup_against_store → clean_and_extract_live → upsert_features → compact_aggregates
```

##### Step 1 — `scrape_incremental`

Fetches articles published since the last successful run. Uses an Airflow Variable (`last_ingest_ts`) as a watermark so restarts don't re-scrape or skip.

```python
from airflow.decorators import task
from airflow.models import Variable
from datetime import datetime, timezone, timedelta

@task(retries=3, retry_delay=timedelta(minutes=2))
def scrape_incremental() -> list[dict]:
    watermark = Variable.get("last_ingest_ts", default_var=None)
    since = datetime.fromisoformat(watermark) if watermark else datetime.now(timezone.utc) - timedelta(minutes=15)

    articles = []
    for source in ["reuters", "bbc", "aljazeera", "guardian", "gdelt"]:
        articles.extend(fetch_source(source, since=since))

    # advance the watermark to the newest article timestamp we saw
    if articles:
        newest = max(a["published_at"] for a in articles)
        Variable.set("last_ingest_ts", newest)

    return articles
```

##### Step 2 — `dedup_against_store`

Articles often appear across multiple wires (Reuters syndicated to BBC, etc.). Dedup **against the feature store**, not just within the batch — so an article seen 9 minutes ago isn't re-processed.

```python
@task
def dedup_against_store(articles: list[dict]) -> list[dict]:
    if not articles:
        return []
    hashes = [simhash(a["text"]) for a in articles]
    seen   = redis.smismember("ingest:seen_hashes", hashes)   # pipelined check
    fresh  = [a for a, s in zip(articles, seen) if not s]
    if fresh:
        redis.sadd("ingest:seen_hashes", *[simhash(a["text"]) for a in fresh])
        redis.expire("ingest:seen_hashes", 60 * 60 * 24 * 7)  # 7-day TTL
    return fresh
```

##### Step 3 — `clean_and_extract_live`

Same transformations as the training DAG's `clean_and_extract`, but operating on a small, recent batch. Runs the identical code path / container so features are byte-identical to training features (train/serve parity).

```python
@task
def clean_and_extract_live(articles: list[dict]) -> list[dict]:
    enriched = []
    for art in articles:
        art["entities"]   = ner_model(art["text"])
        art["event_type"] = event_classifier(art["text"])
        art["tone"]       = escalation_scorer(art["text"])
        art["embedding"]  = sbert.encode(art["text"]).tolist()
        enriched.append(art)
    return enriched
```

##### Step 4 — `upsert_features`

Writes enriched records into the Feature Store (Redis / Feast / Postgres), keyed by `(region, actor_set, hour_bucket)`. Uses upsert semantics so a rerun of the same interval is safe.

**Feature Store key schema:**

```
key    = f"features:{region}:{sorted(actors)}:{hour_bucket_iso}"
value  = {
    "event_counts":     {"airstrike": 2, "diplomatic_breakdown": 1, ...},
    "tone_mean":        -0.41,
    "embedding_mean":   [0.12, -0.04, ...],   # 768-d, averaged over articles in bucket
    "article_count":    7,
    "last_updated":     "2026-04-14T09:40:00Z",
    "source_article_ids": ["reu:...", "bbc:..."]
}
TTL = 90 days
```

```python
@task
def upsert_features(enriched: list[dict]) -> int:
    bucketed = bucket_by_region_actor_hour(enriched)
    pipe = redis.pipeline()
    for key, new_features in bucketed.items():
        existing = redis.json().get(key) or empty_bucket()
        merged   = merge_bucket(existing, new_features)   # sum counts, re-average embeddings
        pipe.json().set(key, "$", merged)
        pipe.expire(key, 60 * 60 * 24 * 90)
    pipe.execute()
    return len(bucketed)
```

##### Step 5 — `compact_aggregates`

Rolls hourly buckets up into daily aggregates and maintains pre-computed "last-N-days" rolling windows keyed by `(region, actor_set)`. This is what `/predict` actually reads at request time — one O(1) lookup instead of recomputing a 30-day window on every request.

```python
@task
def compact_aggregates(bucket_count: int) -> None:
    if bucket_count == 0:
        return
    for region, actors in active_region_actor_pairs():
        for window_days in (7, 14, 30):
            window_features = compute_rolling_window(region, actors, window_days)
            redis.json().set(
                f"window:{region}:{sorted(actors)}:{window_days}d",
                "$", window_features,
            )
            redis.expire(
                f"window:{region}:{sorted(actors)}:{window_days}d",
                60 * 60 * 2,   # 2h freshness guarantee; next run refreshes
            )
```

##### Full mini-DAG wiring

```python
from airflow.decorators import dag
from datetime import datetime, timedelta

@dag(
    dag_id="news_ingest_live",
    schedule="*/10 * * * *",           # every 10 minutes
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,                 # no overlapping runs
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=1),
        "execution_timeout": timedelta(minutes=8),   # must finish before next tick
    },
    tags=["ingest", "live", "feature-store"],
)
def news_ingest_live():
    articles  = scrape_incremental()
    fresh     = dedup_against_store(articles)
    enriched  = clean_and_extract_live(fresh)
    written   = upsert_features(enriched)
    compact_aggregates(written)

news_ingest_live()
```

##### What `/predict` does at request time

```python
# FastAPI handler
@app.post("/predict")
def predict(req: PredictRequest):
    key = f"window:{req.region}:{sorted(req.actors)}:{req.horizon_days}d"
    features = redis.json().get(key)
    if features is None or stale(features):
        raise HTTPException(503, "Feature window not fresh — ingest pipeline may be down")
    return serve_model(features, horizon=req.horizon_days)
```

**No scraping, no NER, no embedding at request time** — all of that already happened in the mini-DAG. The API is a pure model inference call against pre-computed features. P99 latency stays in the tens of milliseconds.

##### Why this design

| Problem | How this solves it |
|--|--|
| Stale features degrade live forecasts | 10-min refresh cycle, `stale()` check returns 503 if ingest is lagging |
| Train/serve skew silently kills calibration | Both DAGs call the **same** `clean_and_extract` code path writing to the **same** Feature Store |
| Scraping in the hot request path | All heavy lifting is pre-computed in the mini-DAG; `/predict` is O(1) lookup + forward pass |
| Overlapping runs corrupt the store | `max_active_runs=1` + idempotent upserts |
| Ingest outage is silent | Watermark staleness + `article_count == 0` alerts fire to on-call |
| Redundant work on wire-syndicated articles | SimHash dedup against a shared store, not per-batch |

##### When to graduate to Option B (Kafka + Flink)

This mini-DAG is appropriate for **10-minute freshness**. If the product requires sub-minute latency (e.g. market-moving events), replace the mini-DAG with Kafka ingestion + a Flink job writing into the same Feature Store. The training DAG and `/predict` code stay unchanged — only the ingest path swaps. That's the value of putting the Feature Store in the middle: ingest strategy and serving strategy are decoupled.

---

### 4. Preprocessing & Feature Engineering

Transforms raw news into structured prediction features **and supervised labels**.

**Feature steps:**
- Deduplication (MinHash / SimHash on article bodies to collapse wire-syndicated duplicates)
- Language filtering (English-only in v1; translation layer optional)
- Named entity recognition (countries, leaders, organizations, weapons, locations) via spaCy / HF NER
- Event extraction (strike, blockade, troop movement, negotiation, sanction) via rule-based triggers + fine-tuned classifier
- Sentiment and escalation-tone scoring (fine-tuned transformer, 0–1 scale)
- Text → embeddings (`sentence-transformers/all-mpnet-base-v2`, 768-d)
- Time-series aggregation (per-actor / per-region event counts at hourly and daily granularity)

**Label generation (for supervised training):**
- Each `(region, actor-set, timestamp t)` tuple becomes a training example.
- Features = news signals in window `[t − N, t]` (N = context window, tuned by Optuna).
- Label = multi-hot vector over event taxonomy indicating which events actually occurred in `[t, t + H]` (H = forecast horizon).
- Ground truth comes from (a) curated Historical Conflict KB timelines and (b) event extractor outputs on the rolling news stream, with analyst review for high-impact events.

---

### 5. Historical Conflict Knowledge Base

A curated, versioned knowledge base of past conflicts and their event sequences:
- Gulf wars, Iran–Iraq war, Israel–Hezbollah 2006, Syria, Ukraine, etc.
- Each conflict stored as a timeline of typed events (buildup → trigger → strike → retaliation → escalation)

**Schema (per conflict):**

```json
{
  "conflict_id": "israel_hezbollah_2006",
  "actors": ["IL", "Hezbollah", "LB"],
  "region": "Levant",
  "timeline": [
    {"t": "2006-07-12T00:00Z", "event": "cross_border_raid", "actor": "Hezbollah"},
    {"t": "2006-07-13T00:00Z", "event": "airstrike",         "actor": "IL"},
    {"t": "2006-07-14T00:00Z", "event": "naval_blockade",    "actor": "IL"}
  ],
  "outcome": "ceasefire_2006_08_14"
}
```

**How the predictor uses it:**
- Each historical timeline is sliced into `(features_[t-N, t], label_[t, t+H])` training pairs — the KB is the **primary source of rare-event positive examples**.
- At inference, the current feature window is embedded and matched via approximate nearest neighbor (FAISS) against historical windows; the top-k analogs and their realized outcomes become both an input feature and a human-readable explanation.

---

### 6. Hyperparameter Optimization (Optuna)

Optuna searches the prediction model space automatically.

**Tuned hyperparameters:**
- Model family (Transformer time-series, Temporal Fusion Transformer, XGBoost on aggregated features, LLM-in-the-loop)
- Context window (days of history)
- Embedding size
- Learning rate, batch size
- Forecast horizon (3 / 7 / 14 days)
- Decision threshold for high-confidence alerts

**Example search:**

```
Trial 1 → Brier = 0.21, AUC = 0.78
Trial 2 → Brier = 0.17, AUC = 0.84
Trial 3 → Brier = 0.14, AUC = 0.89  (BEST)
```

---

### 7. Prediction Model Training

The predictor is trained to output **probabilities for a typed set of geopolitical events** over a fixed horizon.

**Learning paradigm:** supervised, multi-label binary classification (one sigmoid per event type) with a multi-horizon head (e.g. 3 / 7 / 14 day windows). Self-supervised components (sentence embeddings, pretrained NER) feed features but are not the predictor itself.

**Candidate model families:**
- Temporal Fusion Transformer (multi-horizon probabilistic forecasts, native quantile outputs)
- Transformer classifier over event sequences (causal-mask attention on daily event tokens)
- Gradient boosting (XGBoost / LightGBM) on engineered features — strong baseline, fast to retrain
- LLM-assisted reasoning for rare / novel patterns (retrieval over KB + structured output)

**Target event types (controlled taxonomy):**
- `naval_blockade`, `airstrike`, `missile_strike`, `ground_offensive`, `border_incursion`, `proxy_activation`, `diplomatic_breakdown`, `sanction_package`, `ceasefire`, `negotiation_resumed`

**Training data construction:**
- Sliding window over both the Historical KB and the live news stream.
- Stride = 1 day; context window N ∈ {7, 14, 30} days (Optuna-tuned).
- Negative examples: all `(region, actor-set, t)` windows where no target event occurred in `[t, t+H]` — heavily downsampled to control class imbalance.

**Class imbalance (target events are rare):**
- Positive:negative ratios typically 1:50 to 1:500 per event type.
- Handled with: focal loss (γ=2) or class-weighted BCE; hard-negative mining; stratified sampling per event type; optional SMOTE on engineered-feature models only (not on raw sequence models).

**Calibration:**
- Raw model scores are passed through **isotonic regression** fit on a time-forward holdout (never random split — leakage risk). Platt scaling is a fallback for very small positive counts.
- Calibration is re-fit every retrain cycle; the Evaluation Gate blocks deployment if Expected Calibration Error (ECE) regresses.

**Temporal validation:**
- Walk-forward splits only. Train on `[−∞, t_cut]`, validate on `[t_cut, t_cut + V]`, roll forward. Random k-fold is prohibited (would leak future into past).

---

### 8. Experiment Tracking & Registry (MLflow)

MLflow logs every run.

```python
mlflow.log_param("horizon_days", 7)
mlflow.log_metric("brier_score", 0.14)
mlflow.log_metric("auc", 0.89)
mlflow.log_artifact("predictor.pkl")
```

**Registry:**
- v1 — baseline logistic model
- v2 — Transformer + history KB
- v3 — production predictor

---

### 9. Evaluation Gate

Airflow promotes a new predictor only if it beats the current one across a metric suite, not a single number.

**Metric suite (all computed on a time-forward holdout):**

| Metric | Purpose | Gate rule |
|--|--|--|
| Brier score | Overall calibration + sharpness | `new < prod` |
| Expected Calibration Error (ECE) | Calibration only | `new ≤ prod × 1.05` |
| ROC-AUC | Discrimination | `new ≥ prod` |
| PR-AUC (per event type) | Rare-event performance (required because positives are rare) | `new ≥ prod` for each event |
| Recall @ 1% FPR | High-precision regime | `new ≥ prod − 0.02` |
| Reliability diagram | Visual sanity check, logged to MLflow | reviewed |

**Gate logic:**

```python
if (new.brier < prod.brier
        and new.ece <= prod.ece * 1.05
        and new.auc >= prod.auc
        and all(new.pr_auc[e] >= prod.pr_auc[e] for e in EVENT_TYPES)
        and new.recall_at_1pct_fpr >= prod.recall_at_1pct_fpr - 0.02):
    approve_shadow_deployment()
else:
    reject()
```

**Two-stage deployment:**
1. **Shadow** — new model runs in parallel to prod on live traffic for 7 days; predictions logged but not served. Promote only if live calibration holds.
2. **Canary** — 10% of `/predict` traffic for 48h; full rollout only if error budget is clean.

**Why calibration dominates:** a forecast of "70%" must be right ~70% of the time. An uncalibrated model with higher AUC is worse for decision-makers than a calibrated one with lower AUC.

---

### 10. Prediction API

Approved models are served via a FastAPI REST service.

**Inference pipeline (per request):**

1. **Auth & rate-limit** — API key in `Authorization: Bearer <key>`; per-key RPS + daily-quota enforcement (Redis token bucket).
2. **Feature assembly** — pull the last N days (N = model's trained context window) of preprocessed news for the given `region` + `actors`: entity counts, event counts, escalation-tone, embeddings, per-actor time series. Cached in Redis, TTL = ingest cadence.
3. **Historical analog retrieval** — embed the current feature window; FAISS lookup against the KB; top-k analogs attached to the request.
4. **Model forward pass** — promoted model (from MLflow registry) emits raw scores per event type × horizon bucket.
5. **Calibration layer** — isotonic regressor converts raw scores → calibrated probabilities.
6. **Explanation** — SHAP values (tree models) or attention weights (transformer) identify top driving signals; top-k historical analogs are formatted for the response.
7. **Response assembly** — event probabilities, window, driving signals, analogs, `model_version`, `dataset_hash`.

**Request:**

```http
POST /predict
Authorization: Bearer <api_key>
Content-Type: application/json

{
  "region": "Persian Gulf",
  "actors": ["US", "Iran"],
  "horizon_days": 5
}
```

**Response:**

```json
{
  "model_version": "predictor:v3",
  "dataset_hash": "dvc:8f4a...c21",
  "generated_at": "2026-04-14T09:12:00Z",
  "forecasts": [
    {
      "event": "naval_blockade",
      "probability": 0.72,
      "window_days": 5,
      "driving_signals": [
        {"signal": "us_fleet_movement_strait_hormuz", "contribution": 0.31},
        {"signal": "oil_tanker_rerouting_spike",      "contribution": 0.18},
        {"signal": "us_iran_diplomatic_collapse",     "contribution": 0.12}
      ],
      "historical_analogs": [
        {"conflict_id": "gulf_war_1990", "similarity": 0.81, "outcome": "blockade_imposed"},
        {"conflict_id": "iran_iraq_1987", "similarity": 0.74, "outcome": "tanker_war"}
      ]
    },
    { "event": "missile_strike",  "probability": 0.18, "window_days": 5 },
    { "event": "diplomatic_deal", "probability": 0.09, "window_days": 5 }
  ]
}
```

**Other endpoints:**
- `GET /healthz` — liveness
- `GET /model/info` — current model version, training date, dataset hash, calibration stats
- `POST /feedback` — analyst confirms or rejects a past forecast; written to the label store for the next retrain

Every response is fully auditable: `(model_version, dataset_hash)` reproduces the exact prediction.

---

### 11. Production Scaling (Kubeflow)

Kubeflow handles production runtime:
- Replicated prediction services
- GPU inference for Transformer-based predictors
- Distributed training when the historical KB grows
- Traffic-based autoscaling

---

### 12. Monitoring, Drift & Retraining Triggers

Models degrade; geopolitical semantics shift. The system monitors itself and retrains on signals, not just a cron.

**Live monitoring (Prometheus + Grafana):**
- Rolling Brier score and ECE on forecasts that have reached their horizon (backfilled once ground truth is known).
- Reliability diagram refreshed daily.
- Input-feature distribution (PSI / KL divergence vs. training distribution).
- Embedding drift: mean cosine distance of current-week article embeddings to a rolling baseline.
- API latency (p50/p95/p99), error rate, per-event prediction volume.

**Retraining triggers (any one fires a retrain DAG):**
1. **Scheduled** — weekly baseline retrain.
2. **Calibration drift** — rolling Brier exceeds prod baseline + threshold for 3 consecutive days.
3. **Data drift** — PSI > 0.25 on any top-20 feature.
4. **New conflict signal** — event extractor detects a sustained novel event cluster (e.g. new actor pair with rising activity) not in the KB.
5. **Analyst feedback volume** — N corrections via `/feedback` exceed a threshold.

**Rollback:**
- MLflow registry keeps the last 3 promoted versions hot.
- If live calibration regresses post-deploy, Airflow flips the service back to the previous version; the failed model is tagged `rejected_live`.

---

### 13. Cold Start & Rare Event Handling

Some events are genuinely rare (e.g. naval blockade — handful of historical positives); some conflicts are genuinely novel (new actor pairs, new regions).

**Rare-event strategies:**
- Class-weighted / focal loss (§7).
- Hierarchical labels — learn a broader parent class (`kinetic_action`) first, then specialize; useful when child-class positives are too sparse for a direct head.
- Historical-analog features — even if the current conflict is novel, its embedding may be close to a past one, supplying indirect supervision.
- LLM-assisted scorer — for event types with <50 historical positives, an LLM conditioned on the retrieved analogs provides a secondary probability; ensembled with the supervised model via learned weights.

**Cold-start for new actors / regions:**
- Actor embeddings are learned but fall back to a region-level + actor-type prior (e.g. "state actor in Middle East") when an actor is first seen.
- Forecasts for cold-start actors carry a wider uncertainty band in the response.

**Uncertainty quantification:**
- Deep-ensemble or MC-dropout for transformer models.
- Quantile outputs directly from TFT.
- Calibrated interval (e.g. P10–P90) reported alongside point probability for decision-critical events.

---

### 14. Failure Modes & Limitations

Honest limitations — documented so users interpret forecasts correctly.

| Failure mode | Mitigation |
|--|--|
| **Source bias** — Western wire services over-represented | Multi-source ingestion + source-diversity feature + reweighting |
| **Disinformation / propaganda** | Source-credibility score; state-media articles down-weighted; cross-source corroboration required to flip signals |
| **Echo chamber / duplicate amplification** | Aggressive near-duplicate detection (MinHash) before aggregation |
| **Survivorship bias in KB** | Only conflicts that escalated are well-documented; the system can over-predict escalation. Negative-example mining from "calm periods" in the same regions counteracts this |
| **Black swans** | Unprecedented events have no historical analog; the model will under-predict them. Outputs include an `analog_similarity_max` field — low values flag low-confidence regimes |
| **Self-fulfilling prophecy risk** | If forecasts influence markets/policy, they become part of the causal chain. Forecasts should be treated as inputs to human decision-making, not as ground truth |
| **Concept drift in language** | Weekly retrain + drift triggers (§12) |
| **Legal / ethical** | System is decision-support, not autonomous action. `/predict` is rate-limited and audited; no automated targeting or kinetic action loop is exposed |

**Out of scope (explicitly):**
- The system does not recommend actions. It outputs probabilities; humans decide.
- It is not a replacement for intelligence analysts; it is a force multiplier for them.

---

### 15. Natural-Language Explanation Layer (Small LLM)

The core predictor outputs probabilities — numbers, not prose. A **small LLM** sits at the edge of the pipeline to translate those numbers into a short human-readable briefing for analysts. It does **not** generate the probabilities.

**Model:** a small instruction-tuned model (e.g. `Llama-3.2-3B-Instruct`, `Phi-3-mini`, or `Qwen2.5-3B`), quantized (GGUF Q4) for cheap local inference. No frontier model needed — the task is narrow templating, not reasoning.

**Input to the LLM:**
- The ML model's probability JSON (events + probabilities + horizons)
- Top driving signals (SHAP contributions)
- Top-k historical analogs retrieved from the KB
- Region and actors

**Output:** a short paragraph such as:

> *"Model estimates a 72% probability of a US naval blockade on Iran within 5 days. The forecast is driven primarily by US fleet repositioning near the Strait of Hormuz (+0.31), oil-tanker rerouting (+0.18), and the collapse of US–Iran diplomatic channels (+0.12). Closest historical analogs: Gulf War 1990 pre-blockade window (similarity 0.81) and Iran–Iraq Tanker War 1987 (0.74). Confidence interval: 65–80%."*

**Guardrails:**
- The LLM is given the probabilities as *facts it may not alter*. If it changes a number, the response is rejected by a post-check (regex match against source JSON) and regenerated.
- Temperature = 0 for determinism.
- Output length capped; no speculation beyond the supplied evidence.
- Every briefing is logged with the exact prompt, model version, and source JSON for auditability.

**Why a small model, not a frontier LLM:**
- The task is templating, not reasoning — a 3B model is sufficient.
- Latency < 500ms on a single consumer GPU (or CPU with quantization).
- Cost ≈ 0 at scale; no external API dependency, no data leaving the cluster.
- Determinism is easier to enforce with a fixed local model than with an external API whose weights may change.

---

### 16. LLM vs. ML Classifier — Division of Labor

Common misconception: "an LLM would do this better." It would not, for the **core probability step**. LLMs are famously miscalibrated, non-deterministic, and unauditable — all disqualifying for this domain. But they are excellent at the **edges** of the pipeline. The system uses each where it is strongest.

| Dimension | Calibrated ML Classifier (core) | LLM (edges) |
|--|--|--|
| Calibration | "70%" ≈ 70% empirical frequency after isotonic fit | Miscalibrated; probabilities drift |
| Reproducibility | Deterministic — same input → same output | Non-deterministic without heavy guardrails |
| Auditability | `(model_version, dataset_hash)` replays the exact prediction | Opaque; hard to replay |
| Latency / cost | ~10 ms, ~$0 per call | 100 ms – seconds, non-trivial cost |
| Rare-event handling | Focal loss, class weighting, historical analogs | Poor quantitative estimates on rare events |
| Open-vocabulary events | Closed taxonomy | Can reason about novel event types |
| Natural-language output | None | Strong |
| Feature extraction from text | Rule-based + fine-tuned NER | Strong (zero-shot event extraction) |

**Division of labor in this system:**

```
┌──────────────────────────────────────────────────────────────────┐
│  Raw news                                                        │
│      │                                                           │
│      ▼                                                           │
│  [LLM / NER / event extractor]    ← LLM helps here (upstream)    │
│      │                                                           │
│      ▼                                                           │
│  Numeric feature vector                                          │
│      │                                                           │
│      ▼                                                           │
│  [Calibrated ML predictor]        ← ML CORE — no LLM here        │
│      │                                                           │
│      ▼                                                           │
│  Probabilities + driving signals + analogs (JSON)                │
│      │                                                           │
│      ▼                                                           │
│  [Small LLM explainer]            ← LLM helps here (downstream)  │
│      │                                                           │
│      ▼                                                           │
│  Human-readable briefing                                         │
└──────────────────────────────────────────────────────────────────┘
```

**Rule of thumb:**
- If the output must be a **number you can audit**, use the ML classifier.
- If the output must be **text a human can read**, use the LLM.
- Never let the LLM invent numbers; never ask the ML classifier to write prose.

---

## Prediction Outputs

Every forecast carries:

- **Event type** (typed, from a controlled taxonomy)
- **Probability** (calibrated)
- **Time window** (e.g. ≤ 5 days)
- **Driving signals** (which news features pushed the probability up)
- **Historical analogs** (which past conflicts this pattern matches)
- **Model version + dataset hash** (auditability)

---

## End-to-End System Flow

1. Airflow triggers scraping on schedule
2. News is collected across sources
3. DVC versions raw + processed datasets
4. Preprocessing extracts entities, events, and embeddings
5. Data is merged with the historical conflict KB
6. Optuna searches for the best predictor configuration
7. The predictor is trained and calibrated
8. MLflow logs metrics, artifacts, and registers the model
9. Evaluation gate approves or rejects the new model
10. Approved predictors are deployed via Docker/FastAPI
11. Kubeflow scales the service in production

---

## System Intelligence Layers

| Layer | Tool | Purpose |
|-------|------|---------|
| Data | DVC | Version news + historical KB |
| Model | Optuna | Optimize the predictor |
| Pipeline | Airflow | Automate the workflow |
| Experiment | MLflow | Track runs and register models |
| Infrastructure | Kubeflow | Scale production workloads |

---

## Real-World Enhancements

1. **Concept drift handling** — geopolitical semantics shift; models must retrain on rolling windows
2. **Streaming ingestion** — Kafka for near real-time forecasting
3. **Deduplication layer** — collapse repeated coverage of the same event before it biases signals
4. **Calibration monitoring** — track Brier score and reliability diagrams in production
5. **Human-in-the-loop review** — analysts confirm or reject high-impact forecasts to refine training data
6. **OSINT fusion (optional)** — enrich news signals with satellite, AIS ship tracking, flight data

---

## Project Impact

This pipeline turns war-news monitoring from passive reading into **forward-looking, probabilistic, auditable intelligence**:

- Forecasts escalations before they happen
- Grounds every forecast in historical precedent
- Quantifies confidence and time windows
- Self-improves as new conflicts generate new training data
- Deploys new predictors only when they measurably improve

**In one line:** the system **continuously collects war news, learns from historical conflicts, and forecasts what is likely to happen next — automatically, at scale, with calibrated probabilities.**

---

## Tech Stack

- **Orchestration:** Apache Airflow
- **Data Versioning:** DVC
- **Hyperparameter Search:** Optuna
- **Experiment Tracking & Registry:** MLflow
- **Serving:** FastAPI, Docker
- **Scaling:** Kubeflow, Kubernetes
- **Modeling:** PyTorch, Hugging Face Transformers, XGBoost / LightGBM, Temporal Fusion Transformer
- **NLP:** spaCy / Hugging Face NER, sentence-transformers
- **Streaming (optional):** Apache Kafka
- **OSINT enrichment (optional):** GDELT, AIS, satellite imagery
