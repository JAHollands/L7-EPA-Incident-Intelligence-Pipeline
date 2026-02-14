flowchart LR
  subgraph Source
    API[Incident API (FastAPI)]
  end

  subgraph Platform[Local Cloud Simulator (Docker Compose)]
    Orchestrator[Orchestrator (Prefect/Airflow)]
    Store[(Object Storage: MinIO or Azurite)]
    Meta[(Postgres: Run state + metadata)]
    MLflow[(MLflow: Tracking + Model Registry)]
    Monitor[Drift Reports (Evidently HTML)]
  end

  subgraph Data[Medallion]
    Bronze[Bronze: Raw JSON by run_id/date]
    Silver[Silver: Curated Parquet (upserted)]
    Gold[Gold: Feature sets per task]
    Scores[Scores: Predictions + recommendations]
  end

  API --> Orchestrator
  Orchestrator --> Store
  Orchestrator --> Meta

  Store --> Bronze --> Silver --> Gold
  Gold --> MLflow
  MLflow --> Scores
  Gold --> Monitor
  Scores --> Store
