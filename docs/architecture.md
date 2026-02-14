flowchart LR
  %% ============== Source ==============
  subgraph Source
    API[Incident API (FastAPI)]
  end

  %% ============== Platform ==============
  subgraph Platform["Local Cloud Simulator (Docker Compose)"]
    ORCH[Orchestrator (Airflow)]
    OBJ[(Object Storage: MinIO)]
    META[(Postgres: run state + metadata)]
    MLF[(MLflow: tracking + model registry)]
    DRIFT[Drift reports (Evidently HTML)]
  end

  %% ============== Data ==============
  subgraph Data["Medallion"]
    BRZ[Bronze: raw JSON by run_id/date]
    SLV[Silver: curated Parquet (upserted)]
    GLD[Gold: feature sets per task]
    SCR[Scores: predictions + recommendations]
  end

  API --> ORCH
  ORCH --> OBJ
  ORCH --> META

  OBJ --> BRZ --> SLV --> GLD
  GLD --> MLF --> SCR
  GLD --> DRIFT
  SCR --> OBJ
