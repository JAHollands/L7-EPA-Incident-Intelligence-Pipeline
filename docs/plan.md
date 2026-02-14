# L7 EPA Project Plan
Incident Intelligence Pipeline (Local “Cloud Simulator” Deployment)

## 0) How we will use this plan
- This file is the single source of truth.
- Work is tracked via checkboxes. Tick items as they are completed.
- After each work session:
  - commit changes
  - update this plan (tick tasks, add notes, add next actions)
- Keep evidence as you go. Do not leave evidence capture to the end.

### Definition of Done (DoD)
A deliverable is “done” when:
- Code exists, runs end-to-end, and is committed
- Outputs are persisted (data, metrics, reports) in the correct location
- A short doc update exists explaining what changed and why
- Evidence is captured (logs, screenshots, artefacts)

---

## 1) Outcomes (project deliverables)
### Data + Analytics outcomes
1. SLA breach risk prediction (classification)
2. Time-to-resolution prediction (regression)
3. Incident clustering (unsupervised)
4. NLP triage: predict assignment group and/or category from text
5. NLP system/team: predict impacted system (or similar field) and/or team allocation from text + metadata

### Engineering outcomes
- Production-like local deployment using Docker Compose
- Orchestrated runs using Airflow (UI + scheduled DAG)
- Incremental ingestion behaviour (Option A: refresh open incidents each run)
- Model tracking, versioning, and comparison across runs (MLflow)
- Drift reporting over time (Evidently)

---

## 2) Scope boundary
### In scope
- Local “cloud simulator” stack that mimics production patterns:
  - object storage
  - orchestration and scheduling
  - run state and metadata
  - experiment tracking and model registry
  - repeatable deployment and reproducible runs
- Batch pipeline (scheduled)
- ML suite (4 modelling tasks + clustering) with evaluation + reporting
- Evidence capture and documentation suitable for the EPA project report

### Out of scope
- True cloud hosting on Azure with paid services
- High availability, disaster recovery, autoscaling
- Enterprise IAM, managed secrets, private networking
- Real-time streaming (batch runs at a reasonable cadence is enough)

---

## 3) Target architecture (local “cloud simulator”)
### Stack (all free, Docker Compose)
- Airflow (orchestration + UI)
- MinIO (S3-compatible object storage)
- Postgres (Airflow metadata, MLflow backend store)
- MLflow (tracking + model registry)
- Evidently (drift reports saved as HTML)
- FastAPI (incident API, already built)
- Python pipeline code (ingestion, transforms, ML)

### Medallion
- Bronze: raw JSON per run
- Silver: curated, typed, flattened incident table (upserted)
- Gold: feature datasets per task (SLA, TTR, clustering, NLP)
- Scores: batch predictions and recommendations for open incidents

### Diagram
Maintain the rendered Mermaid diagram in `docs/architecture.md`.
Ensure Mermaid blocks are fenced with:
```mermaid
...
