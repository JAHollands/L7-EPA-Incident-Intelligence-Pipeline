# Incident Intelligence Pipeline (Level 7 EPA Project)

An end-to-end data engineering and machine learning pipeline built around an incident management workflow.
The solution ingests incident records from an incident management system API, builds a local bronze/silver/gold data lake in Parquet, trains and evaluates multiple ML models, and surfaces operational and model insights via a local visualisation layer.

The implementation is designed to be source-system agnostic and API-driven, so the same pattern can be repointed to different incident management platforms with minimal change.

---

## Objectives

### Business problem
Incident management teams need earlier indicators of risk and workload so they can:
- identify incidents likely to breach SLA before they breach
- predict likely time-to-resolution to support planning and communications
- understand common incident archetypes and recurring themes
- improve triage quality by suggesting assignment groups and categories from incident details

### ML outcomes
This pipeline targets four ML capabilities:
1. **SLA breach risk classifier**
   Predict probability of breach based on early incident characteristics.
2. **Time-to-resolution regressor**
   Predict expected resolution duration (continuous).
3. **Incident clustering (unsupervised)**
   Group incidents into archetypes to highlight patterns and areas for prevention.
4. **Triage suggestion from incident descriptions (NLP)**
   Predict suggested category / assignment group based on text fields.

---

## Target architecture (conceptual)

**Source: Incident management system API**
- REST API providing paged incident records
- Reference/lookup fields and choice-like fields supported
- Data is privacy-safe (anonymised/representative) while remaining schema-accurate and suitable for model development

**Data Engineering (ELT)**
- **Bronze:** raw API pulls stored as JSON (as received)
- **Silver:** curated tabular dataset (flattened references, typed fields, cleaned values)
- **Gold:** ML-ready feature datasets (training and scoring views)

**Machine Learning**
- Feature engineering
- Training and model selection
- Evaluation (metrics and diagnostics)
- Persisted artefacts and evaluation reports

**Visualisation**
- Local dashboard(s) and model insights reporting:
  - incident operational trends
  - model performance (classification + regression metrics)
  - cluster summaries and explainability artefacts

**Cross-cutting**
- Version control (Git)
- Configuration-driven execution (`config.yaml`)
- Containerised runtime (Docker) for reproducibility and portability

---

## Repository structure (proposed)
.
├── README.md
├── config/
│ ├── config.yaml
│ └── schema_mapping.yaml
├── data/
│ ├── bronze/ # raw API pulls (JSON, partitioned by run/date)
│ ├── silver/ # curated parquet tables
│ ├── gold/ # feature-ready datasets / marts
│ └── model/ # model artefacts + evaluation reports
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_feature_prototyping.ipynb
│ ├── 03_model_experiments.ipynb
│ └── 04_evaluation_review.ipynb
├── src/
│ ├── ingestion/
│ │ └── ingest_incidents.py
│ ├── transform/
│ │ └── transform_incidents.py
│ ├── features/
│ │ └── build_features.py
│ ├── train/
│ │ └── train_models.py
│ ├── evaluate/
│ │ └── evaluate_models.py
│ └── utils/
│ ├── io.py
│ ├── time_features.py
│ └── logging.py
├── dashboards/
│ └── incident_intelligence_dashboard/
├── docker/
│ ├── Dockerfile
│ └── docker-compose.yaml
└── docs/
├── architecture.md
├── development_lifecycle.md
└── decisions.md


---

## Model evaluation

Evaluation is reported in a way suitable for operational use:
- **Classifier:** ROC AUC, Precision/Recall, F1, confusion matrix, calibration
- **Regressor:** MAE, RMSE, residual analysis
- **Clustering:** cluster profiling and exemplar incidents (with supporting internal metrics)
- **NLP triage:** top-k accuracy and error analysis by category / assignment group

---

## Roadmap

- [ ] Source ingestion (paged pulls, incremental runs)
- [ ] Bronze storage (raw JSON)
- [ ] Silver transformation (flatten + clean + curate)
- [ ] Gold feature datasets (train/scoring views)
- [ ] Train models (classification/regression/clustering/NLP)
- [ ] Evaluation reports and artefact persistence
- [ ] Visualisation layer (ops + model insights)
- [ ] Final documentation aligned to EPA requirements

