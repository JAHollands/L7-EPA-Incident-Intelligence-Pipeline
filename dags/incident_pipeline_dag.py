from __future__ import annotations

from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

# Access project repo files via /airflow/project which is mounted into the container in the docker image
PROJECT_DIR = "/opt/airflow/project"

default_args = {"owner": "incident-pipeline"}

# Orchestrates bronze to silver data ingestion and transformation
with DAG(
    dag_id="incident_pipeline_bronze_to_silver",
    default_args=default_args,
    start_date=datetime(2026, 2, 1),
    schedule=None,        # manual trigger
    catchup=False,        # don't backfill past runs
    tags=["incident", "bronze", "silver"], # metadata tags for airflow ui
) as dag:
    ingest_bronze = BashOperator(
        task_id="ingest_bronze",
        bash_command=(
            # Fail fast and show errors clearly in logs
            "set -euo pipefail; "
            # Move into the mounted repo folder
            f"cd {PROJECT_DIR}; "
            # Minimal dependency install in the container environment
            "python -m pip install -q requests pyyaml minio; "
            # Run the ingestion script to get data from the API endpoint and write to bronze storage
            "python src/ingest_incidents.py"
        ),
    )

    transform_silver = BashOperator(
        task_id="transform_silver",
        bash_command=(
            "set -euo pipefail; "
            f"cd {PROJECT_DIR}; "
            # pandas for parquet writing; minio + yaml for stroage and config
            "python -m pip install -q pandas pyyaml minio; "
            # Run the transformation script to read bronze data, transform it, and write to silver storage
            "python src/transform_incidents.py"
        ),
    )

    # Dependencies: transform only runs if ingest was successful
    ingest_bronze >> transform_silver
