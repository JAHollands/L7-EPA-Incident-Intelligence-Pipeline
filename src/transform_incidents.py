import io
import json
from pathlib import Path

import pandas as pd
import yaml
from minio import Minio


# Load secrets from the env file
def load_env_file(path: Path) -> dict:
    # Fail fast if the file is missing
    if not path.exists():
        raise FileNotFoundError(f"Missing env file: {path.resolve()}")

    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()

    return env


# Find the latest bronze run by manifest path
# Return the selected run folder, run id, and page object keys
def get_latest_bronze_run(client: Minio, bucket: str, prefix_root: str) -> tuple[str, str, list[str]]:
    bronze_prefix = f"{prefix_root.rstrip('/')}/incidents_raw/"
    manifest_objects = [
        obj for obj in client.list_objects(bucket, prefix=bronze_prefix, recursive=True)
        if obj.object_name.endswith("manifest.json")
    ]

    if not manifest_objects:
        raise RuntimeError("No bronze manifest files found.")

    latest_manifest_key = max(obj.object_name for obj in manifest_objects)
    latest_run_prefix = latest_manifest_key.rsplit("/", 1)[0]
    bronze_run_id = latest_run_prefix.split("run_ts=")[-1]

    page_keys = sorted(
        obj.object_name
        for obj in client.list_objects(bucket, prefix=f"{latest_run_prefix}/", recursive=True)
        if obj.object_name.endswith(".json") and "incidents_raw_page_" in obj.object_name
    )

    if not page_keys:
        raise RuntimeError(f"No bronze page files found for run: {latest_run_prefix}")

    return latest_run_prefix, bronze_run_id, page_keys


# Read all bronze pages and combine records
# Merge each payload result into one list
def read_bronze_rows(client: Minio, bucket: str, page_keys: list[str]) -> list[dict]:
    bronze_rows: list[dict] = []

    for key in page_keys:
        resp = client.get_object(bucket, key)
        try:
            payload = json.loads(resp.read().decode("utf-8"))
        finally:
            resp.close()
            resp.release_conn()

        bronze_rows.extend(payload.get("result", []))

    return bronze_rows


# Flatten nested cols to display_value
# Keep nested cols unchanged
def flatten_bronze_rows(bronze_rows: list[dict]) -> pd.DataFrame:
    df_bronze = pd.DataFrame(bronze_rows)
    if df_bronze.empty:
        raise RuntimeError("Bronze run contains zero records.")

    def _extract_display_value(value):
        if isinstance(value, dict):
            if "display_value" in value:
                return value.get("display_value")
            return value
        return value

    nested_cols = [
        col for col in df_bronze.columns
        if df_bronze[col].map(lambda value: isinstance(value, dict)).any()
    ]

    df_bronze_flat = df_bronze.copy()
    for col in nested_cols:
        df_bronze_flat[col] = df_bronze_flat[col].map(_extract_display_value)

    return df_bronze_flat


# Build the silver layer from flattened bronze data
# Validate required columns and normalise datatypes
def build_silver_new(df_bronze_flat: pd.DataFrame, bronze_run_id: str) -> pd.DataFrame:
    silver_new = df_bronze_flat.copy()

    required_cols = ["sys_id", "sys_updated_on"]
    missing_required = [col for col in required_cols if col not in silver_new.columns]
    if missing_required:
        raise RuntimeError(f"Missing required column(s) for silver upsert: {missing_required}")

    datetime_cols = [
        "sys_updated_on",
        "opened_at",
        "resolved_at",
        "closed_at",
        "sys_created_on",
        "due_date",
        "activity_due",
    ]
    for dt_col in datetime_cols:
        if dt_col in silver_new.columns:
            silver_new[dt_col] = pd.to_datetime(
                silver_new[dt_col],
                errors="coerce",
                utc=True,
                dayfirst=True,
            )

    if "active" in silver_new.columns:
        silver_new["active"] = (
            silver_new["active"].astype(str).str.lower().map({"true": True, "false": False})
        )

    silver_new["bronze_run_id"] = bronze_run_id
    silver_new["ingested_at_utc"] = pd.Timestamp.now(tz="UTC")

    return silver_new


# Read current silver from storage if present
# Fallback to empty frame with the same columns as new data
def load_existing_silver(client: Minio, bucket: str, object_name: str, columns: pd.Index) -> pd.DataFrame:
    try:
        resp = client.get_object(bucket, object_name)
        try:
            return pd.read_parquet(io.BytesIO(resp.read()))
        finally:
            resp.close()
            resp.release_conn()
    except Exception:
        return pd.DataFrame(columns=columns)


# Union current and new data with latest wins approach
# Keep one row per sys_id based on sys_updated_on
def upsert_silver(silver_existing: pd.DataFrame, silver_new: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([silver_existing, silver_new], ignore_index=True)
    combined["_sort_ts"] = combined["sys_updated_on"].fillna(pd.Timestamp("1970-01-01", tz="UTC"))

    return (
        combined
        .sort_values("_sort_ts")
        .drop_duplicates(subset=["sys_id"], keep="last")
        .drop(columns=["_sort_ts"])
        .reset_index(drop=True)
    )


# Write final silver parquet to object storage
# Overwrite single silver object for simplicity
def write_silver(client: Minio, bucket: str, object_name: str, silver_final: pd.DataFrame) -> None:
    buffer = io.BytesIO()
    silver_final.to_parquet(buffer, index=False)
    buffer.seek(0)

    client.put_object(
        bucket_name=bucket,
        object_name=object_name,
        data=buffer,
        length=buffer.getbuffer().nbytes,
        content_type="application/octet-stream",
    )


# Run the transformation pipeline
# Load config and credentials, transform bronze into silver, then write to storage
def run_transformation(config_path: str = "config/config.yaml") -> tuple[str, int, int]:
    # Resolve default paths from repository root so this is portable.
    repo_root = Path(__file__).resolve().parents[1]
    config_file = Path(config_path) if Path(config_path).is_absolute() else repo_root / config_path

    with config_file.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Read storage connection and target settings from config.
    minio_cfg = cfg["storage"]["minio"]
    minio_endpoint = minio_cfg["endpoint"]
    minio_secure = bool(minio_cfg.get("secure", False))
    bucket = minio_cfg["bucket"]
    prefix_root = minio_cfg.get("prefix_root", "bronze")
    env_path = minio_cfg.get("env_file", "docker/.env")
    env_file = Path(env_path) if Path(env_path).is_absolute() else repo_root / env_path
    silver_object = "silver/incidents/incidents.parquet"

    env = load_env_file(env_file)

    # Create authd client with credentials from docker/.env.
    minio_client = Minio(
        minio_endpoint,
        access_key=env["MINIO_ROOT_USER"],
        secret_key=env["MINIO_ROOT_PASSWORD"],
        secure=minio_secure,
    )

    _, bronze_run_id, page_keys = get_latest_bronze_run(
        client=minio_client,
        bucket=bucket,
        prefix_root=prefix_root,
    )
    bronze_rows = read_bronze_rows(client=minio_client, bucket=bucket, page_keys=page_keys)
    df_bronze_flat = flatten_bronze_rows(bronze_rows=bronze_rows)
    silver_new = build_silver_new(df_bronze_flat=df_bronze_flat, bronze_run_id=bronze_run_id)
    silver_existing = load_existing_silver(
        client=minio_client,
        bucket=bucket,
        object_name=silver_object,
        columns=silver_new.columns,
    )
    silver_final = upsert_silver(silver_existing=silver_existing, silver_new=silver_new)
    write_silver(
        client=minio_client,
        bucket=bucket,
        object_name=silver_object,
        silver_final=silver_final,
    )

    output_uri = f"s3://{bucket}/{silver_object}"
    return output_uri, len(silver_new), len(silver_final)


if __name__ == "__main__":
    try:
        output_path, new_rows, final_rows = run_transformation()
        print(f"Successfully transformed {new_rows} records into silver layer")
        print(f"Silver parquet written to: {output_path} (final rows: {final_rows})")
    except Exception as exc:
        print(f"Transformation pipeline failed: {exc}")
        raise
