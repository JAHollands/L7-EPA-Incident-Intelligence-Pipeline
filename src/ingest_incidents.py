from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path

import requests
import yaml
from minio import Minio

# Build API endpoint URL from config YAML
# Read API base URL and incident path from config.
def build_endpoint_url(cfg: dict) -> str:
    source_cfg = cfg["source"]
    api_base_url = source_cfg["api_base_url"]
    incident_path = source_cfg["incident_path"]

    # Join URL parts
    return f"{api_base_url.rstrip('/')}/{incident_path.lstrip('/')}"


# Load key/value secrets from the local env file
# Parse KEY=VALUE rows and ignore comments
def load_env_file(path: Path) -> dict:
    # Fail fast with a clear path if the file is missing
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


# Fetch all incident records with pagination
# Keep requesting pages until max records or end of data
def fetch_incident_pages(cfg: dict, timeout: int = 30) -> tuple[list[dict], int]:
    # Pull paging settings and runtime flags from config.
    source_cfg = cfg["source"]
    runtime_cfg = cfg.get("runtime", {})
    page_size = int(source_cfg["page_size"])
    max_records = int(source_cfg["max_records"])
    use_env_proxy = bool(runtime_cfg.get("use_env_proxy", False))
    endpoint_url = build_endpoint_url(cfg)

    # Reuse one HTTP session for all page requests.
    session = requests.Session()
    session.trust_env = use_env_proxy

    pages: list[dict] = []
    offset = 0
    total_records = 0

    while total_records < max_records:
        remaining = max_records - total_records
        request_limit = min(page_size, remaining)
        params = {
            "sysparm_limit": request_limit,
            "sysparm_offset": offset,
            "sysparm_display_value": "true",
        }

        response = session.get(endpoint_url, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        batch = payload.get("result", [])

        pages.append(payload)
        batch_count = len(batch)

        # A zero size page means there is nothing left to read.
        if batch_count == 0:
            break

        total_records += batch_count

        # A short page means the source has no more record
        if batch_count < request_limit:
            break

        offset += request_limit

    return pages, total_records


# Save raw JSON pages to bronze storage
# Partition each run into its own timestamp folder.
def save_raw_pages_to_bronze(
    pages: list[dict],
    endpoint_url: str,
    total_records: int,
    minio_client: Minio,
    bucket: str,
    prefix_root: str,
) -> str:
    # Build the run folder once and reuse it for all files
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_prefix = f"{prefix_root.rstrip('/')}/incidents_raw/run_ts={run_id}"

    # Write each page file with stable names like incidents_raw_page_001.json.
    for idx, payload in enumerate(pages, start=1):
        object_name = f"{run_prefix}/incidents_raw_page_{idx:03d}.json"
        page_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        minio_client.put_object(
            bucket_name=bucket,
            object_name=object_name,
            data=io.BytesIO(page_bytes),
            length=len(page_bytes),
            content_type="application/json",
        )

    # Write a manifest for record counts and source metadata.
    manifest = {
        "endpoint_url": endpoint_url,
        "run_id": run_id,
        "page_count": len(pages),
        "record_count": total_records,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "format": "raw_json_pages",
        "storage": {"bucket": bucket, "prefix": run_prefix},
    }

    manifest_name = f"{run_prefix}/manifest.json"
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")

    minio_client.put_object(
        bucket_name=bucket,
        object_name=manifest_name,
        data=io.BytesIO(manifest_bytes),
        length=len(manifest_bytes),
        content_type="application/json",
    )

    return run_prefix


# Run the ingestion proces
# Load config and credentials, fetch pages, then write to storage
def run_ingestion(config_path: str = "config/config.yaml") -> tuple[str, int, str]:
    # Resolve default paths from repository root so this is portable.
    repo_root = Path(__file__).resolve().parents[1]
    config_file = Path(config_path) if Path(config_path).is_absolute() else repo_root / config_path

    with config_file.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Read MinIO connection and target settings from config.
    minio_cfg = cfg["storage"]["minio"]
    minio_endpoint = os.getenv("MINIO_ENDPOINT", minio_cfg["endpoint"])
    minio_secure = bool(minio_cfg.get("secure", False))
    bronze_bucket = minio_cfg["bucket"]
    prefix_root = minio_cfg.get("prefix_root", "bronze")
    env_path = minio_cfg.get("env_file", "docker/.env")
    env_file = Path(env_path) if Path(env_path).is_absolute() else repo_root / env_path

    env = load_env_file(env_file)

    # Create authd client with credentials from docker/.env.
    minio_client = Minio(
        minio_endpoint,
        access_key=env["MINIO_ROOT_USER"],
        secret_key=env["MINIO_ROOT_PASSWORD"],
        secure=minio_secure,
    )

    endpoint_url = build_endpoint_url(cfg)
    pages, record_count = fetch_incident_pages(cfg=cfg)

    run_prefix = save_raw_pages_to_bronze(
        pages=pages,
        endpoint_url=endpoint_url,
        total_records=record_count,
        minio_client=minio_client,
        bucket=bronze_bucket,
        prefix_root=prefix_root,
    )

    output_uri = f"s3://{bronze_bucket}/{run_prefix}"
    return output_uri, record_count, endpoint_url


if __name__ == "__main__":
    try:
        output_path, record_count, endpoint_url = run_ingestion()
        print(f"Successfully ingested {record_count} records from endpoint {endpoint_url}")
        print(f"Bronze raw JSON written to: {output_path}")
    except Exception as exc:
        print(f"Ingestion pipeline failed: {exc}")
        raise
