from datetime import datetime, timezone
import json
from pathlib import Path

import requests
import yaml

# Build API endpoint URL from config YAML
def build_endpoint_url(cfg: dict) -> str:
    source_cfg = cfg["source"]
    api_base_url = source_cfg["api_base_url"]
    incident_path = source_cfg["incident_path"]
    return f"{api_base_url.rstrip('/')}/{incident_path.lstrip('/')}"

# Fetch all incident records with pagination
def fetch_incident_pages(cfg: dict, timeout: int = 30) -> tuple[list[dict], int]:
    source_cfg = cfg["source"]
    runtime_cfg = cfg.get("runtime", {})
    page_size = int(source_cfg["page_size"])
    max_records = int(source_cfg["max_records"])
    use_env_proxy = bool(runtime_cfg.get("use_env_proxy", False))
    endpoint_url = build_endpoint_url(cfg)

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

        if batch_count == 0:
            break

        total_records += batch_count

        # Safe for this API: a short page indicates end-of-data.
        if batch_count < request_limit:
            break

        offset += request_limit

    return pages, total_records

# Save the raw JSON to bronze storage, including a manifest file with metadata
def save_raw_pages_to_bronze(pages: list[dict], cfg: dict, endpoint_url: str, total_records: int) -> Path:
    bronze_root = Path(cfg.get("storage", {}).get("bronze", ".local/data/bronze"))
    bronze_root.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = bronze_root / f"incidents_raw_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)

    for idx, payload in enumerate(pages, start=1):
        page_file = run_dir / f"page_{idx:04d}.json"
        page_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "endpoint_url": endpoint_url,
        "run_id": run_id,
        "page_count": len(pages),
        "record_count": total_records,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "format": "raw_json_pages",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return run_dir

# Run the ingestion process
def run_ingestion(config_path: str = "config/config.yaml") -> tuple[Path, int, str]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    endpoint_url = build_endpoint_url(cfg)
    pages, record_count = fetch_incident_pages(cfg=cfg)
    output_path = save_raw_pages_to_bronze(
        pages=pages,
        cfg=cfg,
        endpoint_url=endpoint_url,
        total_records=record_count,
    )
    return output_path, record_count, endpoint_url


if __name__ == "__main__":
    try:
        output_path, record_count, endpoint_url = run_ingestion()
        print(f"Successfully ingested {record_count} records from endpoint {endpoint_url}")
        print(f"Bronze raw JSON written to: {output_path}")
    except Exception as exc:
        print(f"Ingestion pipeline failed: {exc}")
        raise