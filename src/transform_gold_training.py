import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from minio import Minio
from sklearn.model_selection import train_test_split


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


# Read silver parquet from storage
def load_silver(client: Minio, bucket: str, object_name: str) -> pd.DataFrame:
    resp = client.get_object(bucket, object_name)
    try:
        return pd.read_parquet(io.BytesIO(resp.read()))
    finally:
        resp.close()
        resp.release_conn()


# Apply training quality filters
def filter_silver_for_training(silver_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # Define required columns and check they are present
    required_cols = [
        "sys_id",
        "sys_updated_on",
        "short_description",
        "description",
        "active",
        "state",
        "assignment_group",
    ]
    missing_cols = [col for col in required_cols if col not in silver_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Count open/closed and active true/false, then keep only closed/completed incidents
    state_norm = silver_df["state"].astype(str).str.strip().str.lower()

    keep_mask = state_norm.isin(["closed", "completed"])
    silver_filtered_df = silver_df.loc[keep_mask].copy()

    # Remove duplicate incidents by sys_id
    duplicate_mask = silver_filtered_df.duplicated(subset=["sys_id"], keep="first")
    silver_filtered_df = silver_filtered_df.loc[~duplicate_mask].copy()

    # Remove rows with null/blank values in required training columns
    required_view = silver_filtered_df[required_cols].copy()
    null_or_blank = required_view.isna()

    # Treat whitespace only strings as null/ blank
    for col in required_cols:
        null_or_blank[col] = null_or_blank[col] | required_view[col].astype(str).str.strip().eq("")

    rows_with_missing_required = null_or_blank.any(axis=1)
    silver_filtered_df = silver_filtered_df.loc[~rows_with_missing_required].copy()

    return silver_filtered_df, required_cols


# Build model text feature and clean/fold labels
def build_features_and_labels(
    silver_filtered_df: pd.DataFrame,
    min_class_count: int = 20,
    manual_review_group: str = "manual_review_group",
) -> tuple[pd.DataFrame, list[str], dict, dict, int, int]:
    # Work on a copy so we keep the filtered source frame as is for reference
    df = silver_filtered_df.copy()

    # Build the final text feature from both the short and long ticket content
    df["text"] = (
        df["short_description"].fillna("").astype(str).str.strip()
        + "\n"
        + df["description"].fillna("").astype(str).str.strip()
    ).str.strip()

    # Fold very small classes into a manual review bucket
    df["label_clean"] = df["assignment_group"].astype(str).str.strip()
    counts = df["label_clean"].value_counts()
    rare = counts[counts < min_class_count]

    # Build final label column used for training
    df["label_final"] = df["label_clean"].where(
        ~df["label_clean"].isin(rare.index),
        manual_review_group,
    )
    # Create mappings to convert labels to integer ids for modeling and splits
    classes = sorted(df["label_final"].unique().tolist())
    label_to_id = {c: i for i, c in enumerate(classes)}
    id_to_label = {i: c for c, i in label_to_id.items()}
    df["label_id"] = df["label_final"].map(label_to_id).astype(int)
    rare_class_count = int(len(rare))
    rare_mapped_row_count = int((df["label_final"] == manual_review_group).sum())
    return df, classes, label_to_id, id_to_label, rare_class_count, rare_mapped_row_count


# Create train/validation/test splits
def split_gold_dataset(
    df: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Hold out 10% test set on label_id to preserve class distribution
    train_val, test_df = train_test_split(
        df,
        test_size=0.10,
        random_state=random_state,
        stratify=df["label_id"],
    )

    # 10% for validation from the remaining 90%
    valid_rel = 0.10 / 0.90
    train_df, valid_df = train_test_split(
        train_val,
        test_size=valid_rel,
        random_state=random_state,
        stratify=train_val["label_id"],
    )

    # Check for leakage by unique sys_id field across each split
    overlap = (
        (set(train_df["sys_id"]) & set(valid_df["sys_id"]))
        | (set(train_df["sys_id"]) & set(test_df["sys_id"]))
        | (set(valid_df["sys_id"]) & set(test_df["sys_id"]))
    )
    if overlap:
        raise RuntimeError("Found overlapping sys_id values across train/valid/test splits.")
    return train_df, valid_df, test_df


# Write gold training artifacts to storage
def write_gold_artifacts(
    client: Minio,
    bucket: str,
    gold_training_prefix: str,
    cols: list[str],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_mapping: dict,
) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

    # Save minimal modeling columns for each split
    for name, split_df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        buf = io.BytesIO()
        split_df[cols].to_parquet(buf, index=False)
        split_bytes = buf.getvalue()

        object_name = f"{gold_training_prefix}/{name}.parquet"
        client.put_object(
            bucket_name=bucket,
            object_name=object_name,
            data=io.BytesIO(split_bytes),
            length=len(split_bytes),
            content_type="application/octet-stream",
        )

    # Output label mapping so training/inference use the same class ids
    label_mapping_bytes = json.dumps(label_mapping, indent=2).encode("utf-8")
    label_mapping_object = f"{gold_training_prefix}/label_mapping.json"
    client.put_object(
        bucket_name=bucket,
        object_name=label_mapping_object,
        data=io.BytesIO(label_mapping_bytes),
        length=len(label_mapping_bytes),
        content_type="application/json",
    )


# Write a dataset card for feature/label/split/inference docs
def write_dataset_card(
    client: Minio,
    bucket: str,
    gold_training_prefix: str,
    dataset_version: str,
    silver_bucket: str,
    silver_object: str,
    created_utc: str,
    total_rows: int,
    train_rows: int,
    valid_rows: int,
    test_rows: int,
    class_count: int,
    rare_class_count: int,
    rare_mapped_row_count: int,
    random_state: int,
    min_class_count: int,
    manual_review_group: str,
    id_to_label: dict,
) -> None:
    label_preview = sorted(id_to_label.items(), key=lambda x: x[0])[:20]
    preview_lines = "\n".join([f"| {label_id} | {label_name} |" for label_id, label_name in label_preview])

    dataset_card = f"""# Dataset card

## Overview
- dataset_version: {dataset_version}
- source_silver_key: s3://{silver_bucket}/{silver_object}
- created_utc: {created_utc}
- rows_total: {total_rows}
- split_sizes: train={train_rows}, valid={valid_rows}, test={test_rows}
- class_count: {class_count}
- rare_class_count: {rare_class_count}
- rare_mapped_count: {rare_mapped_row_count}

## Features (Feature dictionary)
- text = short_description + "\\n" + description (fillna -> ""; strip)

## Label (Label dictionary)
- source field: assignment_group
- clean rule: strip whitespace
- rare class policy: labels with < {min_class_count} records -> {manual_review_group}
- full mapping lives in label_mapping.json

### Label preview (first 20)
| label_id | label_name |
|---:|---|
{preview_lines}

## Split rules
- 80/10/10 stratified split on label_id
- random_state: {random_state}

## Inference input contract
- required: at least one of short_description or description
- preprocessing: text = short_description + "\\n" + description (fillna -> ""; strip)
"""

    dataset_card_bytes = dataset_card.encode("utf-8")
    dataset_card_object = f"{gold_training_prefix}/dataset_card.md"
    client.put_object(
        bucket_name=bucket,
        object_name=dataset_card_object,
        data=io.BytesIO(dataset_card_bytes),
        length=len(dataset_card_bytes),
        content_type="text/markdown",
    )


# Run gold transformation pipeline
def run_gold_transformation(config_path: str = "config/config.yaml") -> tuple[str, int, int, int]:
    repo_root = Path(__file__).resolve().parents[1]
    config_file = Path(config_path) if Path(config_path).is_absolute() else repo_root / config_path

    with config_file.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    minio_cfg = cfg["storage"]["minio"]
    minio_endpoint = os.getenv("MINIO_ENDPOINT", minio_cfg["endpoint"])
    minio_secure = bool(minio_cfg.get("secure", False))
    bucket = minio_cfg["bucket"]
    env_path = minio_cfg.get("env_file", "docker/.env")
    env_file = Path(env_path) if Path(env_path).is_absolute() else repo_root / env_path

    env = load_env_file(env_file)
    client = Minio(
        minio_endpoint,
        access_key=env["MINIO_ROOT_USER"],
        secret_key=env["MINIO_ROOT_PASSWORD"],
        secure=minio_secure,
    )

    silver_object = "silver/incidents/incidents.parquet"
    silver_df = load_silver(client=client, bucket=bucket, object_name=silver_object)

    silver_filtered_df, _ = filter_silver_for_training(silver_df)
    min_class_count = 20
    manual_review_group = "manual_review_group"
    (
        df,
        classes,
        label_to_id,
        id_to_label,
        rare_class_count,
        rare_mapped_row_count,
    ) = build_features_and_labels(
        silver_filtered_df,
        min_class_count=min_class_count,
        manual_review_group=manual_review_group,
    )

    random_state = 42
    train_df, valid_df, test_df = split_gold_dataset(df=df, random_state=random_state)

    dataset_version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    gold_training_prefix = f"gold/training/{dataset_version}"

    cols = ["sys_id", "sys_updated_on", "text", "label_final", "label_id"]
    label_mapping = {
        "dataset_version": dataset_version,
        "label_field_source": "assignment_group",
        "label_field_final": "label_final",
        "manual_review_group": manual_review_group,
        "min_class_count": min_class_count,
        "classes": classes,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
    }

    write_gold_artifacts(
        client=client,
        bucket=bucket,
        gold_training_prefix=gold_training_prefix,
        cols=cols,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        label_mapping=label_mapping,
    )

    write_dataset_card(
        client=client,
        bucket=bucket,
        gold_training_prefix=gold_training_prefix,
        dataset_version=dataset_version,
        silver_bucket=bucket,
        silver_object=silver_object,
        created_utc=datetime.now(timezone.utc).isoformat(),
        total_rows=len(df),
        train_rows=len(train_df),
        valid_rows=len(valid_df),
        test_rows=len(test_df),
        class_count=len(classes),
        rare_class_count=rare_class_count,
        rare_mapped_row_count=rare_mapped_row_count,
        random_state=random_state,
        min_class_count=min_class_count,
        manual_review_group=manual_review_group,
        id_to_label=id_to_label,
    )

    output_uri = f"s3://{bucket}/{gold_training_prefix}"
    return output_uri, len(train_df), len(valid_df), len(test_df)


if __name__ == "__main__":
    try:
        output_path, train_rows, valid_rows, test_rows = run_gold_transformation()
        print(f"Gold training artifacts written to: {output_path}")
        print(
            f"Split rows - train: {train_rows}, valid: {valid_rows}, test: {test_rows}"
        )
    except Exception as exc:
        print(f"Gold transformation pipeline failed: {exc}")
        raise
