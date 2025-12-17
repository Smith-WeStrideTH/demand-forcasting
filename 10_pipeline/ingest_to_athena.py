import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from typing import Tuple, Dict, Any, List

import boto3
import pandas as pd
from urllib.parse import urlparse


def _pip_install(pkg: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


# ติดตั้ง dependency
_pip_install("pandas")
_pip_install("boto3")
_pip_install("PyAthena")

from pyathena import connect  # noqa: E402


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """
    s3://bucket/prefix -> ("bucket", "prefix")
    """
    uri = uri.strip()
    if not uri.startswith("s3://"):
        raise ValueError("Not a valid S3 URI: {}".format(uri))
    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def find_eval_files(s3_client, bucket: str, prefix: str) -> Dict[str, str]:
    """
    หา key ของไฟล์ที่เราต้องใช้ใน eval_output_s3 prefix:
      - evaluation_summary.csv
      - shap_feature_importance.csv
      - test_predictions.csv
      - data_profile.json
    """
    required_suffixes = {
        "evaluation_summary": "evaluation_summary.csv",
        "shap_feature_importance": "shap_feature_importance.csv",
        "test_predictions": "test_predictions.csv",
        "data_profile": "data_profile.json",
    }

    found = {k: None for k in required_suffixes.keys()}

    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            for name, suffix in required_suffixes.items():
                if key.endswith(suffix):
                    found[name] = key

    missing = [k for k, v in found.items() if v is None]
    if missing:
        raise RuntimeError(
            f"Missing expected eval files under {prefix}: {missing}"
        )

    return found


def download_to_local(s3_client, bucket: str, key: str, local_path: str) -> None:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"[INGEST] Downloading s3://{bucket}/{key} -> {local_path}")
    s3_client.download_file(bucket, key, local_path)


def load_csv_local(path: str) -> pd.DataFrame:
    print(f"[INGEST] Loading CSV: {path}")
    return pd.read_csv(path)


def convert_data_profile_json_to_df(
    json_path: str,
    eval_timestamp_utc: str,
    mlflow_run_id: str,
    eval_date: str,
) -> pd.DataFrame:
    """
    data_profile.json:
    {
      "generated_at_utc": "...",
      "n_rows": 25,
      "label_units_sold": {...},
      "features": {
        "store_id": {...},
        ...
      }
    }

    แปลงเป็น DataFrame แบบ long:
      - eval_timestamp_utc
      - mlflow_run_id
      - eval_date
      - generated_at_utc
      - feature_name
      - kind ("label" หรือ "feature")
      - n_rows
      - mean, std, min, max, p25, p50, p75
    """
    print(f"[INGEST] Converting data_profile JSON: {json_path}")
    with open(json_path, "r") as f:
        prof = json.load(f)

    generated_at = prof.get("generated_at_utc")
    n_rows = prof.get("n_rows")
    rows: List[Dict[str, Any]] = []

    # label_units_sold
    label_stats = prof.get("label_units_sold", {})
    if label_stats:
        rows.append(
            {
                "eval_timestamp_utc": eval_timestamp_utc,
                "mlflow_run_id": mlflow_run_id,
                "eval_date": eval_date,
                "generated_at_utc": generated_at,
                "feature_name": "units_sold",
                "kind": "label",
                "n_rows": n_rows,
                "mean": label_stats.get("mean"),
                "std": label_stats.get("std"),
                "min": label_stats.get("min"),
                "max": label_stats.get("max"),
                "p25": label_stats.get("p25"),
                "p50": label_stats.get("p50"),
                "p75": label_stats.get("p75"),
            }
        )

    # features
    feat_stats = prof.get("features", {})
    for feat_name, stats in feat_stats.items():
        rows.append(
            {
                "eval_timestamp_utc": eval_timestamp_utc,
                "mlflow_run_id": mlflow_run_id,
                "eval_date": eval_date,
                "generated_at_utc": generated_at,
                "feature_name": feat_name,
                "kind": "feature",
                "n_rows": n_rows,
                "mean": stats.get("mean"),
                "std": stats.get("std"),
                "min": stats.get("min"),
                "max": stats.get("max"),
                "p25": stats.get("p25"),
                "p50": stats.get("p50"),
                "p75": stats.get("p75"),
            }
        )

    df = pd.DataFrame(rows)
    print("[INGEST] data_profile rows:", len(df))
    return df


def write_df_to_s3(
    s3_client,
    df: pd.DataFrame,
    bucket: str,
    prefix: str,
    filename: str,
) -> str:
    """
    เขียน df ไป local แล้ว upload ไป s3://bucket/prefix/filename
    """
    os.makedirs("/opt/ml/processing/tmp_out", exist_ok=True)
    local_path = os.path.join("/opt/ml/processing/tmp_out", filename)
    print(f"[INGEST] Writing DataFrame to local CSV: {local_path}")
    df.to_csv(local_path, index=False)

    key = prefix.rstrip("/") + "/" + filename
    print(f"[INGEST] Uploading {local_path} -> s3://{bucket}/{key}")
    s3_client.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"


# ---------- Athena helpers ----------

def map_dtype_to_athena(dtype: Any) -> str:
    """
    Map pandas dtype -> Athena type
    """
    if pd.api.types.is_integer_dtype(dtype):
        return "bigint"
    if pd.api.types.is_float_dtype(dtype):
        return "double"
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    # default to string
    return "string"


def create_athena_database_if_not_exists(conn, db_name: str) -> None:
    sql = f"CREATE DATABASE IF NOT EXISTS {db_name}"
    print("[INGEST][ATHENA]", sql)
    cursor = conn.cursor()
    cursor.execute(sql)
    cursor.close()


def table_exists(conn, db_name: str, table_name: str) -> bool:
    sql = f"SHOW TABLES IN {db_name} LIKE '{table_name}'"
    print("[INGEST][ATHENA]", sql)
    cursor = conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    cursor.close()
    return len(rows) > 0


def create_athena_table_if_not_exists(
    conn,
    db_name: str,
    table_name: str,
    df_sample: pd.DataFrame,
    table_s3_location: str,
) -> None:
    """
    สร้าง External Table แบบ CSV (textfile) จาก schema ของ df_sample ถ้ายังไม่มี
    """
    if table_exists(conn, db_name, table_name):
        print(f"[INGEST][ATHENA] Table {db_name}.{table_name} already exists, skip CREATE")
        return

    cols_def = []
    for col in df_sample.columns:
        athena_type = map_dtype_to_athena(df_sample[col].dtype)
        safe_col = col.lower().replace(" ", "_").replace("-", "_")
        cols_def.append(f"  `{safe_col}` {athena_type}")

    cols_sql = ",\n".join(cols_def)

    sql = f"""
CREATE EXTERNAL TABLE IF NOT EXISTS {db_name}.{table_name} (
{cols_sql}
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe'
WITH SERDEPROPERTIES (
  'serialization.format' = ',',
  'field.delim' = ','
)
STORED AS TEXTFILE
LOCATION '{table_s3_location.rstrip('/')}/'
TBLPROPERTIES ('skip.header.line.count'='1')
"""
    print("[INGEST][ATHENA] Creating table with SQL:\n", sql)
    cursor = conn.cursor()
    cursor.execute(sql)
    cursor.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval-output-s3",
        type=str,
        required=True,
        help="S3 URI of evaluation output prefix (where evaluation_summary.csv etc. are).",
    )
    parser.add_argument(
        "--monitor-base-s3",
        type=str,
        required=True,
        help="Base S3 URI for monitor_retail_demand (e.g. s3://bucket/monitor_retail_demand)",
    )
    parser.add_argument(
        "--athena-db-name",
        type=str,
        default="monitor_retail_demand",
        help="Athena database name to create/use.",
    )

    args = parser.parse_args()
    eval_output_s3 = args.eval_output_s3
    monitor_base_s3 = args.monitor_base_s3
    athena_db = args.athena_db_name

    print("[INGEST] eval_output_s3 =", eval_output_s3)
    print("[INGEST] monitor_base_s3 =", monitor_base_s3)
    print("[INGEST] athena_db_name =", athena_db)

    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    s3 = boto3.client("s3", region_name=region)

    eval_bucket, eval_prefix = parse_s3_uri(eval_output_s3)
    mon_bucket, mon_prefix = parse_s3_uri(monitor_base_s3)

    # หาไฟล์ทั้ง 4
    eval_files = find_eval_files(s3, eval_bucket, eval_prefix)
    print("[INGEST] Found eval files:", eval_files)

    # ดาวน์โหลดทั้งหมดมา local
    local_dir = "/opt/ml/processing/eval_files"
    os.makedirs(local_dir, exist_ok=True)

    local_eval_summary = os.path.join(local_dir, "evaluation_summary.csv")
    local_shap = os.path.join(local_dir, "shap_feature_importance.csv")
    local_preds = os.path.join(local_dir, "test_predictions.csv")
    local_profile_json = os.path.join(local_dir, "data_profile.json")

    download_to_local(s3, eval_bucket, eval_files["evaluation_summary"], local_eval_summary)
    download_to_local(s3, eval_bucket, eval_files["shap_feature_importance"], local_shap)
    download_to_local(s3, eval_bucket, eval_files["test_predictions"], local_preds)
    download_to_local(s3, eval_bucket, eval_files["data_profile"], local_profile_json)

    # อ่าน evaluation_summary เพื่อดึง eval_timestamp_utc, mlflow_run_id
    df_eval_summary = load_csv_local(local_eval_summary)
    if df_eval_summary.empty:
        raise RuntimeError("evaluation_summary.csv is empty!")

    eval_timestamp_utc = str(df_eval_summary.loc[0, "eval_timestamp_utc"])
    mlflow_run_id = str(df_eval_summary.loc[0, "mlflow_run_id"])
    eval_date = eval_timestamp_utc[:10]  # YYYY-MM-DD

    print("[INGEST] eval_timestamp_utc =", eval_timestamp_utc)
    print("[INGEST] mlflow_run_id =", mlflow_run_id)
    print("[INGEST] eval_date =", eval_date)

    # สร้างสตริง timestamp สำหรับชื่อไฟล์ (safe)
    safe_ts = eval_timestamp_utc.replace(":", "").replace("-", "").replace(".", "").replace("Z", "")

    # เพิ่ม context ให้ evaluation_summary เองด้วย
    df_eval_summary["eval_date"] = eval_date
    df_eval_summary["ingested_at_utc"] = datetime.utcnow().isoformat() + "Z"

    # 1) evaluation_summary.csv -> monitor_retail_demand.evaluation_summary
    eval_prefix = mon_prefix.rstrip("/") + "/evaluation_summary"
    eval_filename = f"evaluation_summary_{safe_ts}.csv"
    eval_s3_uri = write_df_to_s3(s3, df_eval_summary, mon_bucket, eval_prefix, eval_filename)
    print("[INGEST] Wrote evaluation_summary to", eval_s3_uri)

    # 2) shap_feature_importance.csv -> monitor_retail_demand.shap_feature_importance
    df_shap = load_csv_local(local_shap)
    df_shap["eval_timestamp_utc"] = eval_timestamp_utc
    df_shap["mlflow_run_id"] = mlflow_run_id
    df_shap["eval_date"] = eval_date
    shap_prefix = mon_prefix.rstrip("/") + "/shap_feature_importance"
    shap_filename = f"shap_feature_importance_{safe_ts}.csv"
    shap_s3_uri = write_df_to_s3(s3, df_shap, mon_bucket, shap_prefix, shap_filename)
    print("[INGEST] Wrote shap_feature_importance to", shap_s3_uri)

    # 3) test_predictions.csv -> monitor_retail_demand.test_predictions
    df_preds = load_csv_local(local_preds)
    df_preds["eval_timestamp_utc"] = eval_timestamp_utc
    df_preds["mlflow_run_id"] = mlflow_run_id
    df_preds["eval_date"] = eval_date
    preds_prefix = mon_prefix.rstrip("/") + "/test_predictions"
    preds_filename = f"test_predictions_{safe_ts}.csv"
    preds_s3_uri = write_df_to_s3(s3, df_preds, mon_bucket, preds_prefix, preds_filename)
    print("[INGEST] Wrote test_predictions to", preds_s3_uri)

    # 4) data_profile.json -> CSV -> monitor_retail_demand.data_profile
    df_profile = convert_data_profile_json_to_df(
        local_profile_json,
        eval_timestamp_utc=eval_timestamp_utc,
        mlflow_run_id=mlflow_run_id,
        eval_date=eval_date,
    )
    profile_prefix = mon_prefix.rstrip("/") + "/data_profile"
    profile_filename = f"data_profile_{safe_ts}.csv"
    profile_s3_uri = write_df_to_s3(s3, df_profile, mon_bucket, profile_prefix, profile_filename)
    print("[INGEST] Wrote data_profile to", profile_s3_uri)

    # ------------------------------------------------------------------
    # Athena: create database + tables if not exists
    # ------------------------------------------------------------------
    s3_staging_dir = f"s3://{mon_bucket}/athena/staging/"
    print("[INGEST][ATHENA] Using s3_staging_dir:", s3_staging_dir)

    conn = connect(region_name=region, s3_staging_dir=s3_staging_dir)

    create_athena_database_if_not_exists(conn, athena_db)

    # TABLE 1: evaluation_summary
    # ใช้ df_eval_summary เป็น sample schema
    eval_table_location = f"s3://{mon_bucket}/{mon_prefix.strip('/')}/evaluation_summary"
    create_athena_table_if_not_exists(
        conn=conn,
        db_name=athena_db,
        table_name="evaluation_summary",
        df_sample=df_eval_summary,
        table_s3_location=eval_table_location,
    )

    # TABLE 2: shap_feature_importance
    shap_table_location = f"s3://{mon_bucket}/{mon_prefix.strip('/')}/shap_feature_importance"
    create_athena_table_if_not_exists(
        conn=conn,
        db_name=athena_db,
        table_name="shap_feature_importance",
        df_sample=df_shap,
        table_s3_location=shap_table_location,
    )

    # TABLE 3: test_predictions
    preds_table_location = f"s3://{mon_bucket}/{mon_prefix.strip('/')}/test_predictions"
    create_athena_table_if_not_exists(
        conn=conn,
        db_name=athena_db,
        table_name="test_predictions",
        df_sample=df_preds,
        table_s3_location=preds_table_location,
    )

    # TABLE 4: data_profile
    profile_table_location = f"s3://{mon_bucket}/{mon_prefix.strip('/')}/data_profile"
    create_athena_table_if_not_exists(
        conn=conn,
        db_name=athena_db,
        table_name="data_profile",
        df_sample=df_profile,
        table_s3_location=profile_table_location,
    )

    conn.close()
    print("[INGEST] Complete.")


if __name__ == "__main__":
    main()
