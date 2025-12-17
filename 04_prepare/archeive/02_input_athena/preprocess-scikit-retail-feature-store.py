from __future__ import print_function, unicode_literals

from sklearn.model_selection import train_test_split

from datetime import datetime
from time import gmtime, strftime, sleep

import sys
import argparse
import json
import os
import csv
import glob
from pathlib import Path
import time
import boto3
import subprocess
from urllib.parse import urlparse
# 🔧 ติดตั้ง SageMaker SDK ภายใน Processing container
subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker==2.24.1"])

import pandas as pd
import sagemaker
from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.feature_definition import (
    FeatureDefinition,
    FeatureTypeEnum,
)

# --------------------------------------------------------------------
# Global setup: region, role, bucket, clients
# --------------------------------------------------------------------
region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
print("Region: {}".format(region))

# Get role from STS + IAM (ตาม pattern template)
sts = boto3.Session(region_name=region).client(service_name="sts", region_name=region)

caller_identity = sts.get_caller_identity()
print("caller_identity: {}".format(caller_identity))

assumed_role_arn = caller_identity["Arn"]
print("(assumed_role) caller_identity_arn: {}".format(assumed_role_arn))

parts = assumed_role_arn.split("/")
assumed_role_name = parts[-2] if len(parts) >= 2 else parts[-1]

iam = boto3.Session(region_name=region).client(service_name="iam", region_name=region)
get_role_response = iam.get_role(RoleName=assumed_role_name)
print("get_role_response {}".format(get_role_response))
role = get_role_response["Role"]["Arn"]
print("role {}".format(role))

bucket = sagemaker.Session().default_bucket()
print("The DEFAULT BUCKET is {}".format(bucket))

sm = boto3.Session(region_name=region).client(service_name="sagemaker", region_name=region)
featurestore_runtime = boto3.Session(region_name=region).client(
    service_name="sagemaker-featurestore-runtime", region_name=region
)
s3 = boto3.Session(region_name=region).client(service_name="s3", region_name=region)

sagemaker_session = sagemaker.Session(
    boto_session=boto3.Session(region_name=region),
    sagemaker_client=sm,
    sagemaker_featurestore_runtime_client=featurestore_runtime,
)

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------
def cast_object_to_string(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Cast pandas object dtype -> pandas StringDtype (Feature Store-friendly)"""
    for label in data_frame.columns:
        if data_frame.dtypes[label] == "object":
            data_frame[label] = data_frame[label].astype("str").astype("string")
    return data_frame


def wait_for_feature_group_creation_complete(feature_group: FeatureGroup):
    """Wait until FeatureGroupStatus == Created"""
    try:
        status = feature_group.describe().get("FeatureGroupStatus")
        print("Feature Group status: {}".format(status))
        while status == "Creating":
            print("Waiting for Feature Group Creation")
            sleep(5)
            status = feature_group.describe().get("FeatureGroupStatus")
            print("Feature Group status: {}".format(status))
        if status != "Created":
            print("Feature Group status: {}".format(status))
            raise RuntimeError(f"Failed to create feature group {feature_group.name}")
        print(f"FeatureGroup {feature_group.name} successfully created.")
    except Exception as e:
        print("No feature group created yet or describe failed: {}".format(e))
        raise


def create_or_load_feature_group(offline_store_s3_uri: str, feature_group_name: str) -> FeatureGroup:
    """
    สร้างหรือโหลด Feature Group สำหรับ retail-demand
    offline_store_s3_uri: S3 URI เต็มของ offline store เช่น s3://bucket/feature-store/...
    """
    # Feature definitions ให้ครบทุก column ที่เราจะ ingest (รวม feature + label + split_type)
    feature_definitions = [
        FeatureDefinition(feature_name="record_id", feature_type=FeatureTypeEnum.INTEGRAL),
        FeatureDefinition(feature_name="date", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="store_id", feature_type=FeatureTypeEnum.INTEGRAL),
        FeatureDefinition(feature_name="day_of_week", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="is_weekend", feature_type=FeatureTypeEnum.INTEGRAL),
        FeatureDefinition(feature_name="is_holiday", feature_type=FeatureTypeEnum.INTEGRAL),
        FeatureDefinition(feature_name="holiday_name", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="max_temp_c", feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name="rainfall_mm", feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name="is_hot_day", feature_type=FeatureTypeEnum.INTEGRAL),
        FeatureDefinition(feature_name="is_rainy_day", feature_type=FeatureTypeEnum.INTEGRAL),
        FeatureDefinition(feature_name="base_price", feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name="discount_pct", feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name="is_promo", feature_type=FeatureTypeEnum.INTEGRAL),
        FeatureDefinition(feature_name="promo_type", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="final_price", feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name="units_sold", feature_type=FeatureTypeEnum.INTEGRAL),
        # 👉 high_demand: label แบบ binary ที่เราสร้างจาก units_sold
        FeatureDefinition(feature_name="high_demand", feature_type=FeatureTypeEnum.INTEGRAL),
        FeatureDefinition(feature_name="event_time", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="year", feature_type=FeatureTypeEnum.INTEGRAL),
        FeatureDefinition(feature_name="month", feature_type=FeatureTypeEnum.INTEGRAL),
        FeatureDefinition(feature_name="day", feature_type=FeatureTypeEnum.INTEGRAL),
        FeatureDefinition(feature_name="day_of_year", feature_type=FeatureTypeEnum.INTEGRAL),
        FeatureDefinition(feature_name="day_of_week_index", feature_type=FeatureTypeEnum.INTEGRAL),
        FeatureDefinition(feature_name="discount_amount", feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name="is_promo_or_holiday", feature_type=FeatureTypeEnum.INTEGRAL),
        FeatureDefinition(feature_name="split_type", feature_type=FeatureTypeEnum.STRING),
    ]

    feature_group = FeatureGroup(
        name=feature_group_name,
        feature_definitions=feature_definitions,
        sagemaker_session=sagemaker_session,
    )

    print("Feature Group: {}".format(feature_group))

    # ลองรอดูก่อนว่า FG มีอยู่แล้วไหม (เคยสร้างจาก process อื่นหรือยัง)
    try:
        print("Waiting for existing Feature Group (if any) ...")
        wait_for_feature_group_creation_complete(feature_group)
        print("Existing Feature Group is ready.")
        return feature_group
    except Exception as e:
        print("Before CREATE FG wait exception (probably FG does not exist yet): {}".format(e))

    # ถ้าไม่มี → สร้างใหม่
    record_identifier_feature_name = "record_id"
    event_time_feature_name = "event_time"

    try:
        print("Creating Feature Group with role {}...".format(role))
        feature_group.create(
            s3_uri=offline_store_s3_uri,
            record_identifier_name=record_identifier_feature_name,
            event_time_feature_name=event_time_feature_name,
            role_arn=role,
            enable_online_store=False,
        )
        print("Creating Feature Group. Completed.")

        print("Waiting for new Feature Group to become available...")
        wait_for_feature_group_creation_complete(feature_group)
        print("Feature Group available.")
        feature_group.describe()

    except Exception as e:
        print("Exception while creating Feature Group (maybe it already exists): {}".format(e))

    return feature_group


def parse_s3_uri(s3_uri: str):
    """แยก s3://bucket/prefix -> (bucket, prefix)"""
    if not s3_uri.startswith("s3://"):
        raise ValueError("Expected S3 URI like s3://bucket/prefix, got: {}".format(s3_uri))
    no_scheme = s3_uri[5:]
    bucket_name, _, key_prefix = no_scheme.partition("/")
    return bucket_name, key_prefix

# --------------------------------------------------------------------
# run_athena_query to be processed later
# --------------------------------------------------------------------

def run_athena_query(database: str, table: str, query: str, workgroup: str = "primary",run_id: str = None) -> str:
    """
    Run an Athena query and return the S3 URI of the result file.

    If 'query' is None/empty, we default to SELECT * FROM {table}.
    """
    athena = boto3.Session(region_name=region).client("athena", region_name=region)

    if not query:
        if not table:
            raise ValueError("Either --athena-query or --athena-table must be provided when using Athena.")
        query = f"SELECT * FROM {database}.{table}"

    if run_id is None:
        run_id = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

    print("Running Athena query:")
    print(query)

    # Write results under default bucket in a dedicated prefix
    output_s3_prefix = f"s3://{bucket}/retail-demand/athena-preprocess-results-{run_id}/"
    resp = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": output_s3_prefix},
        WorkGroup=workgroup,
    )
    qid = resp["QueryExecutionId"]

    # Wait for query to finish
    while True:
        desc = athena.get_query_execution(QueryExecutionId=qid)
        state = desc["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        print("  Athena query status:", state)
        time.sleep(5)

    if state != "SUCCEEDED":
        reason = desc["QueryExecution"]["Status"].get("StateChangeReason", "")
        raise RuntimeError(f"Athena query failed with state={state}: {reason}")

    output_location = desc["QueryExecution"]["ResultConfiguration"]["OutputLocation"]
    print("Athena result at:", output_location)
    return output_location


# --------------------------------------------------------------------
# Feature engineering + split + ingest
# --------------------------------------------------------------------
def process_engineer_features(
    input_file: str,
    train_data_dir: str,
    validation_data_dir: str,
    test_data_dir: str,
    train_split_percentage: float,
    validation_split_percentage: float,
    test_split_percentage: float,
    balance_dataset: bool,
    feature_group: FeatureGroup,
):
    print("Processing file: {}".format(input_file))
    df = pd.read_csv(input_file)

    # ----- Feature engineering -----
    # Convert date to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Event time for Feature Store (ISO format string)
    df["event_time"] = df["date"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Calendar features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear
    df["day_of_week_index"] = df["date"].dt.dayofweek  # Monday=0

    # Convert back to string date
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # Price / promo features
    df["discount_amount"] = df["base_price"] * df["discount_pct"]
    df["is_promo_or_holiday"] = ((df["is_promo"] == 1) | (df["is_holiday"] == 1)).astype(int)

    # 👉 สร้าง high_demand จาก 75th percentile ของ units_sold ของทั้ง dataset นี้
    p75 = df["units_sold"].quantile(0.75)
    print("75th percentile of units_sold for this file:", p75)
    df["high_demand"] = (df["units_sold"] >= p75).astype(int)
    print("high_demand value counts:\n", df["high_demand"].value_counts())

    # Initialize split_type (จะ override หลัง split)
    df["split_type"] = "train"

    # ----- Train / validation / test split -----
    total = train_split_percentage + validation_split_percentage + test_split_percentage
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Train/Val/Test percentages must sum to 1.0, got {}".format(total))

    df_train, df_tmp = train_test_split(
        df, test_size=(1.0 - train_split_percentage), random_state=42, shuffle=True
    )

    if validation_split_percentage + test_split_percentage > 0:
        relative_test_size = test_split_percentage / (validation_split_percentage + test_split_percentage)
        df_val, df_test = train_test_split(df_tmp, test_size=relative_test_size, random_state=42, shuffle=True)
    else:
        df_val, df_test = None, None

    df_train = df_train.copy()
    df_train["split_type"] = "train"

    if df_val is not None:
        df_val = df_val.copy()
        df_val["split_type"] = "validation"

    if df_test is not None:
        df_test = df_test.copy()
        df_test["split_type"] = "test"

    # ----- Save splits to local dirs -----
    os.makedirs(train_data_dir, exist_ok=True)
    os.makedirs(validation_data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    train_out = os.path.join(train_data_dir, "train.csv")
    df_train.to_csv(train_out, index=False)
    print("Saved train split to {}".format(train_out))

    if df_val is not None:
        val_out = os.path.join(validation_data_dir, "validation.csv")
        df_val.to_csv(val_out, index=False)
        print("Saved validation split to {}".format(val_out))

    if df_test is not None:
        test_out = os.path.join(test_data_dir, "test.csv")
        df_test.to_csv(test_out, index=False)
        print("Saved test split to {}".format(test_out))

    # ----- Ingest records into Feature Store -----
    df_fs_train = cast_object_to_string(df_train.copy())
    df_fs_val = cast_object_to_string(df_val.copy()) if df_val is not None else None
    df_fs_test = cast_object_to_string(df_test.copy()) if df_test is not None else None

    print("Ingesting train features...")
    feature_group.ingest(data_frame=df_fs_train, max_workers=3, wait=True)

    if df_fs_val is not None:
        print("Ingesting validation features...")
        feature_group.ingest(data_frame=df_fs_val, max_workers=3, wait=True)

    if df_fs_test is not None:
        print("Ingesting test features...")
        feature_group.ingest(data_frame=df_fs_test, max_workers=3, wait=True)

    print("...features ingested from file {}!".format(input_file))


# --------------------------------------------------------------------
# Argparse + main process() : tie everything together
# --------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-data",
        type=str,
        default="/opt/ml/processing/input/data/",
        help="Input data directory (mounted from S3)",
    )
    parser.add_argument(
        "--output-data",
        type=str,
        default="/opt/ml/processing/output/retail_product",
        help="Base output directory for train/validation/test splits",
    )
    parser.add_argument("--train-split-percentage", type=float, default=0.9)
    parser.add_argument("--validation-split-percentage", type=float, default=0.05)
    parser.add_argument("--test-split-percentage", type=float, default=0.05)
    parser.add_argument(
        "--balance-dataset",
        type=str,
        default="True",
        help="Whether to apply balancing (not used in this example, kept for compatibility).",
    )
    parser.add_argument(
        "--feature-store-offline-prefix",
        type=str,
        required=True,
        help="S3 URI for Feature Store offline store, e.g. s3://bucket/prefix",
    )
    parser.add_argument(
        "--feature-group-name",
        type=str,
        required=True,
        help="Name of the Feature Group to create or load",
    )
    parser.add_argument(
        "--current-host",
        type=str,
        default=os.environ.get("SM_CURRENT_HOST", "unknown"),
        help="Current host name (for SageMaker Processing clusters)",
    )
    # athena arguement
    parser.add_argument(
        "--athena-database",
        type=str,
        default=None,
        help="If set, read raw data from Athena instead of local CSV inputs.",
    )
    parser.add_argument(
        "--athena-table",
        type=str,
        default=None,
        help="Athena table name (used if --athena-query is not provided).",
    )
    parser.add_argument(
        "--athena-query",
        type=str,
        default=None,
        help="Custom Athena SQL statement. If omitted, defaults to SELECT * FROM <athena-table>.",
    )
    parser.add_argument(
        "--athena-workgroup",
        type=str,
        default="primary",
        help="Athena workgroup name (default: primary).",
    )

    return parser.parse_args()


def process(args):
    print("Current host: {}".format(args.current_host))

    # 1) สร้าง/โหลด Feature Group
    feature_group = create_or_load_feature_group(
        offline_store_s3_uri=args.feature_store_offline_prefix,
        feature_group_name=args.feature_group_name,
    )

    print("Feature Group description:")
    try:
        desc = feature_group.describe()
        print(json.dumps(desc, indent=2, default=str))
    except Exception as e:
        print("Could not describe feature group: {}".format(e))

    try:
        print("Feature Group Hive DDL:")
        print(feature_group.as_hive_ddl())
    except Exception as e:
        print("Could not get Hive DDL: {}".format(e))

    # 2) path สำหรับ output splits
    train_data_dir = os.path.join(args.output_data, "train")
    validation_data_dir = os.path.join(args.output_data, "validation")
    test_data_dir = os.path.join(args.output_data, "test")

    # 3) เตรียม input จาก Athena หรือจาก CSV ใน input-data
    input_files = []

    if args.athena_database:
        # --- Read raw data from Athena ---
        print(
            f"Using Athena as data source: db={args.athena_database}, "
            f"table={args.athena_table}, query={bool(args.athena_query)}"
        )
        run_id = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        print("Preprocess run_id:", run_id)


        # Run Athena query and get S3 URI of result
        result_s3_uri = run_athena_query(
            database=args.athena_database,
            table=args.athena_table,
            query=args.athena_query,
            workgroup=args.athena_workgroup,
            run_id=run_id
        )

        # Download Athena result to local CSV under args.input_data
        os.makedirs(args.input_data, exist_ok=True)
        parsed = urlparse(result_s3_uri)
        result_bucket = parsed.netloc
        result_key = parsed.path.lstrip("/")

        local_csv = os.path.join(args.input_data, "athena_query_output.csv")
        print(f"Downloading Athena result {result_s3_uri} -> {local_csv}")
        s3.download_file(result_bucket, result_key, local_csv)

        input_files = [local_csv]
    else:
        # --- Fallback: use CSV files mounted at input-data (original behavior) ---
        input_files = glob.glob(os.path.join(args.input_data, "*.csv"))
        print("Input files found under {}: {}".format(args.input_data, input_files))

    if not input_files:
        raise FileNotFoundError(
            "No input data found. Either provide CSVs under {} or use --athena-database/--athena-table/--athena-query.".format(
                args.input_data
            )
        )

    if not input_files:
        raise FileNotFoundError("No CSV files found under {}".format(args.input_data))

    # 3) ทำ feature engineering + ingest สำหรับทุกไฟล์ (ส่วนมากของเรามีไฟล์เดียว)
    for input_file in input_files:
        process_engineer_features(
            input_file=input_file,
            train_data_dir=train_data_dir,
            validation_data_dir=validation_data_dir,
            test_data_dir=test_data_dir,
            train_split_percentage=args.train_split_percentage,
            validation_split_percentage=args.validation_split_percentage,
            test_split_percentage=args.test_split_percentage,
            balance_dataset=args.balance_dataset.lower() == "true",
            feature_group=feature_group,
        )

    # 4) รอให้ offline store มีไฟล์จริง ๆ
    offline_store_contents = None
    bucket_offline, prefix_offline = parse_s3_uri(args.feature_store_offline_prefix)

    while offline_store_contents is None:
        objects_in_bucket = s3.list_objects(Bucket=bucket_offline, Prefix=prefix_offline)
        if "Contents" in objects_in_bucket and len(objects_in_bucket["Contents"]) > 0:
            offline_store_contents = objects_in_bucket["Contents"]
        else:
            print("Waiting for data in offline store...\n")
            sleep(60)

    print("Data available in offline store.")
    print("Complete.")


if __name__ == "__main__":
    args = parse_args()
    process(args)
