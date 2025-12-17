
We are going to do project called "retail forecasting" Here is the scenario:
--------------------------------
Demand Forecasting for Products

A supermarket wants to know how many units of bottled water they will sell next week.

Why:

Prevent out-of-stock during promotions

Avoid overstock that increases warehouse cost

Data used:

Past sales

Weather forecast

Holidays / long weekends

Price changes and promotions 

--------------------------------

This project is an ML ops end to end on AWS Sagemaker on different section will be create its own ".ipynb" by step as following
1. Create datasets:  you have to create datasets for "Demand Forecasting for Products" 500 rows (name -> retail-demand-forecasting.csv )
2. Ingestion: Using S3, Athena as follow:
	2.1 Write CSV to S3 raw (this s3 is an raw bucket):
```python
import boto3
import sagemaker
import pandas as pd

sess = sagemaker.Session()
bucket = sess.default_bucket()
role = sagemaker.get_execution_role()
region = boto3.Session().region_name

s3_private_path_csv = "s3://{}/retail-demand-forecasting/csv".format(bucket) 
%store s3_private_path_csv 

# and using !aws s3 cp retail-demand-forecasting.csv $s3_private_path_csv 
```

	2.2 Create Athena Database :
```python
from pyathena import connect
import pandas as pd

%store -r s3_private_path_csv

# Create Athena Database
s3_staging_dir = "s3://{0}/athena/staging".format(bucket)
conn = connect(region_name=region, s3_staging_dir=s3_staging_dir)
statement = "CREATE DATABASE IF NOT EXISTS {}".format(database_name)
pd.read_sql(statement, conn)

# Verify The Database
statement = "SHOW DATABASES"
df_show = pd.read_sql(statement, conn)
df_show.head(5)

if database_name in df_show.values:
    ingest_create_athena_db_passed = True
%store ingest_create_athena_db_passed
```
	2.3 Register CSV Data With Athena :
```python
s3_staging_dir = "s3://{0}/athena/staging".format(bucket)
database_name = "retail_demand"
table_name_tsv = "demand_product"

# Create table and ingest S3 to Athena
conn = connect(region_name=region, s3_staging_dir=s3_staging_dir)
statement = """CREATE EXTERNAL TABLE IF NOT EXISTS {database_name}.{table_name_tsv}(....) LOCATION {s3_private_path_csv}"""
pd.read_sql(statement, conn)

# Verify table
statement = "SHOW TABLES in {}".format(database_name)
df_show = pd.read_sql(statement, conn)

statement = """SELECT * FROM {}.{}""".format(
    database_name, table_name_tsv
)
%store ingest_create_athena_table_csv_passed
```

3. Explore :  crucial step to do data quality and bias

	3.1 Prepare Dataset for Bias Analysis  

```python
import boto3
import sagemaker
import pandas as pd
import numpy as np

sess = sagemaker.Session()
bucket = sess.default_bucket()
role = sagemaker.get_execution_role()
region = boto3.Session().region_name
sm = boto3.Session().client(service_name="sagemaker", region_name=region)

# Get Data from S3

%store -r s3_private_path_csv
!aws s3 cp $s3_private_path_csv ./data-clarify

data = pd.read_csv("./data-clarify/...")

# Detecting Bias with Amazon SageMaker Clarify
from sagemaker import clarify

clarify_processor = clarify.SageMakerClarifyProcessor(
    role=role, 
    instance_count=1, 
    instance_type="ml.c5.xlarge", 
    sagemaker_session=sess
)

bias_report_output_path = "s3://{}/clarify".format(bucket)
# Writing DataConfig
bias_data_config = clarify.DataConfig(
    s3_data_input_path=bias_data_s3_uri,
    s3_output_path=bias_report_output_path,
    label="...",
    headers=data.columns.to_list(),
    dataset_type="text/csv",
)
# Writing BiasConfig
bias_config = clarify.BiasConfig(
    label_values_or_threshold=...,
    facet_name="...",
    facet_values_or_threshold=...,
)

# Detect Bias with a SageMaker Processing Job and Clarify
clarify_processor.run_pre_training_bias(
    data_config=bias_data_config, 
    data_bias_config=bias_config, 
    methods=["CI", "DPL", "KL", "JS", "LP", "TVD", "KS"],
    wait=False, 
    logs=False
)
# monitor and waiting the process
run_pre_training_bias_processing_job_name = clarify_processor.latest_job.job_name
running_processor = sagemaker.processing.ProcessingJob.from_processing_name(
    processing_job_name=run_pre_training_bias_processing_job_name, sagemaker_session=sess
)
running_processor.wait(logs=False)

# Download Report From S3
!aws s3 ls $bias_report_output_path/
!aws s3 cp --recursive $bias_report_output_path ./generated_bias_report/
```

	3.2 Analyze Data Quality with SageMaker Processing Jobs and Spark, at this point we need to create [preprocess-deequ-pyspark.py and upload deequ-1.0.3-rc2.jar]

```python
import sagemaker
import boto3

sess = sagemaker.Session()
bucket = sess.default_bucket()
role = sagemaker.get_execution_role()
region = boto3.Session().region_name

# Spark preprocessing script
!pygmentize preprocess-deequ-pyspark.py

from sagemaker.spark.processing import PySparkProcessor

processor = PySparkProcessor(
    base_job_name="spark-retail-demand-analyzer",
    role=role,
    framework_version="2.4",
    instance_count=1,
    instance_type="ml.m5.2xlarge",
    max_runtime_in_seconds=300,
)


s3_input_data = "s3://{}/retail-demand-forecasting/csv/".format(bucket)


# Setup Output Data
from time import gmtime, strftime
timestamp_prefix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
output_prefix = "retail-demand-forecasting-spark-analyzer-{}".format(timestamp_prefix)
processing_job_name = "retail-demand-forecasting-spark-analyzer-{}".format(timestamp_prefix)
s3_output_analyze_data = "s3://{}/{}/output".format(bucket, output_prefix)


# Start the Spark Processing Job
from sagemaker.processing import ProcessingOutput

processor.run(
    submit_app="preprocess-deequ-pyspark.py",
    submit_jars=["deequ-1.0.3-rc2.jar"],
    arguments=[
        "s3_input_data",
        s3_input_data,
        "s3_output_analyze_data",
        s3_output_analyze_data,
    ],
    logs=True,
    wait=False,
)

# Monitor the Processing Job
running_processor = sagemaker.processing.ProcessingJob.from_processing_name(
    processing_job_name=processing_job_name, sagemaker_session=sess
)
running_processor.wait()

```

	3.3 preprocess-deequ-pyspark.py
```python
from __future__ import print_function
from __future__ import unicode_literals

import time
import sys
import os
import shutil
import csv
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "pydeequ==0.1.5"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas==1.1.4"])

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, IntegerType, DoubleType
from pyspark.sql.functions import *

from pydeequ.analyzers import *
from pydeequ.checks import *
from pydeequ.verification import *
from pydeequ.suggestions import *

def main():
    args_iter = iter(sys.argv[1:])
    args = dict(zip(args_iter, args_iter))

    # Retrieve the args and replace 's3://' with 's3a://' (used by Spark)
    s3_input_data = args["s3_input_data"].replace("s3://", "s3a://")
    print(s3_input_data)
    s3_output_analyze_data = args["s3_output_analyze_data"].replace("s3://", "s3a://")
    print(s3_output_analyze_data)

    spark = SparkSession.builder.appName("....").getOrCreate()

    schema = StructType(
        [
            StructField(“..”, StringType(), True),
            StructField(“..”, IntegerType(), True), …
        ]
    )

    dataset = spark.read.csv(s3_input_data, header=True, schema=schema, sep="\t", quote="")

    # Calculate statistics on the dataset
    analysisResult = (
        AnalysisRunner(spark)
        .onData(dataset)
        .addAnalyzer(Size())
        .addAnalyzer(Completeness(“...”))
        .addAnalyzer(ApproxCountDistinct(“..”))
        .addAnalyzer(Mean(“..”))
        .addAnalyzer(Compliance(“..”, “.. >= ..”))
        .addAnalyzer(Correlation(“..”, “..”))
        .addAnalyzer(Correlation(“..”, “...”))
        .run()
    )

    metrics = AnalyzerContext.successMetricsAsDataFrame(spark, analysisResult)
    metrics.show(truncate=False)
    metrics.repartition(1).write.format("csv").mode("overwrite").option("header", True).option("sep", "\t").save(
        "{}/dataset-metrics".format(s3_output_analyze_data)
    )

    # Check data quality
    verificationResult = (
        VerificationSuite(spark)
        .onData(dataset)
        .addCheck(
            Check(spark, CheckLevel.Error, "Review Check")
            .hasSize(lambda x: x..)
            .hasMin(“..”, lambda x: ..)
            .hasMax(“..”, lambda x: ..)
            .isComplete(“..”)
            .isUnique(“..”)
            .isComplete(“..”)
            .isContainedIn(“..”, [..])
        )
        .run()
    )

    print(f"Verification Run Status: {verificationResult.status}")
    resultsDataFrame = VerificationResult.checkResultsAsDataFrame(spark, verificationResult)
    resultsDataFrame.show(truncate=False)
    resultsDataFrame.repartition(1).write.format("csv").mode("overwrite").option("header", True).option(
        "sep", "\t"
    ).save("{}/constraint-checks".format(s3_output_analyze_data))

    verificationSuccessMetricsDataFrame = VerificationResult.successMetricsAsDataFrame(spark, verificationResult)
    verificationSuccessMetricsDataFrame.show(truncate=False)
    verificationSuccessMetricsDataFrame.repartition(1).write.format("csv").mode("overwrite").option(
        "header", True
    ).option("sep", "\t").save("{}/success-metrics".format(s3_output_analyze_data))

    # Suggest new checks and constraints
    suggestionsResult = ConstraintSuggestionRunner(spark).onData(dataset).addConstraintRule(DEFAULT()).run()

    suggestions = suggestionsResult["constraint_suggestions"]
    parallelizedSuggestions = spark.sparkContext.parallelize(suggestions)

    suggestionsResultsDataFrame = spark.createDataFrame(parallelizedSuggestions)
    suggestionsResultsDataFrame.show(truncate=False)
    suggestionsResultsDataFrame.repartition(1).write.format("csv").mode("overwrite").option("header", True).option(
        "sep", "\t"
    ).save("{}/constraint-suggestions".format(s3_output_analyze_data))


#    spark.stop()


if __name__ == "__main__":
    main()

```
4. Prepare Data using  Feature Transformation you must create file [preprocess-scikit-retail-feature-store.py]
 and have  store feature so we can keep the processed feature ingest in athena everytime this job occur for tracking and experiment furture if needed

	4.1 Feature Transformation using SKLearnProcessor
```python

import sagemaker
import boto3

sess = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = sess.default_bucket()
region = boto3.Session().region_name

raw_input_data_s3_uri = "s3://{}/retail-demand-forecasting/csv/".format(bucket)

# Set the Processing Job Hyper-Parameters
from sagemaker.sklearn.processing import SKLearnProcessor

processing_instance_type = "ml.c5.2xlarge"
processing_instance_count = 2
train_split_percentage = 0.90
validation_split_percentage = 0.05
test_split_percentage = 0.05
balance_dataset = True

# Set and run Processor
processor = SKLearnProcessor(
    framework_version="0.23-1",
    role=role,
    instance_type=processing_instance_type,
    instance_count=processing_instance_count,
    env={"AWS_DEFAULT_REGION": region},
    max_runtime_in_seconds=7200,
)

from sagemaker.processing import ProcessingInput, ProcessingOutput

processor.run(
    code="preprocess-scikit-retail-feature-store.py",
    inputs=[
        ProcessingInput(
            input_name="raw-input-data",
            source=raw_input_data_s3_uri,
            destination="/opt/ml/processing/input/data/",
            s3_data_distribution_type="ShardedByS3Key",
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="bert-train", s3_upload_mode="EndOfJob", source="/opt/ml/processing/output/retail_product/train"
        ),
        ProcessingOutput(
            output_name="bert-validation",
            s3_upload_mode="EndOfJob",
            source="/opt/ml/processing/output/retail_product/validation",
        ),
        ProcessingOutput(
            output_name="bert-test", s3_upload_mode="EndOfJob", source="/opt/ml/processing/output/retail_product/test"
        ),
    ],
    arguments=[
        "--train-split-percentage",
        str(train_split_percentage),
        "--validation-split-percentage",
        str(validation_split_percentage),
        "--test-split-percentage",
        str(test_split_percentage),
        "--feature-store-offline-prefix",
        str(feature_store_offline_prefix),
        "--feature-group-name",
        str(feature_group_name),
    ],
    logs=True,
    wait=False,
)

# Monitor the Processing Job
scikit_processing_job_name = processor.jobs[-1].describe()["ProcessingJobName"]

running_processor = sagemaker.processing.ProcessingJob.from_processing_name(
    processing_job_name=scikit_processing_job_name, sagemaker_session=sess
)

processing_job_description = running_processor.describe()

output_config = processing_job_description["ProcessingOutputConfig"]
for output in output_config["Outputs"]:
    if output["OutputName"] == "...-train":
        processed_train_data_s3_uri = output["S3Output"]["S3Uri"]
    if output["OutputName"] == "...-validation":
        processed_validation_data_s3_uri = output["S3Output"]["S3Uri"]
    if output["OutputName"] == "...-test":
        processed_test_data_s3_uri = output["S3Output"]["S3Uri"]

%store processed_train_data_s3_uri
%store processed_validation_data_s3_uri
%store processed_test_data_s3_uri


```

	4.2 store feature in the script "preprocess-scikit-retail-feature-store.py" may look like this :

```python

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import functools
import multiprocessing

from datetime import datetime
from time import gmtime, strftime, sleep

import sys
import re
import collections
import argparse
import json
import os
import csv
import glob
from pathlib import Path
import time
import boto3
import subprocess


subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker==2.24.1"])

import pandas as pd
import re
import sagemaker
from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.feature_definition import (
    FeatureDefinition,
    FeatureTypeEnum,
)

region = os.environ["AWS_DEFAULT_REGION"]
print("Region: {}".format(region))

# Get the Role and Bucket before setting sm, featurestore_runtime, etc.
sts = boto3.Session(region_name=region).client(service_name="sts", region_name=region)

caller_identity = sts.get_caller_identity()
print("caller_identity: {}".format(caller_identity))

assumed_role_arn = caller_identity["Arn"]
print("(assumed_role) caller_identity_arn: {}".format(assumed_role_arn))

assumed_role_name = assumed_role_arn.split("/")[-2]

iam = boto3.Session(region_name=region).client(service_name="iam", region_name=region)
get_role_response = iam.get_role(RoleName=assumed_role_name)
print("get_role_response {}".format(get_role_response))
role = get_role_response["Role"]["Arn"]
print("role {}".format(role))

bucket = sagemaker.Session().default_bucket()
print("The DEFAULT BUCKET is {}".format(bucket))

# Setup sagemaker-featurestore-runtime
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

# Cast DataFrame Object to Supported Feature Store Data Type String
def cast_object_to_string(data_frame):
    for label in data_frame.columns:
        if data_frame.dtypes[label] == "object":
            data_frame[label] = data_frame[label].astype("str").astype("string")


# function to wait_for_feature_group_creation_complete
def wait_for_feature_group_creation_complete(feature_group):
    try:
        status = feature_group.describe().get("FeatureGroupStatus")
        print("Feature Group status: {}".format(status))
        while status == "Creating":
            print("Waiting for Feature Group Creation")
            time.sleep(5)
            status = feature_group.describe().get("FeatureGroupStatus")
            print("Feature Group status: {}".format(status))
        if status != "Created":
            print("Feature Group status: {}".format(status))
            raise RuntimeError(f"Failed to create feature group {feature_group.name}")
        print(f"FeatureGroup {feature_group.name} successfully created.")
    except:
        print("No feature group created yet.")

# function to create_or_load_feature_group
def create_or_load_feature_group(prefix, feature_group_name):

    # Feature Definitions for our records
    feature_definitions = [
        FeatureDefinition(feature_name="...", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="...", feature_type=FeatureTypeEnum.INTEGRAL),
		...
        FeatureDefinition(feature_name="split_type", feature_type=FeatureTypeEnum.STRING),
    ]

    feature_group = FeatureGroup(
        name=feature_group_name, feature_definitions=feature_definitions, sagemaker_session=sagemaker_session
    )

    print("Feature Group: {}".format(feature_group))

    try:
        print(
            "Waiting for existing Feature Group to become available if it is being created by another instance in our cluster..."
        )
        wait_for_feature_group_creation_complete(feature_group)
    except Exception as e:
        print("Before CREATE FG wait exeption: {}".format(e))
    #        pass

    try:
        record_identifier_feature_name = "..."
        event_time_feature_name = "..."

        print("Creating Feature Group with role {}...".format(role))
        feature_group.create(
            s3_uri=f"s3://{bucket}/{prefix}",
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
        print("Exception: {}".format(e))

    return feature_group


# process feature
def process_engineer_features():
	# feature engineering
	...
	
	# Add record to feature store
    df_fs_train_records = cast_object_to_string(df_train_records)
    df_fs_validation_records = cast_object_to_string(df_validation_records)
    df_fs_test_records = cast_object_to_string(df_test_records)

    print("Ingesting features...")
    feature_group.ingest(data_frame=df_fs_train_records, max_workers=3, wait=True)
    feature_group.ingest(data_frame=df_fs_validation_records, max_workers=3, wait=True)
    feature_group.ingest(data_frame=df_fs_test_records, max_workers=3, wait=True)
    
    offline_store_status = None
    while offline_store_status != 'Active':
        try:
            offline_store_status = feature_group.describe()['OfflineStoreStatus']['Status']
        except:
            pass
        print('Offline store status: {}'.format(offline_store_status))    
    print('...features ingested!')

# process 
def process(args):
    print("Current host: {}".format(args.current_host))

    feature_group = create_or_load_feature_group(
        prefix=args.feature_store_offline_prefix, feature_group_name=args.feature_group_name
    )

    feature_group.describe()

    print(feature_group.as_hive_ddl())

    train_data = "{}/.../train".format(args.output_data)
    validation_data = "{}/.../validation".format(args.output_data)
    test_data = "{}/.../test".format(args.output_data)

    transform_engineer = functools.partial(
        process_engineer_features,
        balance_dataset=..., # from arg
        prefix=..., # from arg
        feature_group_name=..., # from arg
    )

    input_files = glob.glob("{}/*.csv".format(args.input_data)) # change csv if need 

    num_cpus = multiprocessing.cpu_count()
    print("num_cpus {}".format(num_cpus))

    p = multiprocessing.Pool(num_cpus)
    p.map(transform_engineer, input_files)

    print("Listing contents of {}".format(args.output_data))
    dirs_output = os.listdir(args.output_data)
    for file in dirs_output:
        print(file)

    print("Listing contents of {}".format(train_data))
    dirs_output = os.listdir(train_data)
    for file in dirs_output:
        print(file)

    print("Listing contents of {}".format(validation_data))
    dirs_output = os.listdir(validation_data)
    for file in dirs_output:
        print(file)

    print("Listing contents of {}".format(test_data))
    dirs_output = os.listdir(test_data)
    for file in dirs_output:
        print(file)

    offline_store_contents = None
    while offline_store_contents is None:
        objects_in_bucket = s3.list_objects(Bucket=bucket, Prefix=args.feature_store_offline_prefix)
        if "Contents" in objects_in_bucket and len(objects_in_bucket["Contents"]) > 1:
            offline_store_contents = objects_in_bucket["Contents"]
        else:
            print("Waiting for data in offline store...\n")
            sleep(60)

    print("Data available.")

    print("Complete")

if __name__ == "__main__":
	...
```

5. Train : We may decide to use xgboost model and track experiment using mlflow  to see underfit/fit or overfit case and keeps the model's parameters then compare to other model later  you may need to create "src" folder which contain [.gitignore, inference.py, requirements.txt, test-local.sh, training.py]. We will use NVIDIA GPU to train the data

	5.1 training.py may contain something like this

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.exceptions import DataConversionWarning
from sklearn.compose import make_column_transformer

def _pip_install(pkg: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# ตาม requirement 
_pip_install("scikit-learn")
_pip_install("sagemaker==2.219.0")
_pip_install("xgboost")
_pip_install("mlflow==2.13.2")
_pip_install("sagemaker-mlflow==0.1.0")

# ใช้ alias 'mlf' แทน 'mlflow' เพื่อเลี่ยงชื่อชน
import mlflow as mlf

EXPERIMENT_NAME = "..."
MLFLOW_TRACKING_SERVER_ARN = ("...")

def load_csv_from_dir(..):
    ..

def prepare_features(..):
    ..

def train_model(..):
    # Load data
    ..
    # Save feature column order so inference.py can reuse it
    ..

    # Configure MLflow (Managed Tracking Server)
    mlf.set_tracking_uri(MLFLOW_TRACKING_SERVER_ARN)
    mlf.set_experiment(EXPERIMENT_NAME)

    # ใช้เวลาเป็นชื่อ run
    suffix = strftime("%d-%H-%M-%S", gmtime())
    run_name = f"training-{suffix}"

    params = {...}
    # Use GPU if available
    ..

    # Start MLflow run
    with mlf.start_run(
        run_name=run_name,
        description="training retail demand XGBoost model in SageMaker training job",
    ):
        # log params ทั้งหมด
        mlf.log_params(params)

        # log ชื่อ training job ไว้ใช้อ้างอิง
        training_job_name = os.environ.get("SM_TRAINING_JOB_NAME", args.current_host)
        mlf.log_param("sagemaker_training_job", training_job_name)
        ...

        # Train XGBoost model with eval_set + early stopping
        ...
        # Evaluate on train / val / test (final model)
        ...
        # Save model artifact for SageMaker hosting
        ...

        # Copy inference.py into /opt/ml/model/code/ for deployment
        inference_path = os.path.join(local_model_dir, "code/")
        print("Copying inference source files to {}".format(inference_path))
        os.makedirs(inference_path, exist_ok=True)
        os.system("cp inference.py {}".format(inference_path))
        print(glob(inference_path))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_data", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
	parser.add_argument("--validation_data", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
	parser.add_argument("--test_data", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--train_data", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--validation_data", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    parser.add_argument("--test_data", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--output_dir", type=str, default=os.environ["SM_OUTPUT_DIR"])
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current_host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--num_gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--checkpoint_base_path", type=str, default="/opt/ml/checkpoints")
	...
    # hyperparameters as arguments 
    ...

```
	5.2 inference.py may contain something like this
```python
def input_handler(data, context):
	...

def output_handler(response, context):
	...

```
	5.3 train processor

```python

import boto3
import sagemaker
import pandas as pd

sess = sagemaker.Session()
bucket = sess.default_bucket()
role = sagemaker.get_execution_role()
region = boto3.Session().region_name

% store -r processed_train_data_s3_uri
% store -r processed_validation_data_s3_uri
% store -r processed_test_data_s3_uri

sm = boto3.Session().client(service_name="sagemaker", region_name=region)

from sagemaker.inputs import TrainingInput

s3_input_train_data = TrainingInput(s3_data=processed_train_data_s3_uri,content_type =.. , distribution="ShardedByS3Key")
s3_input_validation_data = TrainingInput(s3_data=processed_validation_data_s3_uri, content_type =.. , distribution="ShardedByS3Key")
s3_input_test_data = TrainingInput(s3_data=processed_test_data_s3_uri, content_type =.. , distribution="ShardedByS3Key")

# Setup SageMaker Debugger
from sagemaker.debugger import Rule
from sagemaker.debugger import rule_configs
from sagemaker.debugger import ProfilerRule
from sagemaker.debugger import CollectionConfig
from sagemaker.debugger import DebuggerHookConfig

actions = rule_configs.ActionList(
    #    rule_configs.StopTraining(),
    #    rule_configs.Email("")
)
rules = [
    ProfilerRule.sagemaker(rule_configs.ProfilerReport()),  
	...
]

# Specify a Debugger profiler configuration
from sagemaker.debugger import ProfilerConfig, FrameworkProfile

profiler_config = ProfilerConfig(
    system_monitor_interval_millis=500,
    framework_profile_params=FrameworkProfile(local_path="/opt/ml/output/profiler/", start_step=5, num_steps=10),
)

# Specify Checkpoint S3 Location
checkpoint_s3_prefix = "checkpoints/{}".format(str(uuid.uuid4()))
checkpoint_s3_uri = "s3://{}/{}/".format(bucket, checkpoint_s3_prefix)

# Xgboost training 
sklearn_processor = SKLearnProcessor(
    framework_version="0.20.0", role=role, instance_type="...", instance_count=1
)

from sagemaker.processing import ProcessingInput, ProcessingOutput

sklearn_processor.run(
    code=...,
    inputs=[ProcessingInput(source=raw_data_location, destination="/opt/ml/processing/input")],
    outputs=[
        ProcessingOutput(
            output_name="train_data",
            source="/opt/ml/processing/train",
            destination="s3://" + train_data_location,
        ),
        ProcessingOutput(
            output_name="test_data",
            source="/opt/ml/processing/test",
            destination="s3://" + test_data_location,
        ),
        ProcessingOutput(
            output_name="train_data_headers",
            source="/opt/ml/processing/train_headers",
            destination="s3://" + rawbucket + "/" + prefix + "/train_headers",
        ),
    ],
```

6. Evaluate_Model_Metrics: we need to keep these matric using mlflow and you must create 'evaluate_model_metrics.py'

	6.1 Evaluate Model Metrics Script  

```python
from sagemaker.sklearn.processing import SKLearnProcessor

processor = SKLearnProcessor(
    framework_version="0.23-1",
    role=role,
    instance_type=processing_instance_type,
    instance_count=processing_instance_count,
    max_runtime_in_seconds=7200,
)

from sagemaker.processing import ProcessingInput, ProcessingOutput

processor.run(
    code="evaluate_model_metrics.py",
    inputs=[
        ProcessingInput(
            input_name="model-tar-s3-uri", source=model_dir_s3_uri, destination="/opt/ml/processing/input/model/"
        ),
        ProcessingInput(
            input_name="evaluation-data-s3-uri",
            source=raw_input_data_s3_uri,
            destination="/opt/ml/processing/input/data/",
        ),
    ],
    outputs=[
        ProcessingOutput(s3_upload_mode="EndOfJob", output_name="metrics", source="/opt/ml/processing/output/metrics"),
    ],
    arguments=[...],
    logs=True,
    wait=False,
)

```

	6.2 evaluate_model_metrics.py, this may contain code look like:

```python
import os
import sys
import json
import glob
import tarfile
import argparse
import subprocess
from pathlib import Path
from time import gmtime, strftime

def _pip_install(pkg: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

_pip_install("scikit-learn")
_pip_install("sagemaker==2.219.0")
_pip_install("xgboost")
_pip_install("mlflow==2.13.2")
_pip_install("sagemaker-mlflow==0.1.0")
_pip_install("matplotlib")


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow as mlf
import matplotlib.pyplot as plt
import sagemaker_mlflow  # activate SageMaker MLflow plugin

# MLflow configuration 
EXPERIMENT_NAME = "forcasting_demand_product"
MLFLOW_TRACKING_SERVER_ARN = (
    "arn:aws:sagemaker:us-east-1:423623839320:mlflow-tracking-server/tracking-server-demo"
)

def load_csv_from_dir(..):
	...

# model_dir จะมีไฟล์ model.tar.gz จาก training job เราจะแตกไฟล์ออกมาใน model_dir แล้ว return path กลับ

def extract_model_artifact(model_dir, ...):
	...

# โหลด model + feature_columns.json จาก model_dir
def load_model_and_features(args):
	...

#  Helper: bias metrics
def _safe_div(a, b):
    ..


def compute_bias_metrics(
    y_true_bin: np.ndarray,
    y_true_cont: np.ndarray,
    y_pred_cont: np.ndarray,
    facet: np.ndarray,
):
    """
    # คำนวณ bias metrics แบบง่าย ๆ:
        - DPL (labels): ต่างของ positive rate จริงระหว่างกลุ่ม
        - DPPL (predicted): ต่างของ positive rate ทำนายระหว่างกลุ่ม
        - RD (recall diff), AD (accuracy diff)
    """
    ..

def log_bias_metrics(..) :
    ..

# คำนวณ metrics / สร้างกราฟ / log ลง MLflow
def evaluate_and_log(model, feature_cols ,test_df, output_dir):
    ..

# Helper: SHAP explainability
def compute_and_log_shap(..):
    ..

# Helper: data profile for drift monitoring
def log_data_profile(..):
    ..

# Helper: evaluation summary (1 row per run – good for dashboards)
def write_evaluation_summary(..):
    ..

def main(args):
    # Configure MLflow
    mlf.set_tracking_uri(MLFLOW_TRACKING_SERVER_ARN)
    mlf.set_experiment(EXPERIMENT_NAME)
    suffix = strftime("%d-%H-%M-%S", gmtime())
    run_name = f"evaluate-{suffix}"

    with mlf.start_run(
            run_name=run_name,
            description="evaluate XGBoost model on test dataset in SageMaker processing job",
        ):
    # log อะไรเกี่ยวกับ job / model ไว้บ้าง
    ..

    # Load model + feature list
    ..

    # Load test data
    ..

    # Evaluate & log artifacts
    ..

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_data",
        type=str,
        default="/opt/ml/processing/test",
        help="Directory where test CSV is mounted",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/opt/ml/processing/model",
        help="Directory where model artifact (model.tar.gz) is mounted",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/opt/ml/processing/output/evaluation",
        help="Directory to store evaluation outputs (plots, csv, etc.)",
    )

    args = parser.parse_args()
    print("[EVAL] Arguments:", args)

    main(args)

```






7. Deploy: after the carefully create model and evaluate, we are going to deploy model we will experiment 3 ways  

	7.1 AmazonAthenaPreviewFunctionality: after deploy we can just use athena database and make prediction

	7.2 deploy using REST Endpoint

	7.3 Autoscaling a SageMaker Endpoint

	7.4 deploy using serverless 

8. Auto deploy:
```
SNS notification + auto‑deploy serverless after approval

Now, about the last part of your scenario:

“Then send email via SNS to confirm, wait until approved, then if Approved status appears automatically deploy as serverless.”

In practice, SageMaker Pipelines themselves don’t “wait days” for a manual approval – they’re meant to be finite runs. The common production pattern is:

Pipeline stops after RegisterModel (PendingManualApproval).

EventBridge + Lambda handle:

sending SNS emails when a new Pending model is created

automatically deploying a model when its state changes to Approved.

I’ll give you the Lambda code for the deploy‑on‑approval path; you can adapt it easily.

Lambda: deploy serverless endpoint when a Model Package becomes Approve
```

9. Stream:
 Great I can see that SageMaker Pipeline you start with Data prep / feature engineering. Why don't we start with ingestion like ingesting the data from new coming file trigger from s3? I know this is not proper but why? other option is to start with Data prep / feature engineering which i think is great but data source is raw_data_s3 which is fix bucket and file name and may not proper to do this as automatic life cycle I guess, since the data will be dynamic changing everyday, I think we need the better solution to handling ingestion part to be used for our model pipeline.

What my concern is we are going to split into 2 parts You will need to understand context of this application: #----------------------------------------------
First is the data engineer part: the data will be automatically store in s3 daily as the raw data, then ingest to athena daily.
So Athena will be our big database to be used to next part which is data science part

Second is the data science part: we will do the pipeline ***(this is our next step to do)***:
"""
Step: Preprocess – ProcessingStep (SKLearnProcessor) This will only pull data from Athena and select all data to be preprocess
Step: Train – TrainingStep (XGBoost Estimator using training.py)
Step: Evaluate – ProcessingStep (SKLearnProcessor, evaluate.py)
Step: IngestEval – ProcessingStep (SKLearnProcessor, ingest_to_athena.py)
Step: Condition – if RMSE ≤ threshold & accuracy OK → Register
Step: RegisterModel – RegisterModel → Model Registry, PendingManualApproval
"""
#----------------------------------------------

After we done all of these 2 steps then we are going to do data streaming using kinesis, lambda, cloudwatch and SNS for streaming production.
#----------------------------------------------
1. Transform Data in Kinesis Data Firehose delivery stream (data engineering part + completed data science model):
"""
ขั้นตอนการทำงาน (Step-by-Step)
1. การรับข้อมูลและเริ่มกระบวนการ (Input & Transformation Trigger)
* ข้อมูลดิบ (Input Data) ถูกส่งมาจากผู้ใช้งาน (User) ที่ส่งข้อมูลเข้ามา
* ข้อมูลนี้ประกอบด้วย:
* ข้อมูล columns ที่ใช้ในการ predict
* ในขั้นตอนนี้ Kinesis Firehose ถูกตั้งค่าให้เรียกใช้งานฟังก์ชัน Transform ก่อนบันทึกข้อมูล จึงส่งข้อมูลไปหา AWS Lambda
2. การส่งข้อมูลไปวิเคราะห์ (Request Prediction)
* AWS Lambda (ซึ่งเป็น Compute Service แบบ Serverless) จะทำหน้าที่เป็นตัวกลางในการประมวลผล
* Lambda จะดึงเอาเฉพาะข้อมูลคอลัมน์ที่จำเป็นในการ predict และส่งไปยัง Amazon SageMaker Endpoint
3. การประมวลผลโมเดล (Inference)
* Amazon SageMaker คือบริการสำหรับสร้างและรันโมเดล AI/Machine Learning
* มีโมเดลที่ถูกเทรนไว้แล้วรออยู่ที่ Endpoint “retail-demand-xgb-serverless-20251206-135312” โมเดลจะทำการอ่านข้อมูล และทำนาย
* SageMaker ส่งค่า predict กลับไปให้ AWS Lambda
4. การรวมข้อมูลและจัดเก็บ (Transformation & Output)
* AWS Lambda จะนำค่า predict ที่ได้ มาประกอบรวมกับข้อมูลชุดเดิม กลายเป็น Transformed Input Data
* Lambda ส่งข้อมูลที่สมบูรณ์แล้วกลับคืนสู่ Amazon Kinesis Data Firehose
* สุดท้าย Firehose จะบันทึกข้อมูลทั้งหมดลงในถังเก็บข้อมูล Amazon S3 เพื่อนำไปใช้งานต่อ (เช่น ทำ Analytics หรือ Dashboard)
"""
#----------------------------------------------
2. การคำนวณค่าสถิติและส่งข้อมูลไปเฝ้าระวัง (Aggregation & Monitoring): ภาพนี้เป็นส่วนขยายที่เจาะจงว่า หลังจากเรามีข้อมูล (หรือข้อมูลที่ผ่าน AI มาแล้ว) ไหลเข้ามาในระบบ เราจะทำอย่างไรเพื่อ สรุปผล และ สร้างกราฟแจ้งเตือน ให้คนดูแลระบบทราบครับ

“””
ขั้นตอนการทำงาน (Step-by-Step)
1. Input Streams (การรับข้อมูล)
* Users & Kinesis Data Firehose: ข้อมูล (ผู้ใช้งาน (User) ที่ส่งข้อมูลเข้ามา ไหลมาจากผู้ใช้เข้าสู่ท่อส่งข้อมูลหลัก (Firehose) เหมือนเดิมครับ
* จุดนี้คือการนำข้อมูลดิบ หรือข้อมูลที่ผ่านการประมวลผลเบื้องต้น ส่งต่อให้ส่วนวิเคราะห์
2. In-Application Streams (การประมวลผลภายในแอปฯ)
นี่คือ "สมอง" ส่วนที่คำนวณคณิตศาสตร์ครับ
* Amazon Kinesis Data Analytics Application: แอปพลิเคชันนี้จะรันคำสั่ง SQL ตลอดเวลา (Real-time SQL) และ จะใช้ฟังก์ชันทางสถิติขั้นสูง (บน AWS มักใช้ algorithm ที่เรียกว่า Random Cut Forest) เพื่อเรียนรู้ Pattern ของข้อมูล
* ANOMALY DETECTION: ระบบจะให้คะแนนข้อมูลทุกชุดที่ไหลเข้ามา เรียกว่า "Anomaly Score"
* PUMP: เป็นศัพท์ทางเทคนิคของการเขียน SQL ใน Kinesis Analytics ครับ หมายถึงกลไกที่ทำหน้าที่ "สูบ" หรือส่งต่อข้อมูลจากสตรีมหนึ่ง ไปใส่อีกสตรีมหนึ่งอย่างต่อเนื่อง
* AVG ... : นี่คือผลลัพธ์ของการคำนวณ แทนที่จะดูข้อมูลทีละครั้ง ระบบจะทำการ หาค่าเฉลี่ย (Average) 
3. Destinations (ปลายทาง)
เมื่อได้ค่าเฉลี่ยมาแล้ว จะส่งไปที่ไหนต่อ?
* AWS Lambda: ฟังก์ชันนี้จะรับค่า "ตัวเลขเฉลี่ย" นั้นมาครับ หน้าที่ของมันคือเป็นตัวแปลงหรือตัวกลาง เพื่อเตรียมส่งข้อมูลเข้าสู่ระบบ Monitoring
* Amazon CloudWatch: คือปลายทางสุดท้าย CloudWatch คือบริการสำหรับ Monitoring & Observability
    * ข้อมูลที่ส่งมาที่นี่จะถูกนำไปสร้างเป็น กราฟ (Graph/Dashboard) เพื่อให้ผู้บริหารดูแบบ Real-time
    * หรือตั้งค่า Alarms (แจ้งเตือน) เช่น "ถ้า ... ต่ำกว่า ... ให้ส่ง SMS เข้ามือถือ, Email หาผู้ดูแลระบบ”
* Amazon Simple Notification Service (SNS): คือบริการ "กระจายข่าว" ครับ
    * เมื่อ Lambda ส่งสัญญาณมา SNS จะทำการส่งแจ้งเตือนทันทีผ่านช่องทางต่างๆ เช่น SMS เข้ามือถือ, Email หาผู้ดูแลระบบ, หรือส่งเข้า Slack ของทีม Engineer
* 
“””
#----------------------------------------------