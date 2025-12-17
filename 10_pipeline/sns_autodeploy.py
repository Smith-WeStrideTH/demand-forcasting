# deploy_retail_demand_on_approval.py

import os
import time
import json
import boto3
from datetime import datetime, timezone

sm = boto3.client("sagemaker")
region = os.environ.get("AWS_REGION", "us-east-1")

# IAM role that SageMaker will use to run the model
SAGEMAKER_EXECUTION_ROLE_ARN = os.environ["SAGEMAKER_EXECUTION_ROLE_ARN"]

def lambda_handler(event, context):
    """
    EventBridge rule target.
    Triggered on SageMaker Model Package State Change.
    """
    print("Event:", json.dumps(event))

    detail = event.get("detail", {})
    model_package_arn = detail.get("ModelPackageArn")
    status = detail.get("ModelApprovalStatus")

    if not model_package_arn:
        print("No ModelPackageArn in event; ignoring.")
        return {"statusCode": 200, "body": "No model package ARN"}

    print(f"ModelPackageArn: {model_package_arn}")
    print(f"New status: {status}")

    # Only deploy on Approved
    if status != "Approved":
        print("Status is not Approved; nothing to do.")
        return {"statusCode": 200, "body": "Not approved status"}

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base_name = "retail-demand-xgb"

    model_name = f"{base_name}-model-{ts}"
    endpoint_config_name = f"{base_name}-serverless-config-{ts}"
    endpoint_name = os.environ.get(
        "ENDPOINT_NAME",
        f"{base_name}-serverless",
    )

    # 1) Create Model referencing the Model Package
    print("Creating model:", model_name)
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={"ModelPackageName": model_package_arn},
        ExecutionRoleArn=SAGEMAKER_EXECUTION_ROLE_ARN,
    )

    # 2) Create EndpointConfig with ServerlessInferenceConfig
    print("Creating endpoint config:", endpoint_config_name)
    sm.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "ModelName": model_name,
                "VariantName": "AllTraffic",
                "ServerlessConfig": {
                    "MemorySizeInMB": 4096,
                    "MaxConcurrency": 5,
                },
            }
        ],
    )

    # 3) Create or update Endpoint
    try:
        desc = sm.describe_endpoint(EndpointName=endpoint_name)
        status_ep = desc["EndpointStatus"]
        print("Existing endpoint status:", status_ep)
        # Update existing endpoint
        print("Updating endpoint:", endpoint_name)
        sm.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
    except sm.exceptions.ClientError as e:
        if "Could not find endpoint" in str(e) or "does not exist" in str(e):
            print("Endpoint does not exist; creating new endpoint:", endpoint_name)
            sm.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name,
            )
        else:
            raise

    print("Done. Endpoint:", endpoint_name)
    return {
        "statusCode": 200,
        "body": f"Deployed/updated endpoint {endpoint_name} for {model_package_arn}",
    }
