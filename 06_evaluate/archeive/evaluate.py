import os
import sys
import json
import glob
import tarfile
import argparse
import subprocess
from pathlib import Path
from time import gmtime, strftime

# ----------------------------------------------------------------------
# Install required packages inside the evaluation container
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# MLflow configuration (reuse the same values as training)
# ----------------------------------------------------------------------
EXPERIMENT_NAME = "forcasting_demand_product"
MLFLOW_TRACKING_SERVER_ARN = (
    "arn:aws:sagemaker:us-east-1:423623839320:mlflow-tracking-server/tracking-server-demo"
)


def load_csv_from_dir(directory: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(directory, "*.csv"))
    if not files:
        raise RuntimeError(f"No CSV files found in {directory}")
    print(f"[EVAL] Loading test CSV: {files[0]}")
    return pd.read_csv(files[0])


def extract_model_artifact(model_dir: str) -> str:
    """
    model_dir จะมีไฟล์ model.tar.gz จาก training job
    เราจะแตกไฟล์ออกมาใน model_dir แล้ว return path กลับ
    """
    tar_files = glob.glob(os.path.join(model_dir, "*.tar.gz"))
    if not tar_files:
        # เผื่อกรณีมีไฟล์ถูกแตกไว้แล้ว (เช่น xgboost-model.bst อยู่ตรง ๆ)
        print("[EVAL] No model.tar.gz found, assuming model files already extracted.")
        return model_dir

    tar_path = tar_files[0]
    print(f"[EVAL] Extracting model artifact: {tar_path}")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=model_dir)

    return model_dir


def load_model_and_features(model_dir: str):
    """
    โหลด XGBoost model + feature_columns.json จาก model_dir
    """
    model_dir = extract_model_artifact(model_dir)

    model_path = os.path.join(model_dir, "xgboost-model.bst")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = xgb.XGBRegressor()
    model.load_model(model_path)
    print("[EVAL] Loaded model from", model_path)

    feature_cols_path = os.path.join(model_dir, "feature_columns.json")
    if not os.path.exists(feature_cols_path):
        raise FileNotFoundError(f"feature_columns.json not found at {feature_cols_path}")

    with open(feature_cols_path, "r") as f:
        feature_cols = json.load(f)
    print("[EVAL] Loaded feature column order:", feature_cols)

    return model, feature_cols


def evaluate_and_log(
    model,
    feature_cols,
    test_df: pd.DataFrame,
    output_dir: str,
):
    """
    คำนวณ metrics / สร้างกราฟ / log ลง MLflow
    """
    os.makedirs(output_dir, exist_ok=True)

    # เตรียม X_test, y_test
    X_test = test_df[feature_cols].astype(np.float32)
    y_test = test_df["units_sold"].astype(np.float32)

    preds = model.predict(X_test)

    # ---- Metrics ----
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"[EVAL] Test RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

    # log metrics
    mlf.log_metric("test_rmse", rmse)
    mlf.log_metric("test_mae", mae)
    mlf.log_metric("test_r2", r2)

    # ---- Save predictions CSV (test with prediction) ----
    out_df = test_df.copy()
    out_df["prediction"] = preds
    pred_csv_path = os.path.join(output_dir, "test_predictions.csv")
    out_df.to_csv(pred_csv_path, index=False)
    print("[EVAL] Saved predictions to", pred_csv_path)
    mlf.log_artifact(pred_csv_path)

    # ---- Plot 1: Prediction vs Actual scatter ----
    plt.figure()
    plt.scatter(y_test, preds, alpha=0.3)
    plt.xlabel("Actual units_sold")
    plt.ylabel("Predicted units_sold")
    plt.title("Predicted vs Actual on Test Set")
    scatter_path = os.path.join(output_dir, "pred_vs_actual.png")
    plt.savefig(scatter_path, bbox_inches="tight")
    plt.close()
    print("[EVAL] Saved scatter plot to", scatter_path)
    mlf.log_artifact(scatter_path)

    # ---- Plot 2: Residual histogram ----
    residuals = preds - y_test
    plt.figure()
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual (prediction - actual)")
    plt.ylabel("Count")
    plt.title("Residual Distribution on Test Set")
    resid_path = os.path.join(output_dir, "residual_hist.png")
    plt.savefig(resid_path, bbox_inches="tight")
    plt.close()
    print("[EVAL] Saved residual plot to", resid_path)
    mlf.log_artifact(resid_path)

    # NOTE: สำหรับ regression ไม่มี ROC จริง ๆ
    # ถ้าในอนาคตเป็นปัญหา classification คุณสามารถเพิ่ม:
    #   fpr, tpr, thresholds = roc_curve(y_true, y_score)
    #   figure_path = plot_roc_curve(...) -> save -> mlf.log_artifact(figure_path)
    # ตาม pattern เดียวกันนี้ได้เลย


def main(args):
    # -------------------------------
    # 1. Configure MLflow
    # -------------------------------
    mlf.set_tracking_uri(MLFLOW_TRACKING_SERVER_ARN)
    mlf.set_experiment(EXPERIMENT_NAME)

    suffix = strftime("%d-%H-%M-%S", gmtime())
    run_name = f"evaluate-{suffix}"

    with mlf.start_run(
        run_name=run_name,
        description="evaluate XGBoost model on test dataset in SageMaker processing job",
    ):
        # log อะไรเกี่ยวกับ job / model ไว้บ้าง
        processing_job_name = os.environ.get("AWS_PROCESSING_JOB_NAME", "unknown")
        mlf.log_param("processing_job_name", processing_job_name)
        mlf.log_param("model_source", args.model_dir)
        mlf.log_param("test_data_source", args.test_data)

        # -------------------------------
        # 2. Load model + feature list
        # -------------------------------
        model, feature_cols = load_model_and_features(args.model_dir)

        # -------------------------------
        # 3. Load test data
        # -------------------------------
        test_df = load_csv_from_dir(args.test_data)

        # -------------------------------
        # 4. Evaluate & log artifacts
        # -------------------------------
        evaluate_and_log(
            model=model,
            feature_cols=feature_cols,
            test_df=test_df,
            output_dir=args.output_dir,
        )


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
