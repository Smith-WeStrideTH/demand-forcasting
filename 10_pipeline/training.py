import os
import sys
import json
import glob
import argparse
import subprocess
from pathlib import Path
from time import gmtime, strftime

# ----------------------------------------------------------------------
# Install required packages inside the training container
# ----------------------------------------------------------------------
def _pip_install(pkg: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# ตาม requirement ที่คุณระบุ
_pip_install("scikit-learn")
_pip_install("sagemaker==2.219.0")
_pip_install("xgboost")
_pip_install("mlflow==2.13.2")
_pip_install("sagemaker-mlflow==0.1.0")

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ใช้ alias 'mlf' แทน 'mlflow' เพื่อเลี่ยงชื่อชน
import mlflow as mlf

# ----------------------------------------------------------------------
# MLflow configuration
# ----------------------------------------------------------------------
EXPERIMENT_NAME = "forcasting_demand_product"
MLFLOW_TRACKING_SERVER_ARN = (
    "arn:aws:sagemaker:us-east-1:423623839320:mlflow-tracking-server/tracking-server-demo"
)


def load_csv_from_dir(directory: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(directory, "*.csv"))
    if not files:
        raise RuntimeError(f"No CSV files found in {directory}")
    print(f"Loading {files[0]}")
    return pd.read_csv(files[0])


def prepare_features(df: pd.DataFrame, label_col: str = "units_sold"):
    """
    Drop non-feature columns and split into X, y.
    Adjust this list if you change feature engineering.
    """
    drop_cols = [
        label_col,
        "record_id",
        "event_time",
        "split_type",
        "date",
        "holiday_name",
        "promo_type",
        "day_of_week",  # we will use day_of_week_index instead
        "high_demand"
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    print("Feature columns:", feature_cols)

    X = df[feature_cols].astype(np.float32)
    y = df[label_col].astype(np.float32)
    return X, y, feature_cols


def train_model(args):
    # Paths from SageMaker environment or args
    train_dir = args.train_data
    val_dir = args.validation_data
    test_dir = args.test_data
    output_dir = args.output_dir
    model_dir = args.model_dir

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("Train dir:", train_dir)
    print("Validation dir:", val_dir)
    print("Test dir:", test_dir)
    print("Output dir:", output_dir)
    print("Model dir:", model_dir)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    train_df = load_csv_from_dir(train_dir)
    val_df = load_csv_from_dir(val_dir)
    test_df = load_csv_from_dir(test_dir)

    X_train, y_train, feature_cols = prepare_features(train_df, label_col="units_sold")
    X_val, y_val, _ = prepare_features(val_df, label_col="units_sold")
    X_test, y_test, _ = prepare_features(test_df, label_col="units_sold")

    # Save feature column order so feature_columns.json can reuse it
    feature_cols_folder = os.path.join(output_dir, "features")
    os.makedirs(feature_cols_folder, exist_ok=True)
    feature_file_name = "feature_columns.json"
    feature_cols_path = os.path.join(feature_cols_folder, feature_file_name)
    with open(feature_cols_path, "w") as f:
        json.dump(feature_cols, f) 
    print("Saved feature column order to", feature_cols)

    # ------------------------------------------------------------------
    # 2. Configure MLflow (Managed Tracking Server)
    # ------------------------------------------------------------------
    # ใช้ SageMaker MLflow tracking server ARN
    mlf.set_tracking_uri(MLFLOW_TRACKING_SERVER_ARN)
    mlf.set_experiment(EXPERIMENT_NAME)

    # Hyperparameters (จาก Estimator.hyperparameters)
    params = {
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "reg_lambda": args.reg_lambda,
    }

    # Use GPU if available
    tree_method = "gpu_hist" if args.num_gpus and int(args.num_gpus) > 0 else "hist"
    params["tree_method"] = tree_method

    print("Training params:", params)

    # ใช้เวลาเป็นชื่อ run
    suffix = strftime("%d-%H-%M-%S", gmtime())
    run_name = f"training-{suffix}"

    # ------------------------------------------------------------------
    # 3. Start MLflow run
    # ------------------------------------------------------------------
    with mlf.start_run(
        run_name=run_name,
        description="training retail demand XGBoost model in SageMaker training job",
    ):
        # log params ทั้งหมด
        mlf.log_params(params)

        # log ชื่อ training job ไว้ใช้อ้างอิง
        training_job_name = os.environ.get("SM_TRAINING_JOB_NAME", args.current_host)
        mlf.log_param("sagemaker_training_job", training_job_name)

        # ------------------------------------------------------------------
        # 4. Train XGBoost model with eval_set + early stopping
        # ------------------------------------------------------------------
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            **params,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric="rmse",
            early_stopping_rounds=20,
            verbose=False,
        )

        # Learning curve per round (train/val RMSE)
        evals_result = model.evals_result()
        train_rmse_list = evals_result["validation_0"]["rmse"]
        val_rmse_list = evals_result["validation_1"]["rmse"]

        for i, (tr, vl) in enumerate(zip(train_rmse_list, val_rmse_list)):
            mlf.log_metric("train_rmse_round", tr, step=i)
            mlf.log_metric("val_rmse_round", vl, step=i)

        # Log best iteration and best val score (ถ้ามี)
        best_iter = getattr(model, "best_iteration", None)
        best_score = getattr(model, "best_score", None)
        if best_iter is not None:
            mlf.log_metric("best_iteration", int(best_iter))
        if best_score is not None:
            mlf.log_metric("best_val_rmse", float(best_score))

        # ------------------------------------------------------------------
        # 5. Evaluate on train / val / test (final model)
        # ------------------------------------------------------------------
        def eval_and_log(split_name, X, y):
            preds = model.predict(X)
            rmse = mean_squared_error(y, preds, squared=False)
            mae = mean_absolute_error(y, preds)
            r2 = r2_score(y, preds)
            print(f"{split_name} RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
            mlf.log_metric(f"{split_name}_rmse", rmse)
            mlf.log_metric(f"{split_name}_mae", mae)
            mlf.log_metric(f"{split_name}_r2", r2)
            return rmse, mae, r2

        train_rmse, train_mae, train_r2 = eval_and_log("train", X_train, y_train)
        val_rmse, val_mae, val_r2 = eval_and_log("val", X_val, y_val)
        test_rmse, test_mae, test_r2 = eval_and_log("test", X_test, y_test)

        # Generalization gap / fit-status heuristic
        gap = val_rmse - train_rmse
        ratio = val_rmse / (train_rmse + 1e-8)

        mlf.log_metric("generalization_gap_rmse", gap)
        mlf.log_metric("val_train_rmse_ratio", ratio)

        # simple heuristic thresholds — ปรับได้ตาม scale จริงของคุณ
        if ratio > 1.15 and gap > 5:
            fit_status = "overfit"
        elif ratio < 1.02:
            fit_status = "underfit"
        else:
            fit_status = "good_fit"

        mlf.log_param("fit_status", fit_status)
        print("Fit status:", fit_status)

        # log feature list เป็น artifact
        mlf.log_text("\n".join(feature_cols), "feature_columns.txt")

        # ------------------------------------------------------------------
        # 6. Save model artifact for SageMaker hosting
        # ------------------------------------------------------------------
        model_path = os.path.join(model_dir, "xgboost-model.bst")
        print(f"Saving XGBoost model to {model_path}")
        model.save_model(model_path)

        # (ออปชัน) ถ้าอยาก log model เข้า MLflow ด้วย xgboost plugin:
        try:
            import importlib
            mlf_xgb = importlib.import_module("mlflow.xgboost")
            mlf_xgb.log_model(model, "model")
        except Exception as e:
            print("Could not log MLflow XGBoost model:", e)

    # ----------------------------------------------------------------------
    # 7. Copy inference.py into /opt/ml/model/code/ for deployment
    # ----------------------------------------------------------------------
    local_output_dir = os.environ.get("SM_OUTPUT_DIR", output_dir)
    inference_path = os.path.join(local_output_dir, "code")
    os.makedirs(inference_path, exist_ok=True)
    print("Copying inference.py to", inference_path)
    os.system("cp inference.py {}".format(inference_path))
    print("Contents of code/:", os.listdir(inference_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation_data", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--test_data", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output_dir", type=str, default=os.environ.get("SM_OUTPUT_DIR", "/opt/ml/output"))
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ.get("SM_HOSTS", "[]")))
    parser.add_argument("--current_host", type=str, default=os.environ.get("SM_CURRENT_HOST", "unknown"))
    parser.add_argument("--num_gpus", type=int, default=int(os.environ.get("SM_NUM_GPUS", "0")))
    parser.add_argument("--checkpoint_base_path", type=str, default="/opt/ml/checkpoints")

    # XGBoost hyperparameters as arguments (will come from Estimator.hyperparameters)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--n_estimators", type=int, default=250)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--min_child_weight", type=float, default=1.0)
    parser.add_argument("--reg_lambda", type=float, default=1.0)

    args = parser.parse_args()
    print("Arguments:", args)

    train_model(args)
