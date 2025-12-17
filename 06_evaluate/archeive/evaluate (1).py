import os
import sys
import json
import glob
import tarfile
import argparse
import subprocess
from pathlib import Path
from time import gmtime, strftime
from datetime import datetime

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
_pip_install("shap==0.44.0")

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow as mlf
import matplotlib.pyplot as plt
import shap
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


# ----------------------------------------------------------------------
# Helper: bias metrics
# ----------------------------------------------------------------------
def _safe_div(a, b):
    return float(a) / float(b) if b not in (0, 0.0) else float("nan")


def compute_bias_metrics(
    y_true_bin: np.ndarray,
    y_true_cont: np.ndarray,
    y_pred_cont: np.ndarray,
    facet: np.ndarray,
):
    """
    คำนวณ bias metrics แบบง่าย ๆ:
      - DPL (labels): ต่างของ positive rate จริงระหว่างกลุ่ม
      - DPPL (predicted): ต่างของ positive rate ทำนายระหว่างกลุ่ม
      - RD (recall diff), AD (accuracy diff)
    จะถือว่ามี 2 กลุ่มใน facet (เช่น is_weekend=0,1)
    """
    y_true_bin = np.asarray(y_true_bin).astype(int)
    y_true_cont = np.asarray(y_true_cont).astype(float)
    y_pred_cont = np.asarray(y_pred_cont).astype(float)
    facet = np.asarray(facet)

    # infer threshold ที่ใช้แบ่ง high/low จากข้อมูลจริง:
    # ใช้ min(units_sold | high_demand=1) ถ้ามี, ไม่งั้นใช้ p75
    positives = y_true_cont[y_true_bin == 1]
    if positives.size > 0:
        threshold = float(np.min(positives))
    else:
        threshold = float(np.quantile(y_true_cont, 0.75))

    y_pred_bin = (y_pred_cont >= threshold).astype(int)

    groups = np.unique(facet)
    if groups.size < 2:
        return {
            "note": "only one facet group present, cannot compute group differences",
            "threshold_units_sold": threshold,
        }

    # สมมติ group_a = group ที่ค่าต่ำกว่า, group_d = สูงกว่า (เช่น 0 = weekday, 1 = weekend)
    group_a, group_d = sorted(groups.tolist())[:2]

    def group_stats(g):
        mask = facet == g
        y = y_true_bin[mask]
        yhat = y_pred_bin[mask]
        n = y.size
        if n == 0:
            return {
                "n": 0,
                "pos_rate_true": float("nan"),
                "pos_rate_pred": float("nan"),
                "recall": float("nan"),
                "precision": float("nan"),
                "accuracy": float("nan"),
            }
        TP = int(((y == 1) & (yhat == 1)).sum())
        FP = int(((y == 0) & (yhat == 1)).sum())
        TN = int(((y == 0) & (yhat == 0)).sum())
        FN = int(((y == 1) & (yhat == 0)).sum())
        pos_rate_true = _safe_div((y == 1).sum(), n)
        pos_rate_pred = _safe_div((yhat == 1).sum(), n)
        recall = _safe_div(TP, TP + FN)
        precision = _safe_div(TP, TP + FP)
        accuracy = _safe_div(TP + TN, n)
        return {
            "n": n,
            "pos_rate_true": pos_rate_true,
            "pos_rate_pred": pos_rate_pred,
            "recall": recall,
            "precision": precision,
            "accuracy": accuracy,
        }

    stats_a = group_stats(group_a)
    stats_d = group_stats(group_d)

    metrics = {
        "threshold_units_sold": threshold,
        "group_a_value": float(group_a),
        "group_d_value": float(group_d),
        # group a
        "group_a_n": stats_a["n"],
        "group_a_pos_rate_true": stats_a["pos_rate_true"],
        "group_a_pos_rate_pred": stats_a["pos_rate_pred"],
        "group_a_recall": stats_a["recall"],
        "group_a_precision": stats_a["precision"],
        "group_a_accuracy": stats_a["accuracy"],
        # group d
        "group_d_n": stats_d["n"],
        "group_d_pos_rate_true": stats_d["pos_rate_true"],
        "group_d_pos_rate_pred": stats_d["pos_rate_pred"],
        "group_d_recall": stats_d["recall"],
        "group_d_precision": stats_d["precision"],
        "group_d_accuracy": stats_d["accuracy"],
        # differences (a - d)
        "DPL_labels": stats_a["pos_rate_true"] - stats_d["pos_rate_true"],
        "DPPL_predicted": stats_a["pos_rate_pred"] - stats_d["pos_rate_pred"],
        "RD_recall_diff": stats_a["recall"] - stats_d["recall"],
        "AD_accuracy_diff": stats_a["accuracy"] - stats_d["accuracy"],
    }

    return metrics


def log_bias_metrics(bias_metrics: dict, output_dir: str):
    # log เป็น artifact (JSON)
    os.makedirs(output_dir, exist_ok=True)
    bias_path = os.path.join(output_dir, "bias_metrics.json")
    with open(bias_path, "w") as f:
        json.dump(bias_metrics, f, indent=2)
    print("[EVAL] Saved bias metrics to", bias_path)
    mlf.log_artifact(bias_path)

    # log เป็น MLflow metrics
    for k, v in bias_metrics.items():
        # ข้าม field ที่ไม่ใช่ numeric
        if isinstance(v, (int, float)) and not np.isnan(v):
            mlf.log_metric(f"bias_{k}", float(v))


# ----------------------------------------------------------------------
# Helper: SHAP explainability
# ----------------------------------------------------------------------
def compute_and_log_shap(
    model,
    X_test: pd.DataFrame,
    feature_cols,
    output_dir: str,
    max_samples: int = 300,
):
    """
    คำนวณ SHAP values (TreeExplainer) แล้ว log:
      - shap_feature_importance.csv
      - shap_feature_importance.png
      - shap_values_sample.npy
    """
    os.makedirs(output_dir, exist_ok=True)

    # sample เพื่อไม่ให้หนักเกินไป
    if len(X_test) > max_samples:
        X_sample = X_test.sample(max_samples, random_state=42)
    else:
        X_sample = X_test

    print(f"[EVAL][SHAP] Using {len(X_sample)} samples for SHAP analysis")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # global importance = mean(|shap|) ต่อ feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame(
        {"feature": feature_cols, "mean_abs_shap": mean_abs_shap}
    ).sort_values("mean_abs_shap", ascending=False)

    # CSV
    shap_csv_path = os.path.join(output_dir, "shap_feature_importance.csv")
    importance_df.to_csv(shap_csv_path, index=False)
    print("[EVAL][SHAP] Saved SHAP feature importance CSV to", shap_csv_path)
    mlf.log_artifact(shap_csv_path)

    # Bar plot (top 20)
    top_k = min(20, len(importance_df))
    top_imp = importance_df.head(top_k)
    plt.figure(figsize=(8, max(4, top_k * 0.3)))
    plt.barh(top_imp["feature"][::-1], top_imp["mean_abs_shap"][::-1])
    plt.xlabel("Mean |SHAP value|")
    plt.title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    shap_png_path = os.path.join(output_dir, "shap_feature_importance.png")
    plt.savefig(shap_png_path, bbox_inches="tight")
    plt.close()
    print("[EVAL][SHAP] Saved SHAP bar plot to", shap_png_path)
    mlf.log_artifact(shap_png_path)

    # Save raw shap values sample (numpy)
    shap_values_path = os.path.join(output_dir, "shap_values_sample.npy")
    np.save(shap_values_path, shap_values)
    print("[EVAL][SHAP] Saved SHAP values sample to", shap_values_path)
    mlf.log_artifact(shap_values_path)


# ----------------------------------------------------------------------
# Helper: data profile for drift monitoring
# ----------------------------------------------------------------------
def log_data_profile(
    df: pd.DataFrame,
    feature_cols,
    label_col: str,
    output_dir: str,
):
    """
    สร้าง data profile ง่าย ๆ สำหรับเช็ค drift:
      - n_rows
      - summary stats (mean, std, min, max, p25/p50/p75) ของ numeric features + label
    """
    os.makedirs(output_dir, exist_ok=True)

    profile = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "n_rows": int(len(df)),
    }

    # label summary
    if label_col in df.columns:
        y = df[label_col]
        if np.issubdtype(y.dtype, np.number):
            profile[f"label_{label_col}"] = {
                "mean": float(y.mean()),
                "std": float(y.std()),
                "min": float(y.min()),
                "max": float(y.max()),
                "p25": float(y.quantile(0.25)),
                "p50": float(y.quantile(0.5)),
                "p75": float(y.quantile(0.75)),
            }
            # log metrics label-level
            mlf.log_metric("label_mean", float(y.mean()))
            mlf.log_metric("label_std", float(y.std()))

    # feature summary
    profile["features"] = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
        s = df[col]
        if not np.issubdtype(s.dtype, np.number):
            # ข้าม non-numeric (เช่น category/string)
            continue
        profile["features"][col] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "p25": float(s.quantile(0.25)),
            "p50": float(s.quantile(0.5)),
            "p75": float(s.quantile(0.75)),
        }

    profile_path = os.path.join(output_dir, "data_profile.json")
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)
    print("[EVAL] Saved data profile to", profile_path)
    mlf.log_artifact(profile_path)

    # metrics สำหรับจำนวนแถว / ฯลฯ
    mlf.log_metric("n_rows", int(len(df)))


# ----------------------------------------------------------------------
# Main evaluate function
# ----------------------------------------------------------------------
def evaluate_and_log(
    model,
    feature_cols,
    test_df: pd.DataFrame,
    output_dir: str,
):
    """
    คำนวณ metrics / bias / SHAP / data profile แล้ว log ลง MLflow
    """
    os.makedirs(output_dir, exist_ok=True)

    # เตรียม X_test, y_test
    X_test = test_df[feature_cols].astype(np.float32)
    y_test = test_df["units_sold"].astype(np.float32)

    # ---------------- Basic regression metrics ----------------
    preds = model.predict(X_test)

    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"[EVAL] Test RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

    # log metrics
    mlf.log_metric("test_rmse", rmse)
    mlf.log_metric("test_mae", mae)
    mlf.log_metric("test_r2", r2)

    # log basic label stats (ซ้ำกับ data_profile ก็ได้)
    mlf.log_metric("test_units_sold_mean", float(y_test.mean()))
    mlf.log_metric("test_units_sold_std", float(y_test.std()))

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

    # ---------------- Bias metrics (manual) ----------------
    if ("high_demand" in test_df.columns) and ("is_weekend" in test_df.columns):
        print("[EVAL][BIAS] Computing bias metrics using high_demand & is_weekend")
        bias_metrics = compute_bias_metrics(
            y_true_bin=test_df["high_demand"].astype(int).values,
            y_true_cont=y_test.values,
            y_pred_cont=preds,
            facet=test_df["is_weekend"].astype(int).values,
        )
        log_bias_metrics(bias_metrics, output_dir)
    else:
        print("[EVAL][BIAS] high_demand or is_weekend not found, skipping bias metrics")

    # ---------------- SHAP explainability ----------------
    try:
        compute_and_log_shap(model, X_test, feature_cols, output_dir)
    except Exception as e:
        print(f"[EVAL][SHAP] Failed to compute SHAP values: {e}")

    # ---------------- Data profile for drift monitoring ----------------
    log_data_profile(test_df, feature_cols, label_col="units_sold", output_dir=output_dir)


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
        # 4. Evaluate + bias + SHAP + drift
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
