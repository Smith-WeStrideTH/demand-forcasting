import os
import io
import json
import numpy as np
import pandas as pd
import xgboost as xgb

FEATURE_COLS_FILE = "feature_columns.json"


def model_fn(model_dir):
    """
    Load the XGBoost model + feature column order from model_dir.
    """
    # โหลดโมเดล
    model_path = os.path.join(model_dir, "xgboost-model.bst")
    model = xgb.XGBRegressor()
    model.load_model(model_path)

    # โหลด feature column order (ฟีเจอร์ที่ใช้ train เท่านั้น)
    feature_cols_path = os.path.join(model_dir, FEATURE_COLS_FILE)
    if os.path.exists(feature_cols_path):
        with open(feature_cols_path, "r") as f:
            feature_cols = json.load(f)
        print("Loaded feature column order:", feature_cols)
    else:
        feature_cols = None
        print("WARNING: feature_columns.json not found, will use raw input as-is.")

    # return object ที่รวมทั้ง model และ feature_cols
    return {"model": model, "feature_cols": feature_cols}


def input_fn(request_body, content_type):
    """
    Convert incoming request into something usable by predict_fn.

    - text/csv:
        * สำหรับ Clarify และ batch inference → headerless CSV
        * แปลงเป็น numpy.ndarray (N, n_features)

    - application/json:
        * รองรับทั้ง:
            {"instances": [ {"f1": .., "f2": ..}, ... ]} (dict)
            {"instances": [[...], [...]]} (list-of-lists)
    """
    if content_type == "text/csv":
        # Clarify ส่ง CSV ไม่มี header -> อ่านเป็น numeric array ตรง ๆ
        arr = np.genfromtxt(io.StringIO(request_body), delimiter=",")
        # ถ้ามีแค่ 1 แถว arr จะเป็น 1D -> reshape เป็น (1, n_features)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr.astype(np.float32)

    elif content_type == "application/json":
        data = json.loads(request_body)
        instances = data.get("instances", data)

        # กรณี list of dicts -> ใช้ชื่อคอลัมน์
        if isinstance(instances, list) and len(instances) > 0 and isinstance(instances[0], dict):
            df = pd.DataFrame(instances)
            return df

        # กรณี list of lists หรือ list of scalars
        arr = np.asarray(instances, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model_artifact):
    """
    Run prediction.

    input_data:
      - numpy.ndarray (จาก text/csv หรือ JSON list-of-lists)
      - หรือ pandas.DataFrame (จาก JSON list-of-dicts)

    model_artifact: {"model": xgb_model, "feature_cols": [...]}
    """
    model = model_artifact["model"]
    feature_cols = model_artifact.get("feature_cols")

    # กรณี Clarify/text-csv: เราได้ numpy array แล้ว → Assume order ถูกเตรียมมาให้ถูกแล้ว
    if isinstance(input_data, np.ndarray):
        X = input_data

    else:
        # กรณี JSON list-of-dicts: input_data เป็น DataFrame
        df = input_data.copy()

        if feature_cols is not None:
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required features in input: {missing}")
            X = df[feature_cols].astype(np.float32).values
        else:
            X = df.astype(np.float32).values

    preds = model.predict(X)
    return preds


def output_fn(prediction, accept):
    """
    Format prediction output.
    """
    if accept == "application/json":
        return json.dumps({"predictions": prediction.tolist()}), accept

    elif accept == "text/csv":
        out = io.StringIO()
        np.savetxt(out, prediction, delimiter=",")
        return out.getvalue(), accept

    else:
        raise ValueError(f"Unsupported accept type: {accept}")
