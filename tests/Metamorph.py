
import os
import numpy as np
import pandas as pd
import onnxruntime as ort

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

def main(onnx_path) -> pd.DataFrame:

    rstate = 1
    target = "checked"
    gender = "persoon_geslacht_vrouw"


    data_path = "data/investigation_train_large_checked.csv"
    # onnx_path = "model/gradient_boosting_model.onnx"

    print("Dataset:", data_path)
    print("ONNX model:", onnx_path)

    # Load dataset
    df = pd.read_csv(data_path)

    y = df[target].astype(int).values

    drop_cols = [target, "Ja", "Nee"]

    X = df.drop(columns=drop_cols)


    # Split only for evaluation (model already trained) # use same state for same split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rstate, stratify=y
    )

    # Load ONNX model
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Baseline predictions

    Xdf = X_test
    X_np = Xdf.astype(np.float32).values
    outputs = session.run([output_name], {input_name: X_np})[0]
    p_base = outputs

    yhat_base = (p_base >= 0.5).astype(int)

    acc_base = accuracy_score(y_test, yhat_base)
    roc_base = roc_auc_score(y_test, p_base)
    prauc_base = average_precision_score(y_test, p_base)

    print(f"BASELINE Accuracy: {acc_base:.4f}")
    print(f"BASELINE ROC-AUC:  {roc_base:.4f}")
    print(f"BASELINE PR-AUC:   {prauc_base:.4f}")

    # Apply gender flip
    X_test_flip = X_test.copy()
    X_test_flip[gender] = 1 - X_test_flip[gender]

    # p_flip = predict_proba_onnx(X_test_flip)
    X_np = X_test_flip.astype(np.float32).values
    outputs = session.run([output_name], {input_name: X_np})[0]
    p_flip = outputs

    yhat_flip = (p_flip >= 0.5).astype(int)

    delta = p_flip - p_base
    abs_delta = np.abs(delta)

    pd.Series({
        "n_test": len(X_test),
        "mean_abs_delta": abs_delta.mean(),
        "median_abs_delta": np.median(abs_delta),
        "p95_abs_delta": np.quantile(abs_delta, 0.95),
        "p99_abs_delta": np.quantile(abs_delta, 0.99),
        "max_abs_delta": abs_delta.max(),
        "decision_flip_rate@0.5": np.mean(yhat_flip != yhat_base),
    })

    # Metrics after flip (diagnostic)
    acc_flip = accuracy_score(y_test, yhat_flip)
    roc_flip = roc_auc_score(y_test, p_flip)
    prauc_flip = average_precision_score(y_test, p_flip)

    print(f"AFTER FLIP Accuracy: {acc_flip:.4f}")
    print(f"AFTER FLIP ROC-AUC:  {roc_flip:.4f}")
    print(f"AFTER FLIP PR-AUC:   {prauc_flip:.4f}")

    print("\nDifferences (flip - baseline):")
    print(f"Δ Accuracy: {acc_flip - acc_base:+.6f}")
    print(f"Δ ROC-AUC:  {roc_flip - roc_base:+.6f}")
    print(f"Δ PR-AUC:   {prauc_flip - prauc_base:+.6f}")

    # Delta analysis by original gender
    orig_gender = X_test[gender].values
    rows = []
    for g in [0, 1]:
        m = orig_gender == g
        rows.append({
            "original_gender": g,
            "n": int(m.sum()),
            "mean_abs_delta": abs_delta[m].mean(),
            "p95_abs_delta": np.quantile(abs_delta[m], 0.95),
            "decision_flip_rate@0.5": np.mean(yhat_flip[m] != yhat_base[m]),
            "mean_base_score": p_base[m].mean(),
            "mean_flip_score": p_flip[m].mean()
        })

    print(pd.DataFrame(rows))
    return pd.DataFrame(rows)

onnx_path = "model/good_model_tmp.onnx"
print(main(onnx_path))