
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

    print("Dataset:", data_path)
    print("ONNX model:", onnx_path)

    # Load dataset
    df = pd.read_csv(data_path)

    y = df[target].astype(int).values

    drop_cols = [target, "Ja", "Nee"]

    X = df.drop(columns=drop_cols)


    # Split only for evaluation (model already trained) same state same split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rstate, stratify=y
    )

    # Load ONNX model
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name


    X_np = X_test.astype(np.float32).values
    outputs = session.run([output_name], {input_name: X_np})[0]


    yhat_base = (outputs >= 0.5).astype(int)

    acc_base = accuracy_score(y_test, yhat_base)
    roc_base = roc_auc_score(y_test, outputs)
    prauc_base = average_precision_score(y_test, outputs)

    print(f"BASELINE Accuracy: {acc_base:.4f}")
    print(f"BASELINE ROC-AUC:  {roc_base:.4f}")
    print(f"BASELINE PR-AUC:   {prauc_base:.4f}")


    def safe_roc_auc(y_true, p):
        # ROC-AUC requires both classes present in the subgroup
        return roc_auc_score(y_true, p) if len(np.unique(y_true)) == 2 else np.nan
    
    def safe_pr_auc(y_true, p):
        # PR-AUC is defined even with one class, but can be uninformative; keep it for completeness
        return average_precision_score(y_true, p) if len(np.unique(y_true)) == 2 else np.nan
    
    report = []
    for g in [0, 1]:
        mask = (X_test[gender].values == g)
        y_g = y_test[mask]
        p_g = outputs[mask]
        yhat_g = (p_g >= 0.5).astype(int)
    
        report.append({
            "gender_value": g,
            "n": int(mask.sum()),
            "positive_rate": float(y_g.mean()) if len(y_g) else np.nan,
            "accuracy": float(accuracy_score(y_g, yhat_g)) if len(y_g) else np.nan,
            "roc_auc": float(safe_roc_auc(y_g, p_g)) if len(y_g) else np.nan,
            "pr_auc": float(safe_pr_auc(y_g, p_g)) if len(y_g) else np.nan,
            "mean_score": float(np.mean(p_g)) if len(p_g) else np.nan,
            "selection_rate@0.5": float(np.mean(p_g >= 0.5)) if len(p_g) else np.nan,
        })
    
    print(pd.DataFrame(report).to_csv())

onnx_path = "model/good_model_tmp.onnx"
print(main(onnx_path))
