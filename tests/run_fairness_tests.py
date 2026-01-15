import pandas as pd
import numpy as np
import onnxruntime as ort

from sklearn.metrics import roc_auc_score


DATA_PATH = "data/investigation_train_large_checked.csv"
MODEL_PATH = "model/gradient_boosting_model.onnx"  # change to the ONNX you want to test
LABEL_COL = "checked"
GENDER_COL = "persoon_geslacht_vrouw"  # 1=female, 0=male
THRESHOLD = 0.5


def coerce_label(y: pd.Series) -> np.ndarray:
    if y.dtype == bool:
        return y.astype(int).to_numpy()
    if np.issubdtype(y.dtype, np.number):
        return (y.astype(float) > 0.5).astype(int).to_numpy()
    y = y.astype(str).str.strip().str.lower()
    return y.map({"ja": 1, "nee": 0, "1": 1, "0": 0, "true": 1, "false": 0}).astype(int).to_numpy()


def get_proba(session: ort.InferenceSession, X: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    outs = session.run(None, {input_name: X})
    for o in outs:
        if isinstance(o, np.ndarray) and o.ndim == 2 and o.shape[1] >= 2:
            return o[:, 1].astype(float)
    for o in outs:
        if isinstance(o, np.ndarray) and o.ndim == 1:
            return o.astype(float)
    raise ValueError(f"Cannot extract probabilities. Output shapes: {[getattr(o,'shape',None) for o in outs]}")


def confusion_rates(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    acc = float((y_true == y_pred).mean())
    return acc, tpr, fpr


def main():
    df = pd.read_csv(DATA_PATH)

    y = coerce_label(df[LABEL_COL])

    # Keep original interface: use all columns except label (and accidental label dummies if present)
    Xdf = df.drop(columns=[LABEL_COL], errors="ignore").drop(columns=["Ja", "Nee"], errors="ignore")

    # Must be numeric for ONNX input
    X = Xdf.to_numpy(dtype=np.float32)

    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

    p = get_proba(session, X)
    pred = (p >= THRESHOLD).astype(int)

    # Partitioning by gender
    female_mask = (Xdf[GENDER_COL].astype(int).to_numpy() == 1)
    male_mask = ~female_mask

    for name, mask in [("female", female_mask), ("male", male_mask)]:
        idx = np.where(mask)[0]
        y_g = y[idx]
        p_g = p[idx]
        pred_g = pred[idx]

        auc = float(roc_auc_score(y_g, p_g)) if len(np.unique(y_g)) == 2 else float("nan")
        acc, tpr, fpr = confusion_rates(y_g, pred_g)

        print(f"[Partition {name}] n={len(idx)}  acc={acc:.4f}  auc={auc:.4f}  tpr={tpr:.4f}  fpr={fpr:.4f}")

    # Metamorphic: flip gender
    Xflip = Xdf.copy()
    Xflip[GENDER_COL] = 1 - Xflip[GENDER_COL].astype(int)
    p2 = get_proba(session, Xflip.to_numpy(dtype=np.float32))
    pred2 = (p2 >= THRESHOLD).astype(int)

    print(f"[Metamorphic flip] flip_rate={(pred != pred2).mean():.6f}")
    print(f"[Metamorphic flip] mean_abs_delta_proba={np.mean(np.abs(p - p2)):.6f}")
    print(f"[Metamorphic flip] mean_delta_proba={np.mean(p2 - p):.6f}")


if __name__ == "__main__":
    main()
