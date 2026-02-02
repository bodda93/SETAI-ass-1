import numpy as np
import pandas as pd
import onnxruntime as ort

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score


DATA_PATH = "data/investigation_train_large_checked.csv"
ONNX_PATH = "model/good_model_tmp.onnx"   
TARGET_COL = "checked"

FLIP_COL = "ontheffing_actueel_ind"

THRESHOLD = 0.5
TEST_SIZE = 0.2
RANDOM_STATE = 1



def get_scores(session: ort.InferenceSession, X_np: np.ndarray) -> np.ndarray:
    """Return 1D positive-class scores from common ONNX classifier outputs."""
    input_name = session.get_inputs()[0].name
    outs = session.run(None, {input_name: X_np})

    # Probabilities (n,2) -> take positive class
    for o in outs:
        if isinstance(o, np.ndarray) and o.ndim == 2 and o.shape[1] >= 2:
            return o[:, 1].astype(float)

    # Scores (n,)
    for o in outs:
        if isinstance(o, np.ndarray) and o.ndim == 1:
            return o.astype(float)

    raise ValueError(f"Cannot extract scores. Output shapes: {[getattr(o,'shape',None) for o in outs]}")


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return roc_auc_score(y_true, scores) if len(np.unique(y_true)) == 2 else float("nan")


def safe_prauc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return average_precision_score(y_true, scores) if len(np.unique(y_true)) == 2 else float("nan")


def subgroup_report(scores_base, scores_flip, original_group, thr=0.5) -> pd.DataFrame:
    """Per-group deltas and decision flip rate."""
    abs_delta = np.abs(scores_flip - scores_base)

    base_pred = (scores_base >= thr).astype(int)
    flip_pred = (scores_flip >= thr).astype(int)
    dec_flip = (base_pred != flip_pred).astype(int)

    rows = []
    for g in [0, 1]:
        mask = (original_group == g)
        n = int(mask.sum())
        if n == 0:
            rows.append({
                "original_group": g,
                "n": 0,
                "mean_abs_delta": np.nan,
                "p95_abs_delta": np.nan,
                "decision_flip_rate@0.5": np.nan,
                "mean_base_score": np.nan,
                "mean_flip_score": np.nan,
            })
            continue

        rows.append({
            "original_group": g,
            "n": n,
            "mean_abs_delta": float(abs_delta[mask].mean()),
            "p95_abs_delta": float(np.percentile(abs_delta[mask], 95)),
            "decision_flip_rate@0.5": float(dec_flip[mask].mean()),
            "mean_base_score": float(scores_base[mask].mean()),
            "mean_flip_score": float(scores_flip[mask].mean()),
        })

    df_rep = pd.DataFrame(rows)
    df_rep = df_rep.rename(columns={"original_group": f"original_{FLIP_COL}"})
    return df_rep


def main(onnx_path: str = ONNX_PATH) -> None:
    print(f"Dataset: {DATA_PATH}")
    print(f"ONNX model: {onnx_path}")

    df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COL}")
    if FLIP_COL not in df.columns:
        raise ValueError(f"Flip column not found: {FLIP_COL}")

    y = df[TARGET_COL].astype(int).values

    X = df.drop(columns=[TARGET_COL, "Ja", "Nee"], errors="ignore")

    vals = pd.Series(X[FLIP_COL]).dropna().unique()
    if not set(np.array(vals, dtype=int)).issubset({0, 1}):
        raise ValueError(f"{FLIP_COL} is not binary 0/1. Unique values: {sorted(list(vals))[:20]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    X_test_np = X_test.astype(np.float32).values
    s_base = get_scores(session, X_test_np)
    yhat_base = (s_base >= THRESHOLD).astype(int)

    acc_base = accuracy_score(y_test, yhat_base)
    roc_base = safe_auc(y_test, s_base)
    pr_base = safe_prauc(y_test, s_base)

    print(f"BASELINE Accuracy: {acc_base:.4f}")
    print(f"BASELINE ROC-AUC:  {roc_base:.4f}")
    print(f"BASELINE PR-AUC:   {pr_base:.4f}")

    # Metamorphic flip
    X_flip = X_test.copy()
    original_group = X_flip[FLIP_COL].astype(int).values.copy()

    X_flip[FLIP_COL] = 1 - X_flip[FLIP_COL].astype(int)

    s_flip = get_scores(session, X_flip.astype(np.float32).values)
    yhat_flip = (s_flip >= THRESHOLD).astype(int)

    acc_flip = accuracy_score(y_test, yhat_flip)
    roc_flip = safe_auc(y_test, s_flip)
    pr_flip = safe_prauc(y_test, s_flip)

    print(f"AFTER FLIP Accuracy: {acc_flip:.4f}")
    print(f"AFTER FLIP ROC-AUC:  {roc_flip:.4f}")
    print(f"AFTER FLIP PR-AUC:   {pr_flip:.4f}")
    print()
    print("Differences (flip - baseline):")
    print(f"Delta Accuracy: {acc_flip - acc_base:+.6f}")
    print(f"Delta ROC-AUC:  {roc_flip - roc_base:+.6f}")
    print(f"Delta PR-AUC:   {pr_flip - pr_base:+.6f}")
    print()

    rep = subgroup_report(s_base, s_flip, original_group, thr=THRESHOLD)

    print(rep.to_string(index=False))
    print()
    print(rep.to_string(index=False))


if __name__ == "__main__":
    main()
