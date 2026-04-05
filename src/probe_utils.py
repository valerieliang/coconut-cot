import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import json

def load_reps(path):
    with open(path) as f:
        return json.load(f)

def build_probe_dataset(records, step_idx, rep_key="step_reps"):
    """
    Collect (representation, label) pairs across all problems for a given step index.
    Label is 1 if the ground-truth step conclusion is True, 0 if False.
    Skips problems where extraction failed (rep is None) or step doesn't exist.
    """
    X, y, meta = [], [], []
    for rec in records:
        reps  = rec[rep_key]
        steps = rec["ground_truth_steps"]
        if step_idx >= len(reps) or step_idx >= len(steps):
            continue
        rep = reps[step_idx]
        if rep is None:
            continue
        # Ground-truth label: parse "True"/"False" from step string
        # Adjust this parsing based on your actual step format
        label = 1 if "True" in steps[step_idx] else 0
        X.append(rep)
        y.append(label)
        meta.append({"problem_id": rec["problem_id"], "n_hops": rec["n_hops"]})
    return np.array(X), np.array(y), meta

def train_probe(X, y, n_splits=5):
    """
    L2-regularized logistic regression with inner CV for C selection.
    Returns mean accuracy and 95% CI across outer folds.
    """
    if len(np.unique(y)) < 2:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "note": "only one class present"}

    probe = LogisticRegressionCV(
        Cs=[0.01, 0.1, 1.0, 10.0],
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
        max_iter=1000,
        scoring="balanced_accuracy",   # handles class imbalance
    )
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = cross_val_score(probe, X, y, cv=outer_cv, scoring="balanced_accuracy")

    ci = 1.96 * scores.std() / np.sqrt(len(scores))
    return {
        "mean":    float(scores.mean()),
        "ci_low":  float(scores.mean() - ci),
        "ci_high": float(scores.mean() + ci),
        "n":       len(y),
    }

def run_probe_grid(records, max_steps, rep_key="step_reps"):
    """
    Run probes for every step index up to max_steps.
    Returns a list of dicts: [{step, mean, ci_low, ci_high, n}, ...]
    """
    results = []
    for k in range(max_steps):
        X, y, _ = build_probe_dataset(records, step_idx=k, rep_key=rep_key)
        if len(X) < 10:
            print(f"  step {k}: too few samples ({len(X)}), skipping")
            continue
        result = train_probe(X, y)
        result["step"] = k
        print(f"  step {k}: acc={result['mean']:.3f} ± {(result['ci_high']-result['mean']):.3f}  (n={result['n']})")
        results.append(result)
    return results