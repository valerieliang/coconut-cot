import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import json


def extract_derived_concept(step_string):
    """Extract the concept derived at this reasoning step."""
    return step_string.rstrip(".").split()[-1]


def load_reps(path):
    with open(path) as f:
        return json.load(f)


def build_probe_dataset(records, step_idx, rep_key="step_reps"):
    """
    Collect (representation, label) pairs for a given step index.

    Label = the derived concept at step_idx (e.g. "scrompus", "rempus").
    This tests whether the model's representation at step k encodes
    exactly the concept reached at step k -- not the final answer,
    not a future step.

    Skips problems where:
    - step_idx exceeds the problem's hop count
    - representation extraction failed (None)
    """
    X, y, meta = [], [], []
    for rec in records:
        reps  = rec[rep_key]
        steps = rec["ground_truth_steps"]

        if step_idx >= len(steps) or step_idx >= len(reps):
            continue
        rep = reps[step_idx]
        if rep is None:
            continue

        label = extract_derived_concept(steps[step_idx])
        X.append(rep)
        y.append(label)
        meta.append({
            "problem_id": rec["problem_id"],
            "n_hops":     rec["n_hops"],
            "step_idx":   step_idx,
            "label":      label,
        })

    return np.array(X), np.array(y), meta


def train_probe(X, y, n_splits=5):
    """
    L2-regularized logistic regression with inner CV for C.
    Returns mean balanced accuracy + 95% CI across outer folds.
    Balanced accuracy handles class imbalance across concept labels.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    if len(np.unique(y_enc)) < 2:
        return {
            "mean": float("nan"), "ci_low": float("nan"),
            "ci_high": float("nan"), "note": "only one class"
        }

    probe = LogisticRegressionCV(
        Cs=[0.01, 0.1, 1.0, 10.0],
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
        max_iter=1000,
        scoring="balanced_accuracy",
        multi_class="multinomial",
    )
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = cross_val_score(
        probe, X, y_enc, cv=outer_cv, scoring="balanced_accuracy"
    )

    ci = 1.96 * scores.std() / np.sqrt(len(scores))
    return {
        "mean":    float(scores.mean()),
        "ci_low":  float(scores.mean() - ci),
        "ci_high": float(scores.mean() + ci),
        "n":       len(y),
        "n_classes": int(len(np.unique(y_enc))),
    }


def run_probe_grid(records, max_steps, rep_key="step_reps"):
    """
    Run probes for every step index and every hop count separately.
    Returns list of result dicts for plotting faithfulness trajectories.
    """
    results = []
    for k in range(max_steps):
        for n_hops in [3, 4, 5]:
            subset = [r for r in records if r["n_hops"] == n_hops]
            X, y, meta = build_probe_dataset(subset, step_idx=k, rep_key=rep_key)

            if len(X) < 10:
                continue

            result = train_probe(X, y)
            result["step"]   = k
            result["n_hops"] = n_hops
            print(
                f"  step={k} hops={n_hops}: "
                f"acc={result['mean']:.3f} "
                f"+/- {(result['ci_high']-result['mean']):.3f} "
                f"(n={result['n']}, classes={result['n_classes']})",
                flush=True,
            )
            results.append(result)
    return results


def run_cross_step_probe(records, rep_key="step_reps", max_steps=5):
    """
    Cross-step analysis: train probe to predict step-j's concept
    from step-k's representation, for all (k, j) pairs.
    
    This is the key faithfulness test:
    - High diagonal (k==j): representation at step k encodes step k's concept
    - High off-diagonal final (j==max): representation already encodes answer
      from early steps -> post-hoc rationalization pattern
    
    Returns a matrix of shape (max_steps, max_steps) of mean accuracies.
    """
    matrix = np.full((max_steps, max_steps), np.nan)

    for k in range(max_steps):
        for j in range(max_steps):
            X, y_rep, _ = build_probe_dataset(records, step_idx=k, rep_key=rep_key)
            if len(X) < 10:
                continue

            y_label = []
            X_filtered = []
            for rec in records:
                reps  = rec[rep_key]
                steps = rec["ground_truth_steps"]
                if k >= len(reps) or j >= len(steps) or reps[k] is None:
                    continue
                y_label.append(extract_derived_concept(steps[j]))
                X_filtered.append(reps[k])

            if len(X_filtered) < 10 or len(np.unique(y_label)) < 2:
                continue

            X_arr  = np.array(X_filtered)
            le     = LabelEncoder()
            y_enc  = le.fit_transform(y_label)
            probe  = LogisticRegressionCV(
                Cs=[0.01, 0.1, 1.0, 10.0],
                cv=3, max_iter=1000,
                scoring="balanced_accuracy",
                multi_class="multinomial",
            )
            outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
            scores = cross_val_score(probe, X_arr, y_enc, cv=outer_cv,
                                     scoring="balanced_accuracy")
            matrix[k, j] = scores.mean()
            print(f"  rep_step={k} label_step={j}: acc={scores.mean():.3f}", flush=True)

    return matrix