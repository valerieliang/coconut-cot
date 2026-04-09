# src/probe_utils.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from collections import defaultdict
import json
import warnings


def extract_derived_concept(step_string):
    """Extract the concept derived at this reasoning step."""
    return step_string.rstrip(".").split()[-1]


def load_reps(path):
    with open(path) as f:
        return json.load(f)


def evaluate_binary_probe_similarity(records, rep_step_idx, label_step_idx, 
                                      rep_key="step_reps", n_negatives=10):
    """
    Binary evaluation using cosine similarity ranking.
    
    For each representation at rep_step_idx:
    - Compute cosine similarity to its true concept's centroid (average of reps with that concept)
    - Compare to similarity with n_negatives random other concepts
    - Returns: mean reciprocal rank, accuracy@1, and p-value
    """
    concept_reps = defaultdict(list)
    for rec in records:
        reps = rec[rep_key]
        steps = rec["ground_truth_steps"]
        
        if rep_step_idx >= len(reps) or label_step_idx >= len(steps):
            continue
        rep = reps[rep_step_idx]
        if rep is None:
            continue
            
        concept = extract_derived_concept(steps[label_step_idx])
        concept_reps[concept].append(rep)
    
    concept_centroids = {}
    for concept, rep_list in concept_reps.items():
        if len(rep_list) > 0:
            concept_centroids[concept] = np.mean(rep_list, axis=0)
    
    ranks = []
    correct_at_1 = 0
    total = 0
    
    for rec in records:
        reps = rec[rep_key]
        steps = rec["ground_truth_steps"]
        
        if rep_step_idx >= len(reps) or label_step_idx >= len(steps):
            continue
        rep = reps[rep_step_idx]
        if rep is None:
            continue
            
        true_concept = extract_derived_concept(steps[label_step_idx])
        if true_concept not in concept_centroids:
            continue
        
        other_concepts = [c for c in concept_centroids.keys() if c != true_concept]
        if len(other_concepts) < n_negatives:
            actual_negatives = len(other_concepts)
        else:
            actual_negatives = n_negatives
        
        if actual_negatives == 0:
            continue
            
        sampled_negatives = np.random.choice(other_concepts, actual_negatives, replace=False)
        candidates = [true_concept] + list(sampled_negatives)
        
        rep_array = np.array(rep).reshape(1, -1)
        similarities = {}
        for concept in candidates:
            centroid = concept_centroids[concept].reshape(1, -1)
            sim = cosine_similarity(rep_array, centroid)[0][0]
            similarities[concept] = sim
        
        ranked = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        rank = next(i for i, (c, _) in enumerate(ranked) if c == true_concept) + 1
        ranks.append(rank)
        
        if rank == 1:
            correct_at_1 += 1
        total += 1
    
    if total == 0:
        return {
            "mr": float("nan"), 
            "acc_at_1": float("nan"), 
            "p_value": float("nan"), 
            "n": 0,
            "chance": float("nan"),
            "n_negatives": n_negatives
        }
    
    mr = np.mean(1.0 / np.array(ranks))
    acc1 = correct_at_1 / total
    chance = 1.0 / (1 + n_negatives)
    
    if total > 0:
        p_value = stats.binomtest(correct_at_1, total, p=chance, alternative='greater').pvalue
    else:
        p_value = float("nan")
    
    return {
        "mr": float(mr),
        "acc_at_1": float(acc1),
        "p_value": float(p_value),
        "n": total,
        "chance": float(chance),
        "n_negatives": n_negatives
    }


def build_progress_aggregated_dataset(records, rep_key="step_reps"):
    """
    Group representations by reasoning progress (early, middle, late)
    rather than exact step index.
    """
    stages = {'early': [], 'middle': [], 'late': []}
    
    for rec in records:
        reps = rec[rep_key]
        steps = rec["ground_truth_steps"]
        n_hops = rec["n_hops"]
        
        for step_idx, (rep, step) in enumerate(zip(reps, steps)):
            if rep is None:
                continue
            
            progress = step_idx / n_hops if n_hops > 0 else 0
            
            if progress < 0.33:
                stage = 'early'
            elif progress < 0.66:
                stage = 'middle'
            else:
                stage = 'late'
            
            concept = extract_derived_concept(step)
            stages[stage].append({
                'rep': rep,
                'concept': concept,
                'problem_id': rec['problem_id'],
                'n_hops': n_hops,
                'step_idx': step_idx
            })
    
    return stages


def evaluate_progress_probe(stages, n_splits=5):
    """Evaluate probe accuracy for progress-aggregated stages."""
    results = {}
    
    for stage, items in stages.items():
        if len(items) < 4:
            results[stage] = {
                "mean": float("nan"), 
                "ci_low": float("nan"), 
                "ci_high": float("nan"), 
                "n": len(items),
                "n_classes": 0,
                "chance": float("nan")
            }
            continue
        
        X = np.array([item['rep'] for item in items])
        y = [item['concept'] for item in items]
        
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        n_classes = len(np.unique(y_enc))
        
        if n_classes < 2:
            results[stage] = {
                "mean": float("nan"), 
                "ci_low": float("nan"), 
                "ci_high": float("nan"), 
                "n": len(items),
                "n_classes": n_classes,
                "chance": float("nan")
            }
            continue
        
        min_class_count = int(np.bincount(y_enc).min())
        cv_splits = max(2, min(5, min_class_count))
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                probe = LogisticRegression(
                    C=1.0, 
                    max_iter=1000, 
                    multi_class="multinomial",
                    solver="lbfgs"
                )
                cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
                scores = cross_val_score(probe, X, y_enc, cv=cv, scoring="balanced_accuracy")
            
            ci = 1.96 * scores.std() / np.sqrt(len(scores))
            results[stage] = {
                "mean": float(scores.mean()),
                "ci_low": float(scores.mean() - ci),
                "ci_high": float(scores.mean() + ci),
                "n": len(items),
                "n_classes": n_classes,
                "chance": 1.0 / n_classes
            }
        except Exception as e:
            results[stage] = {
                "mean": float("nan"), 
                "ci_low": float("nan"), 
                "ci_high": float("nan"), 
                "n": len(items),
                "n_classes": n_classes,
                "chance": 1.0 / n_classes if n_classes > 0 else float("nan"),
                "error": str(e)
            }
    
    return results


def run_binary_probe_grid(records, max_steps=5, rep_key="step_reps", n_negatives=10):
    """Run binary similarity-based probe for each (step, hop) combination."""
    results = []
    for k in range(max_steps):
        for n_hops in [3, 4, 5]:
            subset = [r for r in records if r["n_hops"] == n_hops]
            
            if len(subset) < 4:
                continue
            
            eval_result = evaluate_binary_probe_similarity(
                subset, rep_step_idx=k, label_step_idx=k,
                rep_key=rep_key, n_negatives=n_negatives
            )
            
            eval_result["step"] = k
            eval_result["n_hops"] = n_hops
            
            acc = eval_result.get('acc_at_1', float('nan'))
            mr = eval_result.get('mr', float('nan'))
            n = eval_result.get('n', 0)
            chance = eval_result.get('chance', 0.091)
            p_val = eval_result.get('p_value', 1.0)
            
            sig = " **" if p_val < 0.05 else ""
            print(
                f"  step={k} hops={n_hops}: "
                f"acc@1={acc:.3f} "
                f"(MR={mr:.3f}, n={n}, "
                f"chance={chance:.3f}){sig}",
                flush=True
            )
            results.append(eval_result)
    
    return results


def run_binary_cross_step_probe(records, rep_key="step_reps", max_steps=5, n_negatives=10):
    """Cross-step matrix using binary similarity ranking."""
    matrix = np.full((max_steps, max_steps), np.nan)
    p_values = np.full((max_steps, max_steps), np.nan)
    
    for k in range(max_steps):
        for j in range(max_steps):
            eval_result = evaluate_binary_probe_similarity(
                records, rep_step_idx=k, label_step_idx=j,
                rep_key=rep_key, n_negatives=n_negatives
            )
            
            acc = eval_result.get('acc_at_1', float('nan'))
            n = eval_result.get('n', 0)
            p_val = eval_result.get('p_value', 1.0)
            
            if n > 0 and not np.isnan(acc):
                matrix[k, j] = acc
                p_values[k, j] = p_val
                
                sig = " **" if p_val < 0.05 else ""
                print(
                    f"  rep_step={k} label_step={j}: "
                    f"acc@1={acc:.3f} "
                    f"(n={n}, p={p_val:.3f}){sig}",
                    flush=True
                )
            else:
                print(f"  rep_step={k} label_step={j}: skipped (n={n})", flush=True)
    
    return matrix, p_values


def compare_models_statistical(verbal_results, coconut_results):
    """Statistical comparison between Verbal CoT and Coconut using diagonal accuracies."""
    verbal_diag = []
    coconut_diag = []
    
    for vr, cr in zip(verbal_results, coconut_results):
        if vr.get("step", 0) < vr.get("n_hops", 0):
            v_acc = vr.get("acc_at_1", float("nan"))
            c_acc = cr.get("acc_at_1", float("nan"))
            if not np.isnan(v_acc) and not np.isnan(c_acc):
                verbal_diag.append(v_acc)
                coconut_diag.append(c_acc)
    
    if len(verbal_diag) >= 3:
        t_stat, p_value = stats.ttest_rel(verbal_diag, coconut_diag)
        w_stat, w_p = stats.wilcoxon(verbal_diag, coconut_diag)
        
        return {
            "verbal_mean": float(np.mean(verbal_diag)),
            "coconut_mean": float(np.mean(coconut_diag)),
            "difference": float(np.mean(verbal_diag) - np.mean(coconut_diag)),
            "verbal_std": float(np.std(verbal_diag)),
            "coconut_std": float(np.std(coconut_diag)),
            "ttest_statistic": float(t_stat),
            "ttest_p": float(p_value),
            "wilcoxon_statistic": float(w_stat),
            "wilcoxon_p": float(w_p),
            "n_pairs": len(verbal_diag)
        }
    
    return {
        "verbal_mean": float("nan"),
        "coconut_mean": float("nan"),
        "difference": float("nan"),
        "n_pairs": 0,
        "note": "insufficient data for statistical test"
    }