# Coconut Steering Experiment

Tests whether activation steering affects Coconut's continuous thought representations
on the ProsQA graph-traversal reasoning task.

## Execution order (from ~/coconut-cot)

```bash
# Step 1: Extract steering vectors (fix contrastive method first — see NEXT_STEPS.md)
python steering/extract_vectors.py --config steering/steering_config.yaml

# Step 2: Run steering evaluation
python steering/run_steering_eval.py --config steering/steering_config.yaml

# Step 3: Generate plots and diagnostics
python steering/steering_analysis.py --config steering/steering_config.yaml
python plot_results.py --config steering/steering_config.yaml
```

## File structure

```
steering/
├── steering_config.yaml      single source of truth for all parameters
├── extract_vectors.py        contrastive vector extraction from latent passes
├── run_steering_eval.py      injects vectors during generate(), logs accuracy
├── steering_analysis.py      Word2Vec-style analogy plots + separability tests
├── plot_results.py           effect_by_position.png + alpha_sweep.png
└── outputs/
    ├── steering_vectors.pt   saved vectors (one per latent position)
    ├── results.csv           per-sample accuracy log (position × alpha × sample)
    └── figures/              analogy_L1-5.png, effect_by_position.png, alpha_sweep.png

diagnostic/
├── eval_checkpoint48.py      model load sanity check
├── verify_checkpoint48.py    single-sample generation test
└── debug_steering.py         thought norm inspection under injection
```

## Phase 1 results summary

See steering_results.docx for the full report.

- Baseline accuracy: 82.0% (246/300)
- All steered runs (alpha 5–40): delta = 0.000 at every position
- Root cause: steering vectors are noise (LogReg acc ~50%, frac_positive ~25%)
- Contrastive signal from question-word-swapping is too weak
- Alpha sweep was also 20–30× too small (needed ~800–1200 for 1:1 perturbation)

## Phase 2 plan

1. Fix extract_vectors.py to use model-error-based contrast (correct vs wrong predictions)
2. Verify clf_acc > 0.65 before running steering eval
3. Rerun with alpha sweep [100, 300, 600, 1200]
