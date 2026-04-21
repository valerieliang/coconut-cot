# Coconut Steering Experiment

## File Structure & Functionality Reference

### Overview

This document describes every file in the `steering/` experiment directory. The experiment tests two hypotheses about activation steering in Coconut (continuous-thought) models trained on ProsQA:

- **H1:** Intermediate latent stages (L2–L5) are immune to steering because Coconut encodes a superposition of next steps at those positions. A single directional vector cannot perturb a distributed representation.
- **H2:** The first last (L5) latent step is a high-leverage injection points because L5 commits to an answer path.

### File Tree

```
root
├── steering_config.yaml
├── extract_vectors.py
├── run_steering_eval.py
└── plot_results.py
```

### File Reference

| File | Purpose | Key Outputs / Notes |
| :--- | :--- | :--- |
| `steering_config.yaml` | Central config for all experiment parameters | Edit this file to change alpha sweep, layer targets, or injection positions |
| `extract_vectors.py` | Builds per-position steering vectors from the trained model | Outputs `steering_vectors.pt` — one tensor per (layer, latent_position) pair |
| `run_steering_eval.py` | Injects vectors during inference and logs accuracy | Outputs `results.csv` with accuracy per position, alpha, and sample |
| `plot_results.py` | Reads `results.csv` and generates effect-size figures | Outputs `figures/effect_by_position.png` and `figures/alpha_sweep.png` |

### File Details

#### `steering_config.yaml`

Central configuration consumed by all three Python scripts. Keeps hyperparameters out of source code so runs are reproducible from the config alone.

**Key fields:**
- `model_path`: Path to trained Coconut checkpoint
- `val_path`: Path to `prosqa_valid.json`
- `train_path`: Path to `prosqa_train.json` (used for vector extraction)
- `layer`: Transformer layer index to hook into (e.g., 6)
- `latent_positions`: List of positions to test: `[1, 2, 3, 4, 5]`
- `alpha_sweep`: List of steering strengths: `[5, 10, 20, 40]`
- `n_contrast`: Number of correct/wrong pairs for vector extraction
- `output_dir`: Directory for `results.csv` and `figures/`

*Note: Change `layer` to probe different depths of the residual stream.*

#### `extract_vectors.py`

Builds one steering vector per latent position by computing a contrastive mean difference in the residual stream. For each sample in the training set, the script runs two forward passes:

1.  **Correct path:** The sample as-is, where the model reasons toward `target`.
2.  **Counterfactual path:** The sample with `target` swapped for `neg_target`, forcing the model down the wrong branch.

At the specified transformer layer and latent token index, the hidden state is extracted from both passes. The steering vector for position `Li` is calculated as:

```
v_i = mean(correct hidden states at Li) - mean(wrong hidden states at Li)
```

Vectors are saved to `steering_vectors.pt` as a dictionary keyed by position index. This file is consumed by `run_steering_eval.py`.

*ProsQA fields used:* `steps` (to identify how many latent tokens a sample has), `root` and `target` (correct path), `neg_target` (counterfactual), and `edges` (full graph for path verification).

#### `run_steering_eval.py`

The main evaluation loop. For each combination of (`latent_position`, `alpha`), it registers a PyTorch forward hook on the configured transformer layer that adds `alpha * v_i` to the residual stream at the latent token index, then runs inference over the full validation set.

For each sample the script records:
- `position`: Which latent step was perturbed (1–5)
- `alpha`: Steering strength used
- `correct`: `1` if model chose `target`, `0` if `neg_target` or other
- `chosen`: The actual token the model predicted
- `n_steps`: Number of reasoning steps in this sample

Results are appended to `output_dir/results.csv`. A baseline row (`alpha=0`, no hook) is included for every sample so effect sizes are computed relative to unperturbed accuracy.

**Hook registration pattern:**
```python
handle = model.transformer.h[layer].register_forward_hook(hook_fn)
# run inference
handle.remove()
```

The hook modifies `output[0]` (the hidden state tensor) at the precomputed sequence index for the target latent token. The index is derived from the tokenized input by locating the `<bot_token_id>` delimiters that Coconut inserts around continuous thoughts.

#### `plot_results.py`

Reads `results.csv` and produces two figures saved to `output_dir/figures/`:

1.  **`effect_by_position.png`**: Accuracy delta (steered minus baseline) for each latent position, averaged across all alpha values. This is the primary figure for testing both scenarios. Scenario 1 predicts near-zero bars at L2–L5; Scenario 2 predicts large negative bars at L1 and L5.
2.  **`alpha_sweep.png`**: Accuracy vs. alpha for each position on the same axes. Shows whether the effect is monotonic in alpha (expected for a meaningful steering direction) or saturates/reverses (suggests the vector is noisy or the position is resistant).

### Execution Order

Run the scripts in this order after training completes:

```bash
# 1. Extract steering vectors from the trained model
python steering/extract_vectors.py --config steering/steering_config.yaml

# 2. Run steering evaluation over the validation set
python steering/run_steering_eval.py --config steering/steering_config.yaml

# 3. Plot results
python steering/plot_results.py --config steering/steering_config.yaml
```