# Usage Guide: Coconut Steering Experiments on ProsQA

## Overview

This guide provides step-by-step instructions for training a Coconut (Continuous Thought) model on a reasoning dataset and running steering experiments to test the steerability of latent representations at different positions in the reasoning chain.

---

## Part 1: Training the Coconut Model

### 1.1 Setup Environment

```bash
cd your-project/coconut
conda create --name coconut python=3.12
conda activate coconut
pip install -r requirements.txt
wandb login
```

### 1.2 Configure Training

Edit `args/your_dataset_coconut.yaml`:

```yaml
save_path: /path/to/your/checkpoints
bf16: True                    # Enable if GPU supports it
batch_size_training: 8        # Adjust based on GPU memory
gradient_accumulation_steps: 16
num_epochs: 50
max_latent_stage: 6          # Number of latent reasoning steps
epochs_per_stage: 5
uniform_prob: 0.0            # Set to 0.3 for ablation variant
train_path: data/your_train.json
val_path: data/your_valid.json
```

**Note:** Maintain effective batch size of ~128 (`batch_size_training × gradient_accumulation_steps × num_gpus`)

### 1.3 Train Model

```bash
torchrun --nnodes 1 --nproc_per_node <N_GPUS> run.py args/your_dataset_coconut.yaml
```

### 1.4 Select Best Checkpoint

Use wandb to find the checkpoint with highest validation accuracy:

```bash
wandb runs list --project your-project --order "-summary_metrics.val_accuracy"
```

---

## Part 2: Steering Experiments

### 2.1 Configure Steering Experiment

Create `steering/steering_config.yaml`:

```yaml
# Paths
model_path: /path/to/your/best_checkpoint
data_path: /path/to/your/validation_data.json
output_dir: /path/to/steering/outputs

# Experiment parameters
layer: 6                    # Transformer layer index (0 = first)
n_samples: 300              # Number of samples to evaluate
alpha_values: [5, 10, 20, 40]  # Steering strengths
positions: [1, 2, 3, 4, 5]  # Latent positions (1-indexed)

# Model parameters (must match training)
max_latent_stage: 6
latent_token_id: 50257
start_latent_id: 50258
end_latent_id: 50259
eos_token_id: 50256
```

### 2.2 Extract Steering Vectors

Computes steering vectors as: `mean(correct_path_hidden) - mean(wrong_path_hidden)` for each latent position.

```bash
python steering/extract_vectors.py --config steering/steering_config.yaml
```

**Output:** `steering/outputs/steering_vectors.pt`

### 2.3 Run Steering Evaluation

Injects steering vectors at different positions with varying strengths and measures impact on accuracy.

```bash
python steering/run_steering_eval.py --config steering/steering_config.yaml
```

**Output:** `steering/outputs/results.csv`

### 2.4 Generate Results

```bash
python plot_results.py --config steering/steering_config.yaml
```

**Output:** Accuracy vs position and accuracy vs alpha plots in `steering/outputs/figures/`

---

## Key Implementation Details

### How Coconut Processes Latent Thoughts

Unlike standard language models, Coconut does **not** place latent tokens in `input_ids`. Instead:
1. Runs `max_latent_stage` sequential forward passes
2. At each pass, the hidden state before the latent position becomes the "continuous thought"
3. This thought vector is fed back as the embedding for the next latent slot

### Where Steering Occurs

In `coconut.py:forward()`:
```python
thought = hidden_states[batch_idx, token_idx - 1 - hidden_states_offset, :]
```

Both extraction and evaluation scripts intercept `thought` here and optionally add `alpha × steering_vector`.

### Answer Extraction

**Important:** Always use `model.generate()` to produce full responses, then extract answer text after the `###` marker. Never compare predicted token IDs directly.

---

## Expected Results

Based on the theoretical hypotheses:

| Position | Expected Steering Effect |
|----------|-------------------------|
| L1 (first) | Strongest — perturbs start node |
| L2-L4 | Weakest — multiple paths averaged |
| L5 (last) | Strong — perturbs end node |

Vector extraction norms should decrease from L1 to L5.

---

## Dataset Requirements

Your dataset must include:
- `question`: The input query
- `answer`: The correct answer
- `neg_target`: A foil/wrong answer for contrastive pairs

---

## Complete Pipeline

```bash
# 1. Train model
cd your-project/coconut
torchrun --nnodes 1 --nproc_per_node 1 run.py args/your_dataset_coconut.yaml

# 2. Extract steering vectors
cd your-project
python steering/extract_vectors.py --config steering/steering_config.yaml

# 3. Run steering evaluation
python steering/run_steering_eval.py --config steering/steering_config.yaml

# 4. Generate plots
python plot_results.py --config steering/steering_config.yaml
```