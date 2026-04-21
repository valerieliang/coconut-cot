# Usage Guide: Training Coconut Model on ProntoQA

This document provides the setup and execution instructions for training a locally trained Coconut model on the ProntoQA dataset. Use this as a reference for future runs.

## Prerequisites

- Conda installed on your system
- At least one GPU (CUDA-compatible)
- Sufficient disk space for model checkpoints and datasets

## Environment Setup

### 1. Create and Activate Conda Environment

```bash
conda create --name coconut python=3.12
conda activate coconut
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Login to Weights & Biases (for experiment tracking)

```bash
wandb login
```

## Dataset Preparation

### Step 1: Generate ProntoQA Dataset

Clone the official ProntoQA repository and generate the dataset:

```bash
git clone https://github.com/asaparov/prontoqa
cd prontoqa
python run_experiment.py \
   --model-name json \
   --model-size dummy \
   --ordering random \
   --num-trials 10000 \
   --few-shot-examples 0 \
   --ontology true \
   --distractors none \
   --test-distractors none \
   --min-hops 5 \
   --max-hops 5 \
   --hops-skip 1
cd ..
```

### Step 2: Copy and Preprocess Dataset

```bash
cp prontoqa/5hop_0shot_random_trueontology_nodistractor.json coconut/data/5hop_0shot_random.json
cd coconut
python preprocessing/prontoqa.py
```

## Training

### Configure Training Parameters

Edit `args/prontoqa_coconut.yaml` to adjust:
- `save_path`: ./saved_models/prontoqa-coconut_ct1_ep5_ml6
- `batch_size_training`: Batch size per GPU
- `gradient_accumulation_steps`: Number of steps to accumulate gradients
- `model_id`: Base model (default: `openai-community/gpt2`)
- `debug`: Set to `true` for a quick sanity check run
- `max_latent_stage`: Controls training stages (sequential multi-epoch training)

### Launch Training

For single GPU:
```bash
torchrun --nnodes 1 --nproc_per_node 1 run.py args/prontoqa_coconut.yaml
```

For multiple GPUs (e.g., 4):
```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/prontoqa_coconut.yaml
```

**Important:** When changing the number of GPUs, adjust `gradient_accumulation_steps` to maintain the same effective batch size:
- Effective batch size = `batch_size_training × gradient_accumulation_steps × num_gpus`
- Example: Going from 4 GPUs to 1 GPU → multiply `gradient_accumulation_steps` by 4

## Evaluation

### Step 1: Locate Best Checkpoint

After training completes, find the checkpoint with the best validation accuracy in your `save_path` directory (specified in the training config).

### Step 2: Configure Evaluation

Edit `args/prosqa_coconut_eval.yaml` (note the filename uses "prosqa" not "prontoqa"):

```yaml
load_model_path: /path/to/your/best_checkpoint
```

### Step 3: Run Evaluation

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/prosqa_coconut_eval.yaml
```

Adjust `--nproc_per_node` based on your available GPUs.

## Important Notes

1. **Multi-stage Training**: The model uses multiple training stages controlled by `max_latent_stage`. The training will run several epochs sequentially - this is expected behavior.

2. **Checkpoint Selection**: Always use the checkpoint with the highest validation accuracy for evaluation, not necessarily the final checkpoint.

3. **Debug Mode**: Set `debug: true` in the YAML configuration for a quick validation of your setup before running full training.

4. **File Paths**: Ensure all paths in configuration files are absolute paths or correctly relative to the project root.

## Troubleshooting

- **Out of Memory**: Reduce `batch_size_training` or `gradient_accumulation_steps`
- **Slow Training**: Verify GPU utilization with `nvidia-smi`
- **Dataset Not Found**: Confirm the preprocessing step completed successfully and files exist in `coconut/data/`
- **Config File Not Found**: Ensure you're running commands from the `coconut` directory

## Expected Outputs

- **Training**: Model checkpoints saved in `save_path` directory
- **Evaluation**: Accuracy metrics logged to wandb and console output
- **Logs**: Training curves and metrics available in wandb dashboard
