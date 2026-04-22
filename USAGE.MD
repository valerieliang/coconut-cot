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

Note: wandb is deprecated; need to update to be compatible with API keys.

```bash
pip install -r requirements.txt
pip install --upgrade wandb
```

### 3. Login to Weights & Biases (for experiment tracking)

```bash
wandb login
```

## Dataset Preparation