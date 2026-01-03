# Chain of Reward (CoR) Training

This directory contains the training scripts for s1 models, including the new **Chain of Reward (CoR)** framework.

## Overview

CoR extends s1's training with:
1. **Multi-dimensional intrinsic rewards** - evaluates consistency, completeness, clarity, and format
2. **Endogenous self-evaluation** - model generates self-ratings during thinking, and we reward accurate self-assessment
3. **GRPO optimization** - Group Relative Policy Optimization for efficient policy learning

## Files

| File | Description |
|------|-------------|
| `sft.py` | Original s1 SFT training script |
| `sft.sh` | Shell script for SFT training |
| `grpo.py` | **NEW**: GRPO training with CoR rewards |
| `grpo.sh` | **NEW**: Shell script for GRPO training |
| `run_cor_pipeline.sh` | **NEW**: Complete CoR training pipeline |
| `rewards/` | **NEW**: Reward calculation module |

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure TRL >= 0.14.0 for GRPO support
pip install trl>=0.14.0
```

### Option 1: Full Pipeline

Run the complete CoR training pipeline:

```bash
bash train/run_cor_pipeline.sh
```

This will:
1. Prepare data with self-ratings
2. Run SFT (creates reference model)
3. Run GRPO with CoR rewards
4. Evaluate on AIME24, MATH500, GPQA

### Option 2: Step by Step

#### Step 1: Prepare Data with Self-Ratings

```bash
python data/add_self_ratings.py \
    --input_path simplescaling/s1K_tokenized \
    --output_path local_data/s1K_rated \
    --method rule
```

#### Step 2: SFT Training

```bash
bash train/sft.sh
```

#### Step 3: GRPO Training

```bash
# Edit grpo.sh to set ref_model_path to your SFT checkpoint
bash train/grpo.sh
```

## Reward Module (`rewards/`)

The `rewards/` directory contains the CoR reward calculation:

### Components

1. **`calculator.py`** - Main `RewardCalculator` class
   - Combines external (correctness) and intrinsic (thinking quality) rewards
   - Formula: `R(c) = R_ext(c) + λ * R_int(c)`

2. **`self_rating.py`** - Endogenous self-evaluation
   - `SelfRatingExtractor`: Extracts self-ratings from thinking chains
   - `SelfRatingEvaluator`: Evaluates calibration of self-ratings

3. **`intrinsic.py`** - Multi-dimensional intrinsic rewards
   - `ConsistencyReward`: Logical coherence
   - `CompletenessReward`: Step comprehensiveness
   - `ClarityReward`: Reasoning clarity
   - `FormatReward`: Structural correctness

### Usage

```python
from train.rewards import RewardCalculator, RewardConfig

# Create calculator with custom config
config = RewardConfig(
    lambda_intrinsic=1.0,  # Weight for intrinsic rewards
    self_rating_weight=0.2,  # Weight for self-rating quality
)
calculator = RewardCalculator(config)

# Calculate reward
output = calculator.calculate_total_reward(
    thinking_chain="Step 1: ... [Self-Rating: Consistency=8/10]",
    answer="42",
    ground_truth="42"
)

print(f"Total reward: {output.total_reward}")
print(f"External: {output.external_reward}")
print(f"Intrinsic: {output.intrinsic_reward}")
```

### Self-Rating Format

During thinking, models should generate self-ratings like:

```
Step 1: Analyze the problem...
[Self-Rating: Consistency=8/10, Completeness=9/10, Accuracy=7/10, Clarity=8/10]

Step 2: Apply the formula...
[Self-Rating: Consistency=9/10, Completeness=8/10, Accuracy=9/10, Clarity=9/10]

[Overall Quality: 8.5/10]
```

## Testing

Run unit tests:

```bash
python -m pytest train/rewards/test_rewards.py -v
```

## Hyperparameters

### CoR Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_intrinsic` | 1.0 | Weight for intrinsic rewards (λ) |
| `self_rating_weight` | 0.2 | Weight for self-rating quality reward |
| `calibration_bonus` | 0.2 | Bonus for high-high calibration alignment |

### GRPO Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_generations` | 4 | Candidates per input (N) |
| `epsilon` | 0.2 | Clipping range (δ) |
| `beta` | 0.01 | KL penalty coefficient (β) |
| `learning_rate` | 1e-6 | Learning rate |

## Theory

See the paper and `THEORY.md` for full mathematical derivation:

- **Total Reward**: `R(c) = R_ext(c) + λ * R_int(c)`
- **Intrinsic Reward**: `R_int(c) = Σ_d w_d * r_d(y_think) + w_self * r_self_rating_quality`
- **Calibration**: `cal_d(u, v) = 1 - |u - v|`
- **GRPO Objective**: Uses clipped surrogate with KL penalty

## Citation

```bibtex
@article{cor2025,
  title={CoR: Chain of Reward with Endogenous Self-Evaluation for Reasoning},
  author={Chen, Xingqiang},
  year={2025}
}
```
