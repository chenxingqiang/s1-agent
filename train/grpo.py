"""
GRPO Training Script for Chain of Reward (CoR).

Extends s1's SFT training to use Group Relative Policy Optimization
with multi-dimensional intrinsic rewards.

Based on DESIGN.md Section 3.2.2 and THEORY.md Section 4.

Usage:
    python train/grpo.py --model_name Qwen/Qwen2.5-32B-Instruct \
                         --train_file_path simplescaling/s1K_tokenized \
                         --output_dir ckpts/cor-grpo
"""

import os
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import torch
from datasets import load_dataset, Dataset
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import trl
from trl import GRPOTrainer, GRPOConfig

from rewards import RewardCalculator, RewardConfig


@dataclass
class CoRTrainingConfig:
    """Configuration for CoR + GRPO training."""
    
    # Model
    model_name: str = field(default="Qwen/Qwen2.5-32B-Instruct")
    ref_model_name: Optional[str] = field(default=None)  # If None, uses model_name
    
    # Data
    train_file_path: str = field(default='simplescaling/s1K_tokenized')
    block_size: int = field(default=32768)
    
    # GRPO specific
    num_generations: int = field(default=4)  # N candidates per input
    
    # CoR reward configuration
    lambda_intrinsic: float = field(default=1.0)
    self_rating_weight: float = field(default=0.2)
    calibration_bonus: float = field(default=0.2)
    
    # W&B
    wandb_project: Optional[str] = field(default="cor-grpo")
    wandb_entity: Optional[str] = field(default=None)
    
    def __post_init__(self):
        if self.wandb_project:
            os.environ['WANDB_PROJECT'] = self.wandb_project
        if self.wandb_entity:
            os.environ['WANDB_ENTITY'] = self.wandb_entity


def create_reward_fn(config: CoRTrainingConfig):
    """Create reward function for GRPO trainer.
    
    Returns a callable compatible with TRL's GRPOTrainer.
    """
    reward_config = RewardConfig(
        lambda_intrinsic=config.lambda_intrinsic,
        self_rating_weight=config.self_rating_weight,
        calibration_bonus=config.calibration_bonus,
    )
    
    calculator = RewardCalculator(reward_config)
    
    def reward_fn(completions: List[str], **kwargs) -> List[float]:
        """Compute rewards for completions.
        
        Args:
            completions: List of model-generated completions.
            **kwargs: May include 'prompts', 'ground_truths', etc.
            
        Returns:
            List of reward values.
        """
        rewards = []
        
        ground_truths = kwargs.get('ground_truths', [None] * len(completions))
        
        for i, completion in enumerate(completions):
            # Parse completion into thinking and answer
            thinking, answer = parse_qwen_completion(completion)
            
            gt = ground_truths[i] if i < len(ground_truths) else None
            
            if gt is None:
                # No ground truth - use intrinsic reward only
                intrinsic, _ = calculator.calculate_intrinsic_reward(
                    thinking,
                    include_self_rating=True,
                    final_answer_correct=False
                )
                rewards.append(intrinsic)
            else:
                output = calculator.calculate_total_reward(
                    thinking, answer, gt
                )
                rewards.append(output.total_reward)
        
        return rewards
    
    return reward_fn


def parse_qwen_completion(completion: str) -> tuple:
    """Parse Qwen-format completion into thinking and answer.
    
    Format: <|im_start|>assistant\n{thinking}<|im_end|>\n{answer}
    
    Returns:
        Tuple of (thinking, answer).
    """
    import re
    
    # Handle Qwen format
    if '<|im_end|>' in completion:
        parts = completion.split('<|im_end|>')
        thinking = parts[0].replace('<|im_start|>assistant\n', '').strip()
        answer = parts[1].strip() if len(parts) > 1 else ""
        return thinking, answer
    
    # Handle generic format - look for answer markers
    answer_patterns = [
        r'\n[Aa]nswer:\s*(.+?)$',
        r'[Tt]he (?:final )?answer is[:\s]+(.+?)$',
        r'\n[Ff]inal [Aa]nswer:\s*(.+?)$',
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, completion, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            thinking = completion[:match.start()].strip()
            return thinking, answer
    
    # Fallback: last line is answer
    lines = completion.strip().split('\n')
    if len(lines) > 1:
        return '\n'.join(lines[:-1]), lines[-1]
    
    return completion, completion


def prepare_dataset(dataset_path: str, tokenizer) -> Dataset:
    """Prepare dataset for GRPO training.
    
    Expects dataset with 'text' field containing prompts.
    """
    dataset = load_dataset(dataset_path)
    
    if 'train' in dataset:
        dataset = dataset['train']
    
    # Extract prompts from text field
    def extract_prompt(example):
        text = example.get('text', '')
        
        # Extract user prompt from Qwen format
        if '<|im_start|>user' in text:
            import re
            match = re.search(
                r'<\|im_start\|>user\n(.+?)<\|im_end\|>',
                text,
                re.DOTALL
            )
            if match:
                return {'prompt': match.group(1).strip()}
        
        # Fallback
        return {'prompt': text}
    
    return dataset.map(extract_prompt)


def train():
    """Main training function."""
    
    # Parse arguments
    parser = transformers.HfArgumentParser((CoRTrainingConfig, GRPOConfig))
    config, grpo_args = parser.parse_args_into_dataclasses()
    
    log_config = {**asdict(config), **asdict(grpo_args)}
    logging.info(f"Training config: {log_config}")
    
    # Load model
    logging.info(f"Loading model: {config.model_name}")
    
    model_kwargs = {}
    if "70B" in config.model_name or "32B" in config.model_name:
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
            "use_cache": False,
        }
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs
    )
    
    # Load reference model (for KL penalty)
    ref_model_name = config.ref_model_name or config.model_name
    logging.info(f"Loading reference model: {ref_model_name}")
    
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_name,
        **model_kwargs
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    
    if "Qwen" in config.model_name:
        tokenizer.pad_token = "<|fim_pad|>"
    elif "Llama" in config.model_name:
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    
    # Prepare dataset
    logging.info(f"Loading dataset: {config.train_file_path}")
    dataset = prepare_dataset(config.train_file_path, tokenizer)
    logging.info(f"Dataset size: {len(dataset)}")
    
    # Create reward function
    logging.info("Creating reward function...")
    reward_fn = create_reward_fn(config)
    
    # Configure GRPO
    grpo_args.max_completion_length = config.block_size
    
    # Initialize trainer
    logging.info("Initializing GRPOTrainer...")
    
    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        args=grpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )
    
    # Train
    logging.info("Starting training...")
    trainer.train()
    
    # Save
    logging.info(f"Saving model to {grpo_args.output_dir}")
    trainer.save_model(grpo_args.output_dir)
    tokenizer.save_pretrained(grpo_args.output_dir)
    
    logging.info("Training complete!")


if __name__ == "__main__":
    train()
