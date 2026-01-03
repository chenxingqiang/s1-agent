"""
Main Reward Calculator for Chain of Reward (CoR).

Combines external (task) rewards and intrinsic (thinking quality) rewards.
Based on DESIGN.md Section 3.2.1 and THEORY.md Section 2.

Total reward formula:
R(c) = R_ext(c) + λ * R_int(c)

Where R_int includes:
- Multi-dimensional quality scores
- Self-rating quality (endogenous) reward
"""

import re
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

from .self_rating import SelfRatingExtractor, SelfRatingEvaluator
from .intrinsic import IntrinsicRewardCalculator


@dataclass  
class RewardConfig:
    """Configuration for reward calculation.
    
    Based on DESIGN.md Section 8.1.
    """
    # Weight balancing intrinsic vs external rewards (λ)
    lambda_intrinsic: float = 1.0
    
    # Weights for intrinsic dimensions
    dimension_weights: Dict[str, float] = field(default_factory=lambda: {
        "consistency": 0.25,
        "completeness": 0.25,
        "clarity": 0.25,
        "format": 0.25,
    })
    
    # Weight for self-rating quality reward
    self_rating_weight: float = 0.2
    
    # Calibration bonus for high-high alignment
    calibration_bonus: float = 0.2


@dataclass
class RewardOutput:
    """Output from reward calculation."""
    total_reward: float
    external_reward: float
    intrinsic_reward: float
    self_rating_reward: float
    dimension_scores: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class RewardCalculator:
    """Calculate rewards for Chain of Reward training.
    
    Combines:
    1. External reward: Binary correctness of final answer
    2. Intrinsic reward: Multi-dimensional thinking quality
    3. Self-rating reward: Quality of model's self-evaluation
    
    Based on DESIGN.md Section 3.2.1 and THEORY.md.
    """
    
    def __init__(self, config: Optional[RewardConfig] = None):
        """Initialize reward calculator.
        
        Args:
            config: Reward calculation configuration.
        """
        self.config = config or RewardConfig()
        
        self.intrinsic_calculator = IntrinsicRewardCalculator(
            weights=self.config.dimension_weights
        )
        self.self_rating_extractor = SelfRatingExtractor()
        self.self_rating_evaluator = SelfRatingEvaluator(
            calibration_bonus=self.config.calibration_bonus
        )
    
    def calculate_external_reward(
        self,
        answer: str,
        ground_truth: str,
        grader_fn: Optional[callable] = None
    ) -> float:
        """Calculate external (task) reward.
        
        From THEORY.md Definition 4:
        R_ext(c) = I[y_answer == y_gt]
        
        Args:
            answer: Model's final answer.
            ground_truth: Correct answer.
            grader_fn: Optional custom grading function.
                       If None, uses exact string matching.
                       
        Returns:
            1.0 if correct, 0.0 if incorrect.
        """
        if grader_fn is not None:
            return float(grader_fn(answer, ground_truth))
        
        # Default: normalized string comparison
        answer_clean = self._normalize_answer(answer)
        gt_clean = self._normalize_answer(ground_truth)
        
        return 1.0 if answer_clean == gt_clean else 0.0
    
    def _normalize_answer(self, text: str) -> str:
        """Normalize answer for comparison."""
        if text is None:
            return ""
        
        text = text.strip().lower()
        
        # Remove common prefixes
        prefixes = [
            "the answer is",
            "answer:",
            "final answer:",
            "therefore,",
            "thus,",
            "so,",
        ]
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Remove punctuation from end
        text = text.rstrip('.,;:!?')
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def calculate_intrinsic_reward(
        self,
        thinking_chain: str,
        include_self_rating: bool = True,
        final_answer_correct: bool = False,
        **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate intrinsic (thinking quality) reward.
        
        From THEORY.md Definition 5:
        R_int(c) = Σ_d w_d * r_d(y_think) + w_self * r_self_rating_quality
        
        Args:
            thinking_chain: Model's thinking process.
            include_self_rating: Whether to include self-rating quality reward.
            final_answer_correct: Whether final answer was correct (for calibration).
            **kwargs: Additional context for reward functions.
            
        Returns:
            Tuple of (intrinsic_reward, dimension_scores).
        """
        # Calculate multi-dimensional rewards
        weighted_intrinsic, dim_scores = self.intrinsic_calculator.compute_weighted_reward(
            thinking_chain, **kwargs
        )
        
        # Calculate self-rating quality reward
        self_rating_reward = 0.0
        if include_self_rating:
            actual_qualities = self.intrinsic_calculator.get_actual_qualities(
                thinking_chain, **kwargs
            )
            self_rating_reward = self.self_rating_evaluator.compute_self_rating_reward(
                thinking_chain,
                actual_qualities,
                final_answer_correct
            )
            dim_scores["self_rating_quality"] = self_rating_reward
        
        # Combine: weighted intrinsic + self-rating
        # Normalize so total is still in [0, 1]
        if include_self_rating:
            total_weight = 1.0 + self.config.self_rating_weight
            intrinsic_reward = (
                weighted_intrinsic + 
                self.config.self_rating_weight * self_rating_reward
            ) / total_weight
        else:
            intrinsic_reward = weighted_intrinsic
        
        return intrinsic_reward, dim_scores
    
    def calculate_total_reward(
        self,
        thinking_chain: str,
        answer: str,
        ground_truth: str,
        grader_fn: Optional[callable] = None,
        **kwargs
    ) -> RewardOutput:
        """Calculate total reward for a reasoning chain.
        
        From THEORY.md Definition 3:
        R(c) = R_ext(c) + λ * R_int(c)
        
        Args:
            thinking_chain: Model's thinking process.
            answer: Model's final answer.
            ground_truth: Correct answer.
            grader_fn: Optional custom grading function.
            **kwargs: Additional context.
            
        Returns:
            RewardOutput with all reward components.
        """
        # External reward
        external = self.calculate_external_reward(answer, ground_truth, grader_fn)
        
        # Intrinsic reward (with self-rating quality)
        intrinsic, dim_scores = self.calculate_intrinsic_reward(
            thinking_chain,
            include_self_rating=True,
            final_answer_correct=(external > 0.5),
            **kwargs
        )
        
        # Self-rating reward (for separate tracking)
        self_rating = dim_scores.get("self_rating_quality", 0.0)
        
        # Total reward
        total = external + self.config.lambda_intrinsic * intrinsic
        
        return RewardOutput(
            total_reward=total,
            external_reward=external,
            intrinsic_reward=intrinsic,
            self_rating_reward=self_rating,
            dimension_scores=dim_scores,
            metadata={
                "answer": answer,
                "ground_truth": ground_truth,
                "has_self_rating": self.self_rating_extractor.has_self_ratings(thinking_chain),
            }
        )
    
    def __call__(
        self,
        thinking_chain: str,
        answer: str,
        ground_truth: str,
        **kwargs
    ) -> float:
        """Convenience method to get total reward as float.
        
        For use with TRL's reward function interface.
        """
        output = self.calculate_total_reward(
            thinking_chain, answer, ground_truth, **kwargs
        )
        return output.total_reward


def create_reward_function(config: Optional[RewardConfig] = None):
    """Create a reward function for TRL GRPOTrainer.
    
    Returns a callable that takes completions and returns rewards.
    """
    calculator = RewardCalculator(config)
    
    def reward_fn(
        completions: list,
        prompts: list = None,
        ground_truths: list = None,
        **kwargs
    ) -> list:
        """TRL-compatible reward function.
        
        Args:
            completions: List of model completions.
            prompts: List of input prompts.
            ground_truths: List of correct answers.
            
        Returns:
            List of reward values.
        """
        rewards = []
        
        ground_truths = ground_truths or [None] * len(completions)
        
        for i, completion in enumerate(completions):
            # Parse completion into thinking and answer
            thinking, answer = parse_completion(completion)
            
            gt = ground_truths[i] if ground_truths else None
            
            if gt is None:
                # No ground truth - only use intrinsic reward
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


def parse_completion(completion: str) -> Tuple[str, str]:
    """Parse completion into thinking and answer parts.
    
    Supports formats:
    - Qwen: <|im_start|>assistant\n{thinking}<|im_end|>\n{answer}
    - Generic: {thinking}\n\nAnswer: {answer}
    """
    # Try Qwen format
    if '<|im_end|>' in completion:
        parts = completion.split('<|im_end|>')
        thinking = parts[0].replace('<|im_start|>assistant\n', '').strip()
        answer = parts[1].strip() if len(parts) > 1 else ""
        return thinking, answer
    
    # Try "Answer:" format
    if '\nAnswer:' in completion or '\nanswer:' in completion.lower():
        match = re.search(r'\n[Aa]nswer:\s*(.+?)$', completion, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            thinking = completion[:match.start()].strip()
            return thinking, answer
    
    # Try "The answer is" format
    match = re.search(r'[Tt]he (?:final )?answer is[:\s]+(.+?)$', completion, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        thinking = completion[:match.start()].strip()
        return thinking, answer
    
    # Fallback: entire completion is thinking, last line is answer
    lines = completion.strip().split('\n')
    if len(lines) > 1:
        return '\n'.join(lines[:-1]), lines[-1]
    
    return completion, completion
