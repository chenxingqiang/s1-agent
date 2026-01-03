"""
Chain of Reward (CoR) - Reward Calculation Module

Implements multi-dimensional reward system for CoR training.
Based on DESIGN.md and THEORY.md specifications.
"""

from .calculator import RewardCalculator, RewardConfig, RewardOutput
from .self_rating import SelfRatingExtractor, SelfRatingEvaluator, SelfRating
from .intrinsic import (
    IntrinsicRewardCalculator,
    ConsistencyReward,
    CompletenessReward,
    ClarityReward,
    FormatReward,
)

__all__ = [
    "RewardCalculator",
    "RewardConfig",
    "RewardOutput",
    "SelfRatingExtractor", 
    "SelfRatingEvaluator",
    "SelfRating",
    "IntrinsicRewardCalculator",
    "ConsistencyReward",
    "CompletenessReward",
    "ClarityReward",
    "FormatReward",
]
