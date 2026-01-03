"""
Self-Rating Extraction and Evaluation for Endogenous Rewards.

Core innovation from CoR: Model generates self-ratings during thinking,
and we evaluate the quality of these self-ratings.

Based on DESIGN.md Section 3.2.1, 4.2.2 and THEORY.md Section 3.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SelfRating:
    """A single self-rating extracted from thinking chain."""
    dimension: str
    score: float  # 0-10 scale
    normalized: float  # 0-1 scale
    raw_text: str


class SelfRatingExtractor:
    """Extract self-ratings from model's thinking chain.
    
    Supports multiple formats:
    - [Self-Rating: Consistency=8/10, Completeness=9/10]
    - [评分: 逻辑一致性=8/10, 步骤完整性=9/10]
    - Step N: ... [Rating: X/10]
    
    Based on DESIGN.md Section 3.2.1.
    """
    
    # Dimension name mappings (English and Chinese)
    DIMENSION_ALIASES = {
        # English
        "consistency": "consistency",
        "completeness": "completeness", 
        "accuracy": "accuracy",
        "clarity": "clarity",
        "confidence": "confidence",
        "quality": "quality",
        # Chinese
        "逻辑一致性": "consistency",
        "一致性": "consistency",
        "完整性": "completeness",
        "步骤完整性": "completeness",
        "准确性": "accuracy",
        "事实准确性": "accuracy",
        "清晰度": "clarity",
        "清晰性": "clarity",
        "置信度": "confidence",
        "质量": "quality",
    }
    
    # Regex patterns for extracting self-ratings
    PATTERNS = [
        # [Self-Rating: Dim1=X/10, Dim2=Y/10]
        r'\[Self-Rating:\s*([^\]]+)\]',
        # [评分: Dim1=X/10, Dim2=Y/10]
        r'\[评分:\s*([^\]]+)\]',
        # [Rating: X/10] or [Score: X/10]
        r'\[(?:Rating|Score):\s*(\d+(?:\.\d+)?)/10\]',
        # [Overall Quality: X/10]
        r'\[Overall Quality:\s*(\d+(?:\.\d+)?)/10\]',
        # [总评: X/10]
        r'\[总评:\s*(\d+(?:\.\d+)?)/10\]',
        # Dimension: X/10 (standalone)
        r'(\w+(?:一致性|完整性|准确性|清晰度)?)\s*[:=]\s*(\d+(?:\.\d+)?)/10',
    ]
    
    def __init__(self, default_dimensions: Optional[List[str]] = None):
        """Initialize extractor.
        
        Args:
            default_dimensions: Default dimensions to look for.
                Defaults to ["consistency", "completeness", "accuracy", "clarity"].
        """
        self.default_dimensions = default_dimensions or [
            "consistency", "completeness", "accuracy", "clarity"
        ]
    
    def extract(self, thinking_chain: str) -> Dict[str, SelfRating]:
        """Extract all self-ratings from thinking chain.
        
        Args:
            thinking_chain: The model's thinking process text.
            
        Returns:
            Dict mapping dimension names to SelfRating objects.
        """
        ratings = {}
        
        # Try structured format first: [Self-Rating: Dim1=X/10, Dim2=Y/10]
        structured_match = re.search(
            r'\[Self-Rating:\s*([^\]]+)\]', 
            thinking_chain, 
            re.IGNORECASE
        )
        if structured_match:
            content = structured_match.group(1)
            ratings.update(self._parse_structured_rating(content, structured_match.group(0)))
        
        # Try Chinese format: [评分: ...]
        chinese_match = re.search(r'\[评分:\s*([^\]]+)\]', thinking_chain)
        if chinese_match:
            content = chinese_match.group(1)
            ratings.update(self._parse_structured_rating(content, chinese_match.group(0)))
        
        # Try overall quality format
        overall_match = re.search(
            r'\[(?:Overall Quality|总评):\s*(\d+(?:\.\d+)?)/10\]',
            thinking_chain,
            re.IGNORECASE
        )
        if overall_match:
            score = float(overall_match.group(1))
            ratings["overall"] = SelfRating(
                dimension="overall",
                score=score,
                normalized=score / 10.0,
                raw_text=overall_match.group(0)
            )
        
        # Try individual dimension patterns
        for pattern in [
            r'(?:Consistency|一致性)\s*[:=]\s*(\d+(?:\.\d+)?)/10',
            r'(?:Completeness|完整性)\s*[:=]\s*(\d+(?:\.\d+)?)/10',
            r'(?:Accuracy|准确性)\s*[:=]\s*(\d+(?:\.\d+)?)/10',
            r'(?:Clarity|清晰度)\s*[:=]\s*(\d+(?:\.\d+)?)/10',
        ]:
            match = re.search(pattern, thinking_chain, re.IGNORECASE)
            if match:
                # Infer dimension from pattern
                dim = self._infer_dimension_from_pattern(pattern)
                if dim and dim not in ratings:
                    score = float(match.group(1))
                    ratings[dim] = SelfRating(
                        dimension=dim,
                        score=score,
                        normalized=score / 10.0,
                        raw_text=match.group(0)
                    )
        
        return ratings
    
    def _parse_structured_rating(self, content: str, raw_text: str) -> Dict[str, SelfRating]:
        """Parse structured rating content like 'Dim1=X/10, Dim2=Y/10'."""
        ratings = {}
        
        # Split by comma or semicolon
        parts = re.split(r'[,;]\s*', content)
        
        for part in parts:
            # Match patterns like "Dimension=8/10" or "Dimension: 8/10"
            match = re.match(r'(\w+(?:\s+\w+)?)\s*[:=]\s*(\d+(?:\.\d+)?)/10', part.strip())
            if match:
                dim_name = match.group(1).strip().lower()
                score = float(match.group(2))
                
                # Normalize dimension name
                normalized_dim = self.DIMENSION_ALIASES.get(dim_name, dim_name)
                
                ratings[normalized_dim] = SelfRating(
                    dimension=normalized_dim,
                    score=score,
                    normalized=score / 10.0,
                    raw_text=raw_text
                )
        
        return ratings
    
    def _infer_dimension_from_pattern(self, pattern: str) -> Optional[str]:
        """Infer dimension name from regex pattern."""
        if "Consistency" in pattern or "一致性" in pattern:
            return "consistency"
        elif "Completeness" in pattern or "完整性" in pattern:
            return "completeness"
        elif "Accuracy" in pattern or "准确性" in pattern:
            return "accuracy"
        elif "Clarity" in pattern or "清晰度" in pattern:
            return "clarity"
        return None
    
    def has_self_ratings(self, thinking_chain: str) -> bool:
        """Check if thinking chain contains any self-ratings."""
        return len(self.extract(thinking_chain)) > 0
    
    def get_average_rating(self, thinking_chain: str) -> float:
        """Get average of all extracted self-ratings.
        
        Returns 0.5 (neutral) if no ratings found.
        """
        ratings = self.extract(thinking_chain)
        if not ratings:
            return 0.5
        
        scores = [r.normalized for r in ratings.values()]
        return sum(scores) / len(scores)


class SelfRatingEvaluator:
    """Evaluate quality of model's self-ratings.
    
    Core endogenous reward: We reward models for accurate self-assessment.
    
    Based on DESIGN.md Section 4.2.2 and THEORY.md Section 3.
    """
    
    def __init__(self, calibration_bonus: float = 0.2):
        """Initialize evaluator.
        
        Args:
            calibration_bonus: Bonus for high-high alignment (α in theory).
        """
        self.calibration_bonus = calibration_bonus
        self.extractor = SelfRatingExtractor()
    
    def compute_calibration(
        self,
        self_rating: float,
        actual_quality: float,
        apply_bonus: bool = True
    ) -> float:
        """Compute calibration score between self-rating and actual quality.
        
        From THEORY.md Definition 8:
        cal_d(u, v) = 1 - |u - v|
        
        With optional bonus for high-high alignment:
        cal_d^enhanced(u, v) = cal_d(u, v) + α * I[u > 0.8 and v > 0.8]
        
        Args:
            self_rating: Model's self-rating (0-1 scale).
            actual_quality: Actual quality score (0-1 scale).
            apply_bonus: Whether to apply high-high alignment bonus.
            
        Returns:
            Calibration score in [0, 1 + α] range.
        """
        # Base calibration
        cal = 1.0 - abs(self_rating - actual_quality)
        
        # Bonus for high-high alignment
        if apply_bonus and self_rating > 0.8 and actual_quality > 0.8:
            cal += self.calibration_bonus
        
        return cal
    
    def evaluate_self_rating_quality(
        self,
        self_ratings: Dict[str, SelfRating],
        actual_qualities: Dict[str, float],
        final_answer_correct: bool
    ) -> Dict[str, float]:
        """Evaluate quality of model's self-assessment.
        
        From THEORY.md Definition 9:
        r_self_rating_quality = (1/D) * Σ_d cal_d(self_rating_d/10, actual_quality_d)
        
        Args:
            self_ratings: Extracted self-ratings from thinking chain.
            actual_qualities: Actual quality scores per dimension (0-1 scale).
            final_answer_correct: Whether the final answer was correct.
            
        Returns:
            Dict with quality metrics:
            - per_dimension_calibration: Calibration per dimension
            - overall_calibration: Average calibration across dimensions
            - correctness_calibration: Alignment of confidence with correctness
            - completeness_score: How many expected dimensions were rated
        """
        result = {
            "per_dimension_calibration": {},
            "overall_calibration": 0.0,
            "correctness_calibration": 0.0,
            "completeness_score": 0.0,
        }
        
        if not self_ratings:
            return result
        
        # Per-dimension calibration
        calibration_scores = []
        for dim, rating in self_ratings.items():
            if dim in actual_qualities:
                cal = self.compute_calibration(
                    rating.normalized,
                    actual_qualities[dim]
                )
                result["per_dimension_calibration"][dim] = cal
                calibration_scores.append(cal)
        
        # Overall calibration (average)
        if calibration_scores:
            result["overall_calibration"] = sum(calibration_scores) / len(calibration_scores)
        
        # Correctness calibration: high self-ratings should correlate with correct answers
        avg_self_rating = sum(r.normalized for r in self_ratings.values()) / len(self_ratings)
        if final_answer_correct:
            # Reward high confidence when correct
            result["correctness_calibration"] = avg_self_rating
        else:
            # Reward low confidence when incorrect (model knows it's uncertain)
            result["correctness_calibration"] = 1.0 - avg_self_rating
        
        # Completeness: what fraction of expected dimensions were rated
        expected_dims = {"consistency", "completeness", "accuracy", "clarity"}
        rated_dims = set(self_ratings.keys()) & expected_dims
        result["completeness_score"] = len(rated_dims) / len(expected_dims)
        
        return result
    
    def compute_self_rating_reward(
        self,
        thinking_chain: str,
        actual_qualities: Dict[str, float],
        final_answer_correct: bool
    ) -> float:
        """Compute the self-rating quality reward.
        
        This is r_self_rating_quality from THEORY.md.
        
        Args:
            thinking_chain: Model's thinking process.
            actual_qualities: Actual quality scores per dimension.
            final_answer_correct: Whether answer was correct.
            
        Returns:
            Self-rating quality reward in [0, 1] range.
        """
        self_ratings = self.extractor.extract(thinking_chain)
        
        if not self_ratings:
            # No self-ratings found - return neutral score
            # This encourages model to generate self-ratings
            return 0.3
        
        quality = self.evaluate_self_rating_quality(
            self_ratings, actual_qualities, final_answer_correct
        )
        
        # Combine metrics into single reward
        # Weights can be tuned
        reward = (
            0.4 * quality["overall_calibration"] +
            0.3 * quality["correctness_calibration"] +
            0.3 * quality["completeness_score"]
        )
        
        return min(1.0, max(0.0, reward))
