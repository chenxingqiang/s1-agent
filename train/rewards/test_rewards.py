"""
Unit Tests for Chain of Reward (CoR) Reward Calculation.

Run with:
    python -m pytest train/rewards/test_rewards.py -v
"""

import pytest
from .self_rating import SelfRatingExtractor, SelfRatingEvaluator
from .intrinsic import (
    ConsistencyReward,
    CompletenessReward,
    ClarityReward,
    FormatReward,
    IntrinsicRewardCalculator,
)
from .calculator import RewardCalculator, RewardConfig, parse_completion


class TestSelfRatingExtractor:
    """Tests for self-rating extraction."""
    
    def setup_method(self):
        self.extractor = SelfRatingExtractor()
    
    def test_extract_structured_format(self):
        """Test extraction of structured self-rating format."""
        thinking = """
        Step 1: Analyze the problem.
        [Self-Rating: Consistency=8/10, Completeness=9/10, Accuracy=7/10, Clarity=8/10]
        
        Step 2: Apply the formula.
        The answer is 42.
        """
        
        ratings = self.extractor.extract(thinking)
        
        assert "consistency" in ratings
        assert "completeness" in ratings
        assert ratings["consistency"].score == 8
        assert ratings["completeness"].score == 9
        assert ratings["accuracy"].score == 7
        assert ratings["clarity"].score == 8
    
    def test_extract_chinese_format(self):
        """Test extraction of Chinese self-rating format."""
        thinking = """
        第一步：分析问题。
        [评分: 一致性=8/10, 完整性=9/10]
        
        答案是42。
        """
        
        ratings = self.extractor.extract(thinking)
        
        assert "consistency" in ratings
        assert "completeness" in ratings
        assert ratings["consistency"].score == 8
    
    def test_extract_overall_quality(self):
        """Test extraction of overall quality rating."""
        thinking = """
        The reasoning is complete.
        [Overall Quality: 8.5/10]
        """
        
        ratings = self.extractor.extract(thinking)
        
        assert "overall" in ratings
        assert ratings["overall"].score == 8.5
    
    def test_no_ratings(self):
        """Test when no ratings present."""
        thinking = "This is just plain text without any ratings."
        
        ratings = self.extractor.extract(thinking)
        
        assert len(ratings) == 0
    
    def test_has_self_ratings(self):
        """Test has_self_ratings method."""
        with_ratings = "Some text [Self-Rating: Consistency=8/10]"
        without_ratings = "Some text without ratings"
        
        assert self.extractor.has_self_ratings(with_ratings) == True
        assert self.extractor.has_self_ratings(without_ratings) == False
    
    def test_get_average_rating(self):
        """Test average rating calculation."""
        thinking = "[Self-Rating: Consistency=8/10, Completeness=6/10]"
        
        avg = self.extractor.get_average_rating(thinking)
        
        assert avg == 0.7  # (0.8 + 0.6) / 2


class TestSelfRatingEvaluator:
    """Tests for self-rating quality evaluation."""
    
    def setup_method(self):
        self.evaluator = SelfRatingEvaluator(calibration_bonus=0.2)
    
    def test_compute_calibration_perfect(self):
        """Test calibration when self-rating matches actual."""
        cal = self.evaluator.compute_calibration(0.8, 0.8, apply_bonus=False)
        assert cal == 1.0
    
    def test_compute_calibration_with_bonus(self):
        """Test calibration bonus for high-high alignment."""
        cal = self.evaluator.compute_calibration(0.9, 0.9, apply_bonus=True)
        assert cal == 1.2  # 1.0 + 0.2 bonus
    
    def test_compute_calibration_mismatch(self):
        """Test calibration when ratings don't match."""
        cal = self.evaluator.compute_calibration(0.9, 0.5, apply_bonus=False)
        assert cal == 0.6  # 1.0 - 0.4
    
    def test_evaluate_self_rating_quality(self):
        """Test full self-rating quality evaluation."""
        from .self_rating import SelfRating
        
        self_ratings = {
            "consistency": SelfRating("consistency", 8, 0.8, ""),
            "completeness": SelfRating("completeness", 7, 0.7, ""),
        }
        
        actual_qualities = {
            "consistency": 0.8,
            "completeness": 0.6,
        }
        
        quality = self.evaluator.evaluate_self_rating_quality(
            self_ratings, actual_qualities, final_answer_correct=True
        )
        
        assert "overall_calibration" in quality
        assert "correctness_calibration" in quality
        assert "completeness_score" in quality
        
        # Consistency calibration should be 1.0 (perfect match)
        assert quality["per_dimension_calibration"]["consistency"] >= 1.0
        
        # Completeness calibration should be 0.9 (0.1 off)
        assert quality["per_dimension_calibration"]["completeness"] == 0.9


class TestIntrinsicRewards:
    """Tests for intrinsic reward functions."""
    
    def test_consistency_reward_high(self):
        """Test high consistency score for good reasoning."""
        reward = ConsistencyReward()
        
        thinking = """
        First, we identify the key variables.
        Given that x = 5, we can derive y.
        Therefore, y = 2x = 10.
        Thus, the answer is 10.
        """
        
        score = reward.compute(thinking)
        assert score > 0.8
    
    def test_consistency_reward_low(self):
        """Test low consistency score for contradictory reasoning."""
        reward = ConsistencyReward()
        
        thinking = """
        The answer is 5.
        Wait, that's wrong. Let me reconsider.
        Actually, I made a mistake. Let me start over.
        This contradicts my earlier statement.
        """
        
        score = reward.compute(thinking)
        assert score < 0.7
    
    def test_completeness_reward_high(self):
        """Test high completeness score for structured reasoning."""
        reward = CompletenessReward()
        
        thinking = """
        Step 1: Understand the problem
        The problem asks us to find x.
        
        Step 2: Apply the formula
        Using the given equation, x = 2y.
        
        Step 3: Substitute values
        Given y = 5, we get x = 10.
        
        Step 4: Verify
        Let's verify: 2 * 5 = 10. Correct!
        
        Therefore, x = 10.
        """
        
        score = reward.compute(thinking)
        assert score > 0.7
    
    def test_format_reward_with_self_rating(self):
        """Test format reward when self-ratings are present."""
        reward = FormatReward()
        
        thinking = """
        Step 1: Analyze
        [Self-Rating: Consistency=8/10, Completeness=9/10]
        
        The answer is 42.
        """
        
        score = reward.compute(thinking)
        assert score > 0.7
    
    def test_intrinsic_calculator_weighted(self):
        """Test weighted combination of dimensions."""
        calculator = IntrinsicRewardCalculator(weights={
            "consistency": 0.5,
            "completeness": 0.5,
            "clarity": 0.0,
            "format": 0.0,
        })
        
        thinking = "Step 1: Start. Therefore, the answer is 5."
        
        total, per_dim = calculator.compute_weighted_reward(thinking)
        
        assert 0 <= total <= 1
        assert "consistency" in per_dim
        assert "completeness" in per_dim


class TestRewardCalculator:
    """Tests for main reward calculator."""
    
    def setup_method(self):
        self.calculator = RewardCalculator()
    
    def test_external_reward_correct(self):
        """Test external reward for correct answer."""
        reward = self.calculator.calculate_external_reward("42", "42")
        assert reward == 1.0
    
    def test_external_reward_incorrect(self):
        """Test external reward for incorrect answer."""
        reward = self.calculator.calculate_external_reward("41", "42")
        assert reward == 0.0
    
    def test_external_reward_with_prefix(self):
        """Test external reward handles common prefixes."""
        reward = self.calculator.calculate_external_reward(
            "The answer is 42",
            "42"
        )
        assert reward == 1.0
    
    def test_intrinsic_reward(self):
        """Test intrinsic reward calculation."""
        thinking = """
        Step 1: Analyze the problem.
        [Self-Rating: Consistency=8/10]
        
        Therefore, x = 10.
        """
        
        intrinsic, dim_scores = self.calculator.calculate_intrinsic_reward(
            thinking,
            include_self_rating=True,
            final_answer_correct=True
        )
        
        assert 0 <= intrinsic <= 1
        assert "consistency" in dim_scores
    
    def test_total_reward(self):
        """Test total reward calculation."""
        thinking = """
        Step 1: Given x = 5, find y.
        y = 2x = 10.
        [Self-Rating: Consistency=9/10, Accuracy=9/10]
        
        Therefore, y = 10.
        """
        
        output = self.calculator.calculate_total_reward(
            thinking_chain=thinking,
            answer="10",
            ground_truth="10"
        )
        
        assert output.external_reward == 1.0
        assert output.intrinsic_reward > 0
        assert output.total_reward > 1.0  # ext + lambda * int


class TestParseCompletion:
    """Tests for completion parsing."""
    
    def test_parse_qwen_format(self):
        """Test parsing Qwen format completion."""
        completion = "<|im_start|>assistant\nStep 1: Think<|im_end|>\n42"
        
        thinking, answer = parse_completion(completion)
        
        assert "Step 1" in thinking
        assert answer == "42"
    
    def test_parse_answer_format(self):
        """Test parsing 'Answer:' format."""
        completion = "Step 1: Think about it.\n\nAnswer: 42"
        
        thinking, answer = parse_completion(completion)
        
        assert "Step 1" in thinking
        assert answer == "42"
    
    def test_parse_the_answer_is_format(self):
        """Test parsing 'The answer is' format."""
        completion = "We calculate step by step. The answer is 42"
        
        thinking, answer = parse_completion(completion)
        
        assert "calculate" in thinking
        assert answer == "42"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
