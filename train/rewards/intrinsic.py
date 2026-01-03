"""
Intrinsic Reward Functions for Chain of Reward.

Multi-dimensional quality assessment of thinking chains.
Based on DESIGN.md Section 4.2 and THEORY.md Section 2.

Dimensions:
- Consistency: Logical coherence of reasoning
- Completeness: Step comprehensiveness  
- Accuracy: Factual correctness
- Clarity: Reasoning clarity
- Format: Structural correctness
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod


class BaseIntrinsicReward(ABC):
    """Base class for intrinsic reward functions."""
    
    @abstractmethod
    def compute(self, thinking_chain: str, **kwargs) -> float:
        """Compute reward for thinking chain.
        
        Args:
            thinking_chain: The model's thinking process text.
            **kwargs: Additional context (e.g., question, answer).
            
        Returns:
            Reward score in [0, 1] range.
        """
        pass
    
    @property
    @abstractmethod
    def dimension_name(self) -> str:
        """Return the dimension name."""
        pass


class ConsistencyReward(BaseIntrinsicReward):
    """Evaluate logical consistency of reasoning.
    
    Checks:
    - No self-contradictions
    - Conclusions follow from premises
    - Intermediate results align with final answer
    """
    
    @property
    def dimension_name(self) -> str:
        return "consistency"
    
    def compute(self, thinking_chain: str, **kwargs) -> float:
        """Compute consistency reward."""
        score = 1.0
        
        # Check for contradiction indicators
        contradiction_patterns = [
            r'(?:wait|actually|no)[,\s]+(?:that\'s|this is)\s+(?:wrong|incorrect)',
            r'I made (?:a|an) (?:mistake|error)',
            r'(?:Let me|I\'ll|I should)\s+(?:reconsider|recalculate|redo|start over)',
            r'(?:This|That) contradicts',
            r'(?:This|That) doesn\'t make sense',
        ]
        
        for pattern in contradiction_patterns:
            matches = re.findall(pattern, thinking_chain, re.IGNORECASE)
            # Penalize for contradictions, but not too harshly
            # (some self-correction is good, excessive is not)
            score -= 0.1 * min(len(matches), 3)
        
        # Check for logical flow markers (positive)
        flow_patterns = [
            r'(?:Therefore|Thus|Hence|So|Consequently)',
            r'(?:This means|This implies|It follows)',
            r'(?:Given that|Since|Because)',
            r'(?:From|Based on) (?:this|the above)',
        ]
        
        flow_count = 0
        for pattern in flow_patterns:
            flow_count += len(re.findall(pattern, thinking_chain, re.IGNORECASE))
        
        # Reward logical flow (up to a point)
        score += 0.05 * min(flow_count, 5)
        
        return max(0.0, min(1.0, score))


class CompletenessReward(BaseIntrinsicReward):
    """Evaluate step comprehensiveness.
    
    Checks:
    - Presence of clear reasoning steps
    - Problem decomposition
    - Verification/checking steps
    """
    
    @property
    def dimension_name(self) -> str:
        return "completeness"
    
    def compute(self, thinking_chain: str, **kwargs) -> float:
        """Compute completeness reward."""
        score = 0.0
        
        # Check for numbered/bulleted steps
        step_patterns = [
            r'(?:Step\s+\d+|First|Second|Third|Finally)',
            r'^\s*\d+[\.\)]\s+',  # "1. " or "1) "
            r'^\s*[-•]\s+',  # Bullets
        ]
        
        step_count = 0
        for pattern in step_patterns:
            step_count += len(re.findall(pattern, thinking_chain, re.MULTILINE))
        
        # Reward having steps (3-10 is ideal)
        if step_count >= 3:
            score += 0.3
        if step_count >= 5:
            score += 0.2
        
        # Check for problem analysis
        analysis_patterns = [
            r'(?:Let\'s|I\'ll|We need to)\s+(?:analyze|understand|break down)',
            r'(?:The problem|This question)\s+(?:asks|requires|involves)',
            r'(?:Given|Known|We have)',
        ]
        
        for pattern in analysis_patterns:
            if re.search(pattern, thinking_chain, re.IGNORECASE):
                score += 0.1
        
        # Check for verification
        verify_patterns = [
            r'(?:Let\'s|I\'ll|Let me)\s+(?:verify|check|confirm|validate)',
            r'(?:To verify|Checking|Double-check)',
            r'(?:This|The answer) (?:seems|looks|appears) (?:correct|right)',
        ]
        
        for pattern in verify_patterns:
            if re.search(pattern, thinking_chain, re.IGNORECASE):
                score += 0.15
        
        # Check for conclusion
        conclusion_patterns = [
            r'(?:Therefore|Thus|Hence|So|In conclusion)',
            r'(?:The answer is|Final answer)',
            r'(?:We conclude|I conclude)',
        ]
        
        for pattern in conclusion_patterns:
            if re.search(pattern, thinking_chain, re.IGNORECASE):
                score += 0.15
        
        return max(0.0, min(1.0, score))


class ClarityReward(BaseIntrinsicReward):
    """Evaluate reasoning clarity.
    
    Checks:
    - Clear structure and organization
    - Explicit variable/term definitions
    - Readable formatting
    """
    
    @property
    def dimension_name(self) -> str:
        return "clarity"
    
    def compute(self, thinking_chain: str, **kwargs) -> float:
        """Compute clarity reward."""
        score = 0.5  # Start neutral
        
        # Check for definitions/explanations
        definition_patterns = [
            r'(?:Let|Define|Denote)\s+\w+\s*(?:=|as|to be)',
            r'(?:where|here)\s+\w+\s+(?:is|represents|denotes)',
        ]
        
        for pattern in definition_patterns:
            if re.search(pattern, thinking_chain, re.IGNORECASE):
                score += 0.1
        
        # Check for clear section markers
        section_patterns = [
            r'(?:Step\s+\d+:|Part\s+\d+:|First,|Second,|Finally,)',
            r'(?:Analysis:|Solution:|Approach:|Method:)',
        ]
        
        section_count = 0
        for pattern in section_patterns:
            section_count += len(re.findall(pattern, thinking_chain, re.IGNORECASE))
        
        score += 0.05 * min(section_count, 5)
        
        # Penalize for very long unbroken text (wall of text)
        paragraphs = thinking_chain.split('\n\n')
        for para in paragraphs:
            if len(para) > 1000:  # Very long paragraph
                score -= 0.1
        
        # Check average sentence length (too long = unclear)
        sentences = re.split(r'[.!?]+', thinking_chain)
        if sentences:
            avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_len > 50:  # Very long sentences
                score -= 0.1
            elif avg_len < 30:  # Reasonable length
                score += 0.1
        
        return max(0.0, min(1.0, score))


class FormatReward(BaseIntrinsicReward):
    """Evaluate structural correctness of output format.
    
    Checks:
    - Proper thinking/answer separation
    - Self-rating format presence
    - No malformed markers
    """
    
    @property
    def dimension_name(self) -> str:
        return "format"
    
    def compute(self, thinking_chain: str, **kwargs) -> float:
        """Compute format reward."""
        score = 0.5  # Start neutral
        
        # Check for self-rating format (important for CoR)
        self_rating_patterns = [
            r'\[Self-Rating:',
            r'\[评分:',
            r'\[Rating:.*?/10\]',
        ]
        
        has_self_rating = False
        for pattern in self_rating_patterns:
            if re.search(pattern, thinking_chain, re.IGNORECASE):
                has_self_rating = True
                score += 0.3
                break
        
        # Check for proper structure markers
        if '<|im_start|>' in kwargs.get('full_text', ''):
            # Qwen format
            if '<|im_end|>' in kwargs.get('full_text', ''):
                score += 0.1
        
        # Check for markdown formatting (headers, lists)
        markdown_patterns = [
            r'^#+\s+',  # Headers
            r'^\s*[-*]\s+',  # Lists
            r'\*\*[^*]+\*\*',  # Bold
            r'`[^`]+`',  # Code
        ]
        
        markdown_count = 0
        for pattern in markdown_patterns:
            markdown_count += len(re.findall(pattern, thinking_chain, re.MULTILINE))
        
        if markdown_count > 0:
            score += 0.1
        
        # Penalize for incomplete/malformed content
        if thinking_chain.strip().endswith('...'):
            score -= 0.2
        
        if thinking_chain.count('(') != thinking_chain.count(')'):
            score -= 0.1
        
        if thinking_chain.count('[') != thinking_chain.count(']'):
            score -= 0.1
        
        return max(0.0, min(1.0, score))


class IntrinsicRewardCalculator:
    """Calculate multi-dimensional intrinsic rewards.
    
    Combines multiple reward dimensions with configurable weights.
    Based on THEORY.md Definition 5:
    R_int(c) = Σ_d w_d * r_d(y_think)
    """
    
    DEFAULT_WEIGHTS = {
        "consistency": 0.25,
        "completeness": 0.25,
        "clarity": 0.25,
        "format": 0.25,
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """Initialize calculator.
        
        Args:
            weights: Weights for each dimension. Defaults to equal weights.
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        
        # Initialize reward functions
        self.reward_functions: Dict[str, BaseIntrinsicReward] = {
            "consistency": ConsistencyReward(),
            "completeness": CompletenessReward(),
            "clarity": ClarityReward(),
            "format": FormatReward(),
        }
        
        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
    
    def compute_all_dimensions(
        self, 
        thinking_chain: str,
        **kwargs
    ) -> Dict[str, float]:
        """Compute rewards for all dimensions.
        
        Args:
            thinking_chain: The model's thinking process.
            **kwargs: Additional context.
            
        Returns:
            Dict mapping dimension names to reward scores.
        """
        results = {}
        
        for dim, func in self.reward_functions.items():
            results[dim] = func.compute(thinking_chain, **kwargs)
        
        return results
    
    def compute_weighted_reward(
        self,
        thinking_chain: str,
        **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """Compute weighted intrinsic reward.
        
        Args:
            thinking_chain: The model's thinking process.
            **kwargs: Additional context.
            
        Returns:
            Tuple of (weighted_total, per_dimension_scores).
        """
        per_dim = self.compute_all_dimensions(thinking_chain, **kwargs)
        
        weighted_total = sum(
            self.weights.get(dim, 0.0) * score
            for dim, score in per_dim.items()
        )
        
        return weighted_total, per_dim
    
    def get_actual_qualities(self, thinking_chain: str, **kwargs) -> Dict[str, float]:
        """Get actual quality scores for self-rating calibration.
        
        These are used to evaluate how well the model's self-ratings
        match the actual quality metrics.
        """
        return self.compute_all_dimensions(thinking_chain, **kwargs)
