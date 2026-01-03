"""
Add Self-Ratings to Thinking Chains for CoR Training.

This script enhances existing thinking chains with multi-dimensional
self-ratings, preparing data for CoR training cold-start.

Based on DESIGN.md Section 3.2.3 and 7.

Methods:
1. Rule-based: Add self-ratings based on heuristic analysis
2. LLM-based: Use GPT-4o/Claude to generate self-ratings

Usage:
    python data/add_self_ratings.py --input_path qfq/geminiall \
                                    --output_path qfq/geminiall_rated \
                                    --method llm
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import json
import logging
import argparse
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

from datasets import load_dataset, Dataset
from data.utils.io_utils import jload, jdump, question_hash
from data.utils.inference_utils import apiqa

logging.basicConfig(level=logging.INFO)


# Self-rating prompt template
SELF_RATING_SYSTEM_PROMPT = """You are an expert at evaluating reasoning quality. Given a thinking chain, analyze it and add self-ratings at appropriate points.

For each major step or section in the thinking, add a rating in this format:
[Self-Rating: Consistency=X/10, Completeness=X/10, Accuracy=X/10, Clarity=X/10]

Rating dimensions:
- Consistency: How logically coherent is this step? Does it follow from previous steps?
- Completeness: How comprehensive is the reasoning? Are all necessary sub-steps included?
- Accuracy: Are the facts, formulas, and calculations correct?
- Clarity: How clear and understandable is the explanation?

At the end, add an overall rating:
[Overall Quality: X/10]

Guidelines:
- Be honest and calibrated in your ratings
- High ratings (8-10) should only be given for genuinely excellent reasoning
- Identify specific issues when giving lower ratings
- Add ratings after each major reasoning step, not every sentence"""

SELF_RATING_USER_PROMPT = """Please analyze this thinking chain and add self-ratings:

## Thinking Chain
{thinking}

## Final Answer
{answer}

Return the thinking chain with self-ratings inserted at appropriate points. Keep the original content intact and only add the [Self-Rating: ...] markers."""


def add_ratings_rule_based(thinking: str) -> str:
    """Add self-ratings using rule-based heuristics.
    
    This is faster but less accurate than LLM-based.
    Used for initial cold-start when LLM calls are expensive.
    """
    # Import intrinsic reward calculator
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'train'))
    from rewards.intrinsic import IntrinsicRewardCalculator
    
    calculator = IntrinsicRewardCalculator()
    
    # Split thinking into sections
    sections = split_into_sections(thinking)
    
    rated_sections = []
    for section in sections:
        # Compute quality scores for this section
        scores = calculator.compute_all_dimensions(section)
        
        # Format as self-rating
        rating = format_self_rating(scores)
        
        # Add rating at end of section
        rated_sections.append(section.strip() + "\n" + rating)
    
    # Combine sections
    rated_thinking = "\n\n".join(rated_sections)
    
    # Add overall rating
    overall_scores = calculator.compute_all_dimensions(thinking)
    overall = sum(overall_scores.values()) / len(overall_scores) * 10
    rated_thinking += f"\n\n[Overall Quality: {overall:.1f}/10]"
    
    return rated_thinking


def split_into_sections(thinking: str) -> List[str]:
    """Split thinking chain into logical sections."""
    
    # Try to split by numbered steps
    step_pattern = r'(?:Step\s+\d+[:\.]|^\d+[\.\)]|\n(?:First|Second|Third|Finally)[,:])'
    
    parts = re.split(step_pattern, thinking, flags=re.MULTILINE | re.IGNORECASE)
    
    if len(parts) > 1:
        return [p.strip() for p in parts if p.strip()]
    
    # Fallback: split by double newlines (paragraphs)
    paragraphs = thinking.split('\n\n')
    
    # Group small paragraphs
    sections = []
    current = ""
    for para in paragraphs:
        current += para + "\n\n"
        if len(current) > 500:  # Target section size
            sections.append(current.strip())
            current = ""
    
    if current.strip():
        sections.append(current.strip())
    
    return sections if sections else [thinking]


def format_self_rating(scores: Dict[str, float]) -> str:
    """Format quality scores as self-rating string."""
    
    # Map dimension names and convert to /10 scale
    dimension_map = {
        "consistency": "Consistency",
        "completeness": "Completeness",
        "accuracy": "Accuracy",
        "clarity": "Clarity",
    }
    
    parts = []
    for dim, label in dimension_map.items():
        if dim in scores:
            score = scores[dim] * 10  # Convert 0-1 to 0-10
            parts.append(f"{label}={score:.0f}/10")
    
    return f"[Self-Rating: {', '.join(parts)}]"


def add_ratings_llm(
    thinking: str,
    answer: str,
    model: str = "claude-3-5-sonnet-20241022"
) -> str:
    """Add self-ratings using LLM.
    
    More accurate but slower and costs API calls.
    """
    user_prompt = SELF_RATING_USER_PROMPT.format(
        thinking=thinking,
        answer=answer
    )
    
    try:
        completion, _ = apiqa(
            user_prompt,
            model,
            SELF_RATING_SYSTEM_PROMPT,
            json_format=False
        )
        
        # Validate that response has self-ratings
        if "[Self-Rating:" in completion or "[评分:" in completion:
            return completion
        else:
            logging.warning("LLM response missing self-ratings, using rule-based fallback")
            return add_ratings_rule_based(thinking)
            
    except Exception as e:
        logging.error(f"LLM call failed: {e}, using rule-based fallback")
        return add_ratings_rule_based(thinking)


def process_example(
    example: Dict,
    method: str = "rule",
    model: str = "claude-3-5-sonnet-20241022"
) -> Dict:
    """Process a single example to add self-ratings.
    
    Args:
        example: Dataset example with 'thinking' and 'response' fields.
        method: 'rule' for rule-based, 'llm' for LLM-based.
        model: LLM model to use if method='llm'.
        
    Returns:
        Updated example with 'thinking_rated' field.
    """
    thinking = example.get('thinking', '') or example.get('thinking_trajectories', [''])[0]
    answer = example.get('response', '') or example.get('attempt', '')
    
    if not thinking:
        logging.warning(f"Empty thinking for example, skipping")
        return example
    
    if method == "llm":
        rated_thinking = add_ratings_llm(thinking, answer, model)
    else:
        rated_thinking = add_ratings_rule_based(thinking)
    
    example['thinking_rated'] = rated_thinking
    example['has_self_ratings'] = True
    
    return example


def format_for_training(example: Dict, tokenizer_name: str = "Qwen/Qwen2.5-32B-Instruct") -> Dict:
    """Format example for SFT training with self-ratings.
    
    Creates 'text' field in Qwen format with rated thinking.
    """
    question = example.get('question', '')
    thinking = example.get('thinking_rated', example.get('thinking', ''))
    answer = example.get('response', '') or example.get('attempt', '')
    
    # Qwen format
    text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{thinking}<|im_end|>\n{answer}"
    
    example['text'] = text
    
    return example


def main():
    parser = argparse.ArgumentParser(description="Add self-ratings to thinking chains")
    parser.add_argument("--input_path", type=str, required=True,
                       help="Input dataset path (HuggingFace)")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output dataset path")
    parser.add_argument("--method", type=str, choices=["rule", "llm"], default="rule",
                       help="Method for adding self-ratings")
    parser.add_argument("--model", type=str, default="claude-3-5-sonnet-20241022",
                       help="LLM model for llm method")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Max samples to process (for testing)")
    parser.add_argument("--num_workers", type=int, default=1,
                       help="Number of parallel workers")
    
    args = parser.parse_args()
    
    logging.info(f"Loading dataset from {args.input_path}")
    dataset = load_dataset(args.input_path)['train']
    
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    logging.info(f"Processing {len(dataset)} examples with method={args.method}")
    
    # Process examples
    process_fn = partial(process_example, method=args.method, model=args.model)
    
    if args.num_workers > 1 and args.method == "rule":
        # Parallel processing for rule-based (LLM has rate limits)
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            results = list(tqdm(
                executor.map(process_fn, dataset),
                total=len(dataset)
            ))
    else:
        results = [process_fn(ex) for ex in tqdm(dataset)]
    
    # Format for training
    logging.info("Formatting for training...")
    results = [format_for_training(ex) for ex in results]
    
    # Create dataset
    output_dataset = Dataset.from_list(results)
    
    # Save
    if args.output_path.startswith("qfq/") or "/" in args.output_path:
        logging.info(f"Pushing to HuggingFace: {args.output_path}")
        output_dataset.push_to_hub(args.output_path)
    else:
        logging.info(f"Saving locally to {args.output_path}")
        output_dataset.save_to_disk(args.output_path)
    
    logging.info("Done!")
    
    # Print sample
    logging.info("\nSample output:")
    sample = results[0]
    if 'thinking_rated' in sample:
        print(sample['thinking_rated'][:1000] + "...")


if __name__ == "__main__":
    main()
