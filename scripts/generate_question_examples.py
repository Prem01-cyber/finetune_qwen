#!/usr/bin/env python3
"""
Generate question-generation training examples using GPT-4 API.

This script creates synthetic training data for the question generation task
by using GPT-4/Claude to generate grade-school math problems that match
GSM8K style and difficulty.

Usage:
    python scripts/generate_question_examples.py \
        --output data/sft/question_generation.jsonl \
        --count 5000 \
        --api-key $OPENAI_API_KEY \
        --batch-size 10
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

# System prompt for question generation task
QUESTION_GENERATOR_SYSTEM_PROMPT = (
    "You are a math problem generator. "
    "Generate grade-school level math word problems that require 2-5 steps to solve. "
    "Problems should involve realistic scenarios and use simple arithmetic, fractions, "
    "percentages, or basic algebra. "
    "Output ONLY the problem statement, no solutions or steps."
)

# Topics and patterns extracted from GSM8K analysis
TOPIC_TEMPLATES = [
    # Money and shopping
    {
        "topic": "money and fractions",
        "template": "Create a word problem about someone shopping with an initial amount of money, "
                   "spending fractions of it on different items. Require {steps} steps to solve.",
        "example": "money, spending, fractions, shopping"
    },
    {
        "topic": "money and percentages",
        "template": "Generate a problem about discounts, sales, or price changes using percentages. "
                   "Include {steps} calculation steps.",
        "example": "discount, sale, percentage, price"
    },
    {
        "topic": "money and multiple purchases",
        "template": "Create a problem about buying multiple items with different prices and quantities. "
                   "Should require {steps} steps.",
        "example": "buying, items, total cost, quantity"
    },
    
    # Time and rate
    {
        "topic": "time and distance",
        "template": "Generate a problem about distance, speed, and time. "
                   "Should involve {steps} steps of calculation.",
        "example": "distance, speed, time, travel"
    },
    {
        "topic": "time and work rate",
        "template": "Create a problem about completing work at a certain rate over time. "
                   "Require {steps} steps to find the answer.",
        "example": "work, rate, time, completing tasks"
    },
    {
        "topic": "time and scheduling",
        "template": "Generate a problem about scheduling activities over time periods. "
                   "Should need {steps} steps.",
        "example": "schedule, hours, days, activities"
    },
    
    # Quantities and ratios
    {
        "topic": "ratios and sharing",
        "template": "Create a problem about dividing or sharing items in specific ratios. "
                   "Require {steps} calculation steps.",
        "example": "sharing, ratio, division, distribution"
    },
    {
        "topic": "collections and counting",
        "template": "Generate a problem about collecting items over time with additions and subtractions. "
                   "Should involve {steps} steps.",
        "example": "collecting, items, adding, removing"
    },
    {
        "topic": "area and perimeter",
        "template": "Create a problem about calculating areas or perimeters of simple shapes. "
                   "Require {steps} steps.",
        "example": "area, perimeter, rectangle, shape"
    },
    
    # Multi-step arithmetic
    {
        "topic": "multi-step arithmetic",
        "template": "Generate a problem requiring multiple arithmetic operations in sequence. "
                   "Should need exactly {steps} steps.",
        "example": "addition, subtraction, multiplication, division"
    },
    {
        "topic": "groups and distribution",
        "template": "Create a problem about distributing items among groups or people. "
                   "Require {steps} steps to solve.",
        "example": "groups, people, distribution, equal sharing"
    },
    {
        "topic": "age problems",
        "template": "Generate a problem about comparing ages of people now and in the future/past. "
                   "Should involve {steps} calculation steps.",
        "example": "age, years, older, younger"
    },
    
    # Food and cooking
    {
        "topic": "recipes and scaling",
        "template": "Create a problem about scaling recipe quantities up or down. "
                   "Require {steps} steps.",
        "example": "recipe, ingredients, servings, scaling"
    },
    {
        "topic": "food costs",
        "template": "Generate a problem about calculating costs of meals or ingredients. "
                   "Should need {steps} steps.",
        "example": "food, cost, meals, ingredients"
    },
    
    # School and learning
    {
        "topic": "test scores",
        "template": "Create a problem about calculating test scores, averages, or grades. "
                   "Require {steps} steps.",
        "example": "test, score, average, points"
    },
    {
        "topic": "class supplies",
        "template": "Generate a problem about distributing or counting school supplies. "
                   "Should involve {steps} steps.",
        "example": "students, supplies, pencils, notebooks"
    },
]


def load_existing_questions(gsm8k_path: Path) -> set[str]:
    """Load existing GSM8K questions to avoid duplicates."""
    questions = set()
    if gsm8k_path.exists():
        with gsm8k_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    # Extract question from user message
                    for msg in record.get("messages", []):
                        if msg.get("role") == "user":
                            # Remove the wrapper text
                            content = msg.get("content", "")
                            if "Problem:" in content:
                                q = content.split("Problem:")[-1].strip()
                                questions.add(q.lower())
    return questions


def generate_prompt(topic_info: dict[str, str], steps: int) -> str:
    """Generate a specific prompt for GPT-4 to create a question."""
    template = topic_info["template"].format(steps=steps)
    return f"### Task: Generate Question\n{template}\n\nGenerate ONE grade-school math word problem."


def call_openai_api(
    prompt: str,
    system_prompt: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
) -> str | None:
    """Call OpenAI API to generate a question."""
    try:
        import openai
    except ImportError:
        raise ImportError("Please install openai: pip install openai")
    
    client = openai.OpenAI(api_key=api_key)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,  # Higher temperature for diversity
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  API error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return None
    return None


def validate_question(question: str, existing_questions: set[str]) -> tuple[bool, str]:
    """
    Validate generated question for quality.
    
    Returns (is_valid, reason)
    """
    if not question or len(question.strip()) < 20:
        return False, "too_short"
    
    if len(question) > 1000:
        return False, "too_long"
    
    # Check if it's actually a question (should not contain solutions)
    solution_markers = ["step 1:", "step 2:", "final answer:", "solution:", "answer:"]
    if any(marker in question.lower() for marker in solution_markers):
        return False, "contains_solution"
    
    # Check for duplicate
    if question.lower().strip() in existing_questions:
        return False, "duplicate"
    
    # Check if it's a reasonable word problem (contains numbers)
    if not re.search(r'\d+', question):
        return False, "no_numbers"
    
    # Check if it asks for something (should end with ? or ask for calculation)
    ask_markers = ["?", "how many", "how much", "what is", "calculate", "find"]
    if not any(marker in question.lower() for marker in ask_markers):
        return False, "not_question"
    
    return True, "valid"


def create_question_record(
    question: str,
    prompt: str,
    topic: str,
    idx: int,
) -> dict[str, Any]:
    """Create a training record for question generation."""
    return {
        "id": f"qgen_{idx:05d}",
        "skill_id": "question_generation",
        "source": "gpt4_synthetic",
        "topic": topic,
        "messages": [
            {"role": "system", "content": QUESTION_GENERATOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": question},
        ],
        # Convenience for non-chat trainers
        "text": f"<|system|>\n{QUESTION_GENERATOR_SYSTEM_PROMPT}\n<|user|>\n{prompt}\n<|assistant|>\n{question}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate question-generation training examples using GPT-4."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path for question-generation examples",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5000,
        help="Number of question examples to generate",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use (gpt-4o-mini is cheaper, gpt-4o for best quality)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of questions to generate before saving checkpoint",
    )
    parser.add_argument(
        "--gsm8k-path",
        type=Path,
        default=Path("data/sft/gsm8k_sft.jsonl"),
        help="Path to existing GSM8K data to avoid duplicates",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume generation from existing output file",
    )
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Error: OpenAI API key required. "
            "Use --api-key or set OPENAI_API_KEY environment variable."
        )
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing questions to avoid duplicates
    print("Loading existing GSM8K questions...")
    existing_questions = load_existing_questions(args.gsm8k_path)
    print(f"Loaded {len(existing_questions)} existing questions")
    
    # Check if resuming
    start_idx = 0
    if args.resume and args.output.exists():
        with args.output.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    idx = int(record["id"].split("_")[-1])
                    start_idx = max(start_idx, idx + 1)
                    # Add to existing questions
                    for msg in record.get("messages", []):
                        if msg.get("role") == "assistant":
                            existing_questions.add(msg.get("content", "").lower())
        print(f"Resuming from index {start_idx} ({start_idx}/{args.count} complete)")
    
    # Open output file
    mode = "a" if args.resume else "w"
    out_f = args.output.open(mode, encoding="utf-8")
    
    # Statistics
    stats = {
        "generated": 0,
        "valid": 0,
        "too_short": 0,
        "too_long": 0,
        "contains_solution": 0,
        "duplicate": 0,
        "no_numbers": 0,
        "not_question": 0,
        "api_errors": 0,
    }
    
    print(f"\nGenerating {args.count - start_idx} question examples using {args.model}...")
    print(f"Output: {args.output}")
    print(f"Batch size: {args.batch_size}")
    print("-" * 60)
    
    try:
        idx = start_idx
        attempts = 0
        max_attempts = args.count * 3  # Allow up to 3x attempts for failures
        
        while idx < args.count and attempts < max_attempts:
            attempts += 1
            
            # Select random topic and step count
            topic_info = random.choice(TOPIC_TEMPLATES)
            steps = random.randint(2, 5)  # 2-5 steps as per plan
            
            # Generate prompt
            prompt = generate_prompt(topic_info, steps)
            
            # Call API
            question = call_openai_api(
                prompt=prompt,
                system_prompt=QUESTION_GENERATOR_SYSTEM_PROMPT,
                api_key=api_key,
                model=args.model,
            )
            
            stats["generated"] += 1
            
            if question is None:
                stats["api_errors"] += 1
                continue
            
            # Clean up question (remove any "Problem:" prefix if added)
            question = re.sub(r'^(Problem:\s*)', '', question, flags=re.IGNORECASE).strip()
            
            # Validate
            is_valid, reason = validate_question(question, existing_questions)
            
            if is_valid:
                # Create record
                record = create_question_record(
                    question=question,
                    prompt=prompt,
                    topic=topic_info["topic"],
                    idx=idx,
                )
                
                # Write to file
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                
                # Add to existing questions
                existing_questions.add(question.lower())
                
                stats["valid"] += 1
                idx += 1
                
                # Progress update
                if idx % args.batch_size == 0:
                    pct = (idx / args.count) * 100
                    print(f"Progress: {idx}/{args.count} ({pct:.1f}%) - "
                          f"Valid: {stats['valid']}, "
                          f"Rejected: {stats['generated'] - stats['valid']}, "
                          f"API errors: {stats['api_errors']}")
            else:
                stats[reason] += 1
                
                # Occasional feedback on rejections
                if attempts % 50 == 0:
                    print(f"  [Attempt {attempts}] Rejected: {reason} - "
                          f"Dups: {stats['duplicate']}, "
                          f"Solutions: {stats['contains_solution']}, "
                          f"Other: {stats['generated'] - stats['valid'] - stats['duplicate'] - stats['contains_solution']}")
        
        out_f.close()
        
        # Final statistics
        print("\n" + "=" * 60)
        print("Generation Complete!")
        print("=" * 60)
        print(f"Total API calls: {stats['generated']}")
        print(f"Valid questions: {stats['valid']}")
        print(f"API errors: {stats['api_errors']}")
        print(f"\nRejection reasons:")
        print(f"  Duplicates: {stats['duplicate']}")
        print(f"  Contains solution: {stats['contains_solution']}")
        print(f"  Too short: {stats['too_short']}")
        print(f"  Too long: {stats['too_long']}")
        print(f"  No numbers: {stats['no_numbers']}")
        print(f"  Not a question: {stats['not_question']}")
        print(f"\nOutput written to: {args.output}")
        
        if stats["valid"] < args.count:
            print(f"\nWarning: Only generated {stats['valid']}/{args.count} valid questions.")
            print(f"Consider running again with --resume to generate more.")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress saved.")
        out_f.close()
        print(f"Generated {stats['valid']} valid questions so far.")
        print(f"Resume with: --resume flag")
    
    except Exception as e:
        out_f.close()
        raise


if __name__ == "__main__":
    main()
