"""
Centralized prompt configuration for math problem generation and solving.

All prompts used across SFT training, GRPO training, PPO training, and inference
are defined here to ensure consistency.
"""

# Task prefixes used in dual-task training
SOLVE_TASK_PREFIX = "### Task: Solve Problem\n"
GENERATE_TASK_PREFIX = "### Task: Generate Question\n"


# System prompts for solution generation
SOLVER_SYSTEM_PROMPT = (
    "You are a step-by-step math solver. "
    "Solve the given problem one step at a time. "
    "Each step must be on its own line, starting with 'Step N:'. "
    "End with a line starting with 'Final Answer:'. "
    "Write every mathematical expression in Python/SymPy syntax "
    "so it can be verified programmatically."
)


# System prompts for question generation
GENERATOR_SYSTEM_PROMPT = (
    "You are a math problem generator. "
    "Generate grade-school level math word problems. "
    "Problems should involve realistic scenarios and use simple arithmetic, fractions, "
    "percentages, or basic algebra. "
    "Output ONLY the problem statement, no solutions or steps."
)


def format_solver_user_message(question: str) -> str:
    """Format user message for solution generation."""
    return f"{SOLVE_TASK_PREFIX}Problem: {question}\nSolution:"


def format_generator_user_message(instruction: str) -> str:
    """Format user message for question generation."""
    return f"{GENERATE_TASK_PREFIX}{instruction}"


def create_solver_messages(question: str) -> list[dict[str, str]]:
    """Create chat messages for solution generation."""
    return [
        {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
        {"role": "user", "content": format_solver_user_message(question)},
    ]


def create_generator_messages(instruction: str) -> list[dict[str, str]]:
    """Create chat messages for question generation."""
    return [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": format_generator_user_message(instruction)},
    ]
