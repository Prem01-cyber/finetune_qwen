"""Configuration package for prompts and other settings."""

from src.config.prompts import (
    SOLVE_TASK_PREFIX,
    GENERATE_TASK_PREFIX,
    SOLVER_SYSTEM_PROMPT,
    GENERATOR_SYSTEM_PROMPT,
    format_solver_user_message,
    format_generator_user_message,
    create_solver_messages,
    create_generator_messages,
)

__all__ = [
    "SOLVE_TASK_PREFIX",
    "GENERATE_TASK_PREFIX",
    "SOLVER_SYSTEM_PROMPT",
    "GENERATOR_SYSTEM_PROMPT",
    "format_solver_user_message",
    "format_generator_user_message",
    "create_solver_messages",
    "create_generator_messages",
]
