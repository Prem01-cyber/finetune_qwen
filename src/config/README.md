# Centralized Prompt Configuration

All prompts for question generation and solution generation are centralized in `src/config/prompts.py`.

## Why Centralized?

- **Consistency**: Same prompts across SFT training, GRPO, PPO, and inference
- **Maintainability**: Single source of truth for all prompt text
- **Flexibility**: Easy to tune prompts without hunting through multiple files

## Usage

### Question Generation

```python
from src.config.prompts import create_generator_messages

instruction = "Generate a problem about fractions in a shopping context"
messages = create_generator_messages(instruction)
# Returns:
# [
#   {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
#   {"role": "user", "content": "### Task: Generate Question\n{instruction}"}
# ]
```

### Solution Generation

```python
from src.config.prompts import create_solver_messages

question = "If John has 5 apples and gives 2 away, how many does he have?"
messages = create_solver_messages(question)
# Returns:
# [
#   {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
#   {"role": "user", "content": "### Task: Solve Problem\nProblem: {question}\nSolution:"}
# ]
```

## Files Using Centralized Prompts

- `scripts/run_grpo_training.py` - GRPO question generation
- `scripts/dual_task_sft_pipeline.py` - SFT training
- `scripts/create_dual_task_dataset.py` - Dataset creation
- `src/rl/math_environment.py` - PPO environment
- `src/rl/triple_verifier.py` - Consensus verification

## Prompt Design Principles

### Question Generation
- **No explicit step constraints**: Let the model decide complexity naturally
- Focus on **realistic scenarios** and **simple operations** (grade-school level)
- Output **only the problem statement**, no solutions

### Solution Generation
- **Step-by-step format**: Each step on its own line starting with "Step N:"
- **Final Answer format**: Line starting with "Final Answer:"
- **Python/SymPy syntax**: All math expressions verifiable programmatically
