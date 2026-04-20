# Dual-Task Math Model Training

Complete implementation of dual-task model training with RL self-improvement for math problem generation and solving.

## Overview

This implementation allows you to train a model that can:
1. **Generate math questions** when prompted with topic/constraint instructions
2. **Solve math problems** step-by-step with SymPy-verifiable arithmetic

The training happens in two phases:
- **Phase 1**: Supervised Fine-Tuning (SFT) on dual tasks
- **Phase 2**: Reinforcement Learning (RL) self-improvement loop

## Directory Structure

```
Finetune_qwen/
├── data/
│   ├── sft/
│   │   ├── gsm8k_sft.jsonl              # Existing solution data
│   │   ├── question_generation.jsonl    # Generated question examples
│   │   ├── dual_task_train.jsonl        # Mixed training data
│   │   └── dual_task_val.jsonl          # Validation split
│   ├── rl/
│   │   └── self_play_iteration_*.jsonl  # Self-play trajectories
│   └── eval/
│       └── question_gen_prompts.jsonl   # Evaluation prompts
├── scripts/
│   ├── generate_question_examples.py    # GPT-4 question generation
│   ├── create_dual_task_dataset.py      # Dataset mixing
│   ├── dual_task_sft_pipeline.py        # Dual-task training
│   ├── eval_dual_task.py                # Dual evaluation
│   ├── self_play_generate.py            # Self-play generation
│   ├── rl_self_improve.py               # DPO training
│   ├── run_self_improve_iterations.py   # Multi-iteration loop
│   └── eval_self_improve_results.py     # Results analysis
├── src/
│   ├── sft/
│   │   ├── solution_format.py           # Format validation
│   │   ├── step_verify_sympy.py         # Arithmetic verification
│   │   └── sympy_normalize.py           # Expression normalization
│   └── rl/
│       └── reward_calculator.py         # Reward function
└── checkpoints/
    ├── dual_task_v1/                    # Phase 1 output
    └── rl_iterations/                   # Phase 2 outputs
        └── iteration_001/
```

## Phase 1: Dual-Task Supervised Fine-Tuning

### Step 1: Generate Question-Generation Training Data

Use GPT-4 to create synthetic question-generation examples:

```bash
python scripts/generate_question_examples.py \
    --output data/sft/question_generation.jsonl \
    --count 5000 \
    --api-key $OPENAI_API_KEY \
    --model gpt-4o-mini \
    --batch-size 10
```

**Options:**
- `--model`: Use `gpt-4o-mini` (cheaper) or `gpt-4o` (better quality)
- `--resume`: Resume from existing file if interrupted
- `--gsm8k-path`: Path to GSM8K data for duplicate detection

**Expected output**: 5,000 question-generation examples (~$10-20 for GPT-4o-mini)

### Step 2: Create Dual-Task Training Dataset

Mix solution and question-generation data with task prefixes:

```bash
python scripts/create_dual_task_dataset.py \
    --solution-data data/sft/gsm8k_sft.jsonl \
    --question-data data/sft/question_generation.jsonl \
    --output-train data/sft/dual_task_train.jsonl \
    --output-val data/sft/dual_task_val.jsonl \
    --mix-ratio 0.8 \
    --val-split 0.1
```

**Expected output**:
- Training: ~7,900 examples (80% solutions, 20% questions)
- Validation: ~880 examples

### Step 3: Train Dual-Task Model

Train model on mixed dataset:

```bash
python scripts/dual_task_sft_pipeline.py train \
    --data data/sft/dual_task_train.jsonl \
    --output-dir checkpoints/dual_task_v1 \
    --epochs 2 \
    --model Qwen/Qwen2.5-Math-7B-Instruct
```

**Training time**: 6-12 hours on A100 (2 epochs, ~8.75k examples)

**Hyperparameters** (defaults are good starting points):
- `--epochs 2`: Increased from 1.0 for dual-task complexity
- `--learning-rate 2e-4`: Standard for LoRA
- `--batch-size 1 --grad-accum 8`: Effective batch size of 8
- `--lora-rank 16 --lora-alpha 32`: LoRA configuration

### Step 4: Evaluate Dual-Task Model

Test both capabilities:

```bash
python scripts/eval_dual_task.py \
    --adapter checkpoints/dual_task_v1 \
    --question-prompts data/eval/question_gen_prompts.jsonl \
    --solution-data data/sft/gsm8k_sft.jsonl \
    --max-question-samples 100 \
    --max-solution-samples 100 \
    --output-json reports/dual_task_eval.json
```

**Success criteria**:
- Question generation: 80%+ valid questions
- Solution accuracy: ≥95% of single-task baseline
- Format compliance: ≥90%

### Step 5: Test Inference

**Question Generation**:
```bash
python scripts/dual_task_sft_pipeline.py infer \
    --adapter checkpoints/dual_task_v1 \
    --task generate \
    --prompt "Create a word problem about fractions and money requiring 3 steps."
```

**Solution Generation**:
```bash
python scripts/dual_task_sft_pipeline.py infer \
    --adapter checkpoints/dual_task_v1 \
    --task solve \
    --problem "Janet has 16 eggs. She eats 3. How many are left?"
```

## Phase 2: RL Self-Improvement Loop

Once Phase 1 produces a working dual-task model, implement self-improvement.

### Option A: Manual Single Iteration

**Step 1: Self-play generation**
```bash
python scripts/self_play_generate.py \
    --adapter checkpoints/dual_task_v1 \
    --output data/rl/self_play_iteration_001.jsonl \
    --num-samples 1000 \
    --reference-questions data/sft/gsm8k_sft.jsonl
```

**Step 2: DPO training**
```bash
python scripts/rl_self_improve.py \
    --base-adapter checkpoints/dual_task_v1 \
    --trajectories data/rl/self_play_iteration_001.jsonl \
    --output checkpoints/iteration_001 \
    --num-pairs 500
```

### Option B: Automated Multi-Iteration Loop (Recommended)

Run multiple iterations with automatic convergence monitoring:

```bash
python scripts/run_self_improve_iterations.py \
    --initial-adapter checkpoints/dual_task_v1 \
    --output-dir checkpoints/rl_iterations \
    --max-iterations 5 \
    --samples-per-iteration 1000 \
    --pairs-per-iteration 500 \
    --eval-data data/sft/gsm8k_sft.jsonl \
    --eval-samples 100 \
    --reference-questions data/sft/gsm8k_sft.jsonl
```

**Hyperparameters**:
- `--temperature 0.9`: Higher for diversity in self-play
- `--learning-rate 5e-7`: Lower than SFT for stability
- `--beta 0.1`: KL penalty strength for DPO
- `--patience 3`: Stop if no improvement for 3 iterations

**Training time per iteration**: 4-8 hours (1k self-play + DPO)

### Analyze Results

Generate comprehensive analysis report:

```bash
python scripts/eval_self_improve_results.py \
    --training-dir checkpoints/rl_iterations \
    --output reports/self_improve_analysis.md \
    --json-output reports/self_improve_analysis.json
```

## Reward Function

The reward calculator (`src/rl/reward_calculator.py`) scores trajectories:

### Question Quality (0-1)
- **Solvability (0.4)**: Can the model solve the generated question?
  - 1.0 if solution verifies correctly
  - 0.5 if format OK but arithmetic fails
  - 0.0 if unparseable
- **Novelty (0.3)**: Different from GSM8K training set?
  - Measured via n-gram Jaccard similarity
  - 1.0 if very novel, 0.0 if near-duplicate
- **Difficulty (0.3)**: Appropriate challenge level
  - Target: 2-5 steps = 1.0
  - Linear decay outside range

### Solution Quality (0-1)
- **Correctness (0.6)**: SymPy verification
  - `steps_verified_ok / steps_total`
- **Format (0.2)**: Proper step/answer structure
  - Binary: passes validation
- **Efficiency (0.2)**: Reasonable step count
  - Penalty for >8 steps

### Combined Reward
```
reward = 0.5 * question_score + 0.5 * solution_score
```

## Task Prefixes

The model uses clear prefixes to switch between tasks:

### Question Generation
```
System: "You are a math problem generator..."
User: "### Task: Generate Question\nCreate a problem about fractions..."
Assistant: "[Generated question]"
```

### Solution Generation
```
System: "You are a step-by-step math solver..."
User: "### Task: Solve Problem\n[Question]"
Assistant: "Step 1: ...\nStep 2: ...\nFinal Answer: X"
```

## Expected Results

### Phase 1 Success
- Model generates coherent questions
- GSM8K accuracy: ≥95% of baseline
- 80%+ generated questions are solvable

### Phase 2 Success
- Question quality improves across iterations
- GSM8K accuracy: +2-5% absolute improvement
- More diverse and challenging problems
- Solution verification pass rate increases

## Troubleshooting

### Phase 1 Issues

**Problem**: Task prefix collision (model confuses tasks)
- **Solution**: Ensure prefixes are very distinct, check training data

**Problem**: Solution performance degrades
- **Solution**: Increase solution ratio (e.g., 90/10), reduce epochs

**Problem**: Generated questions too simple/hard
- **Solution**: Curate GPT-4 data, adjust prompt templates

### Phase 2 Issues

**Problem**: Reward hacking (model exploits reward function)
- **Solution**: Inspect trajectories, adjust reward weights, add penalties

**Problem**: Mode collapse (all questions similar)
- **Solution**: Increase temperature, add diversity penalty, more diverse prompts

**Problem**: Training instability (performance fluctuates)
- **Solution**: Lower learning rate, increase beta (KL penalty), smaller batch size

## Cost Estimates

### Phase 1
- **GPT-4 data generation**: $10-100 (5k examples)
- **Training**: 6-12 hours on A100 (~$10-30 cloud cost)
- **Total**: ~$20-130

### Phase 2
- **Per iteration**: 4-8 hours A100 (~$5-20)
- **5 iterations**: ~$25-100
- **Total**: ~$25-100

## Next Steps

1. **Run Phase 1 MVP**: Test with 500 questions first
2. **Validate dual-task**: Ensure both capabilities work
3. **Scale up**: Full 5k question dataset
4. **Run Phase 2**: Multi-iteration self-improvement
5. **Analyze & iterate**: Adjust hyperparameters based on results

## Additional Resources

- **Plan file**: `.cursor/plans/dual-task_math_model_training_*.plan.md`
- **Existing verification**: `src/sft/step_verify_sympy.py`
- **Format validation**: `src/sft/solution_format.py`
- **Original pipeline**: `scripts/gsm8k_sft_pipeline.py`

## Credits

Implementation follows the dual-task training + RL self-improvement paradigm for math problem generation and solving.
