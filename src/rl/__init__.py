# RL components for self-improvement

from src.rl.mdp_components import Action, State, Trajectory, Transition
from src.rl.rollout_buffer import GAEComputer, RolloutBuffer
from src.rl.question_classifier import QuestionClassifier
from src.rl.curriculum_manager import CurriculumManager
from src.rl.question_quality_evaluator import QuestionQualityEvaluator
from src.rl.expert_panel import SimulatedExpertPanel
from src.rl.quality_filter import QualityFilter
from src.rl.replay_buffer import GenerationalReplayBuffer

# Optional heavy imports (transformers / torch model deps).
try:
    from src.rl.value_network import ValueHead
    from src.rl.reward_calculator import RewardCalculator
    from src.rl.ppo_trainer import PPOTrainer
    from src.rl.math_environment import MathEnvironment
    from src.rl.math_environment_consensus import ConsensusMathEnvironment
    from src.rl.math_environment_curriculum import CurriculumMathEnvironment
except ModuleNotFoundError:  # pragma: no cover
    ValueHead = None
    RewardCalculator = None
    PPOTrainer = None
    MathEnvironment = None
    ConsensusMathEnvironment = None
    CurriculumMathEnvironment = None

__all__ = [
    "State",
    "Action",
    "Transition",
    "Trajectory",
    "ValueHead",
    "RewardCalculator",
    "GAEComputer",
    "RolloutBuffer",
    "PPOTrainer",
    "MathEnvironment",
    "ConsensusMathEnvironment",
    "CurriculumMathEnvironment",
    "QuestionClassifier",
    "CurriculumManager",
    "QuestionQualityEvaluator",
    "SimulatedExpertPanel",
    "QualityFilter",
    "GenerationalReplayBuffer",
]
