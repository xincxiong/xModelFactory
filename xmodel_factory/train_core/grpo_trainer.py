"""
Group Relative Policy Optimization trainer.
"""

from typing import List, Tuple, Callable
from torch.utils.data import Dataset

from .base_trainer import BaseTrainer
from .train_configs import TrainConfig


class GRPOTrainer(BaseTrainer):
    """Trainer for Group Relative Policy Optimization."""

    def __init__(
        self,
        *,
        train_config: TrainConfig,
        reward_func: Callable,
        eval_prompts: List[str],
    ):
        """Initialize GRPO trainer."""
        grpo_config = train_config.grpo_config or {}

        super().__init__(
            train_config=train_config,
            eval_prompts=eval_prompts,
            gradient_accumulation_steps=1  # GRPO doesn't use gradient accumulation
        )

        self.reward_func = reward_func
        self.group_size = grpo_config.get('group_size', 12)

        print("[GRPOTrainer] GRPO trainer initialized")

    def _create_dataset(self, file_idx) -> Tuple[Dataset, str]:
        """Create RL dataset for GRPO."""
        if self.train_config.file_dataset:
            file_path = self.train_config.file_dataset[file_idx]
        else:
            file_path = f"synthetic_grpo_{file_idx}"

        from .ppo_trainer import RLDataset
        dataset = RLDataset(file_path=file_path)
        return dataset, file_path

    def train(self):
        """GRPO training loop."""
        print("[GRPOTrainer] Starting GRPO training...")
        print(f"[GRPOTrainer] Group size: {self.group_size}")

        # For demo, simplified training
        # In real implementation, would:
        # 1. Generate multiple completions for each prompt (group)
        # 2. Compute relative rewards within group
        # 3. Update policy with GRPO objective
        super().train()
