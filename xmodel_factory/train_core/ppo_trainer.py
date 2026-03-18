"""
Proximal Policy Optimization trainer.
"""

from typing import List, Tuple
from torch.utils.data import Dataset

from .base_trainer import BaseTrainer
from .train_configs import TrainConfig


class RLDataset(Dataset):
    """RL dataset for PPO training."""

    def __init__(self, file_path: str):
        self.file_path = file_path

        # For demo, create synthetic prompts
        import torch
        self.data = []
        for _ in range(100):
            prompt_len = torch.randint(10, 50, (1,)).item()
            prompt = torch.randint(0, 32000, (prompt_len,))
            self.data.append({'prompt': prompt})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PPOTrainer(BaseTrainer):
    """Trainer for Proximal Policy Optimization."""

    def __init__(
        self,
        *,
        train_config: TrainConfig,
        eval_prompts: List[str],
    ):
        """Initialize PPO trainer."""
        ppo_config = train_config.ppo_config or {}

        super().__init__(
            train_config=train_config,
            eval_prompts=eval_prompts,
            gradient_accumulation_steps=ppo_config.get('gradient_accumulation_steps', 1)
        )

        print("[PPOTrainer] PPO trainer initialized")

    def _create_dataset(self, file_idx) -> Tuple[Dataset, str]:
        """Create RL dataset."""
        if self.train_config.file_dataset:
            file_path = self.train_config.file_dataset[file_idx]
        else:
            file_path = f"synthetic_rl_{file_idx}"

        dataset = RLDataset(file_path=file_path)
        return dataset, file_path

    def train(self):
        """PPO training loop."""
        print("[PPOTrainer] Starting PPO training...")
        # For demo, simplified training
        # In real implementation, would:
        # 1. Generate completions with old policy
        # 2. Compute advantages with value function
        # 3. Update policy with clipped objective
        super().train()
