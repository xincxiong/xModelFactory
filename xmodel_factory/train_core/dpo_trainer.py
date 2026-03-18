"""
Direct Preference Optimization trainer.
"""

from typing import List, Tuple
from torch.utils.data import Dataset

from .base_trainer import BaseTrainer
from .train_configs import TrainConfig


class DPODataset(Dataset):
    """DPO dataset with chosen and rejected completions."""

    def __init__(self, file_path: str, block_size: int):
        self.file_path = file_path
        self.block_size = block_size

        # For demo, create synthetic data
        import torch
        self.data = []
        for _ in range(100):
            prompt_len = torch.randint(10, 50, (1,)).item()
            chosen_len = torch.randint(20, 100, (1,)).item()
            rejected_len = torch.randint(20, 100, (1,)).item()

            prompt = torch.randint(0, 32000, (prompt_len,))
            chosen = torch.randint(0, 32000, (chosen_len,))
            rejected = torch.randint(0, 32000, (rejected_len,))

            self.data.append({
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Create chosen and rejected sequences
        chosen_full = torch.cat([item['prompt'], item['chosen']])
        rejected_full = torch.cat([item['prompt'], item['rejected']])

        # Truncate
        if len(chosen_full) > self.block_size:
            chosen_full = chosen_full[:self.block_size]
        if len(rejected_full) > self.block_size:
            rejected_full = rejected_full[:self.block_size]

        # Create labels (only compute loss on completion)
        prompt_len = len(item['prompt'])
        chosen_labels = chosen_full.clone()
        rejected_labels = rejected_full.clone()
        chosen_labels[:prompt_len] = -100
        rejected_labels[:prompt_len] = -100

        return {
            'chosen_inputs': chosen_full,
            'chosen_labels': chosen_labels,
            'rejected_inputs': rejected_full,
            'rejected_labels': rejected_labels
        }


class DPOTrainer(BaseTrainer):
    """Trainer for Direct Preference Optimization."""

    def __init__(
        self,
        *,
        train_config: TrainConfig,
        eval_prompts: List[str],
    ):
        """Initialize DPO trainer."""
        dpo_config = train_config.dpo_config or {}

        super().__init__(
            train_config=train_config,
            eval_prompts=eval_prompts,
            gradient_accumulation_steps=dpo_config.get('gradient_accumulation_steps', 1)
        )

    def _create_dataset(self, file_idx) -> Tuple[Dataset, str]:
        """Create DPO dataset."""
        if self.train_config.file_dataset:
            file_path = self.train_config.file_dataset[file_idx]
        else:
            file_path = f"synthetic_dpo_{file_idx}"

        dataset = DPODataset(
            file_path=file_path,
            block_size=self.train_config.dataset_block_size
        )

        return dataset, file_path

    def train(self):
        """DPO training loop with preference optimization."""
        # For demo, use standard training loop
        # In real implementation, would compute DPO loss with reference model
        print("[DPOTrainer] Starting DPO training...")
        super().train()
