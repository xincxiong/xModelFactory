"""
Pre-training trainer.
"""

from typing import List, Tuple
from torch.utils.data import Dataset

from .base_trainer import BaseTrainer
from .train_configs import TrainConfig


class PretrainDataset(Dataset):
    """Simple pretraining dataset."""

    def __init__(self, file_path: str, block_size: int, stride: int):
        self.file_path = file_path
        self.block_size = block_size
        self.stride = stride

        # For demo, create synthetic data
        import torch
        self.data = torch.randint(0, 32000, (1000, block_size))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        import torch
        tokens = self.data[idx]
        return {
            'inputs': tokens,
            'labels': tokens.clone()
        }


class Trainer(BaseTrainer):
    """Trainer for pre-training."""

    def __init__(
        self,
        *,
        train_config: TrainConfig,
        eval_prompts: List[str],
    ):
        """Initialize pre-training trainer."""
        super().__init__(
            train_config=train_config,
            eval_prompts=eval_prompts,
            kd_config=train_config.pretrain_config.kd_config if train_config.pretrain_config else None,
            gradient_accumulation_steps=(
                train_config.pretrain_config.gradient_accumulation_steps
                if train_config.pretrain_config else 1
            )
        )

    def _create_dataset(self, file_idx) -> Tuple[Dataset, str]:
        """Create pre-training dataset."""
        if self.train_config.file_dataset:
            file_path = self.train_config.file_dataset[file_idx]
        else:
            # Use synthetic data for demo
            file_path = f"synthetic_{file_idx}"

        dataset = PretrainDataset(
            file_path=file_path,
            block_size=self.train_config.dataset_block_size,
            stride=self.train_config.dataset_block_size
        )

        return dataset, file_path
