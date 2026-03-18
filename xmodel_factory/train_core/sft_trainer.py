"""
Supervised Fine-Tuning trainer.
"""

from typing import List, Tuple
from torch.utils.data import Dataset

from .base_trainer import BaseTrainer
from .train_configs import TrainConfig


class SFTDataset(Dataset):
    """SFT dataset with prompt-completion pairs."""

    def __init__(self, file_path: str, block_size: int, mask_prompt: bool = True):
        self.file_path = file_path
        self.block_size = block_size
        self.mask_prompt = mask_prompt

        # For demo, create synthetic data
        import torch
        self.data = []
        for _ in range(100):
            prompt_len = torch.randint(10, 50, (1,)).item()
            completion_len = torch.randint(20, 100, (1,)).item()

            prompt = torch.randint(0, 32000, (prompt_len,))
            completion = torch.randint(0, 32000, (completion_len,))

            self.data.append({
                'prompt': prompt,
                'completion': completion,
                'prompt_len': prompt_len
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        completion = item['completion']

        # Concatenate prompt and completion
        full_sequence = torch.cat([prompt, completion])

        # Truncate if needed
        if len(full_sequence) > self.block_size:
            full_sequence = full_sequence[:self.block_size]

        # Create labels (mask prompt if needed)
        labels = full_sequence.clone()
        if self.mask_prompt:
            labels[:len(prompt)] = -100  # Ignore prompt in loss

        return {
            'inputs': full_sequence,
            'labels': labels
        }


class SFTTrainer(BaseTrainer):
    """Trainer for Supervised Fine-Tuning."""

    def __init__(
        self,
        *,
        train_config: TrainConfig,
        eval_prompts: List[str],
        eval_image_tags: List[str] = None,
    ):
        """Initialize SFT trainer."""
        sft_config = train_config.sft_config or {}

        super().__init__(
            train_config=train_config,
            eval_prompts=eval_prompts,
            kd_config=sft_config.get('kd_config'),
            gradient_accumulation_steps=sft_config.get('gradient_accumulation_steps', 1)
        )

        self.eval_image_tags = eval_image_tags or []

    def _create_dataset(self, file_idx) -> Tuple[Dataset, str]:
        """Create SFT dataset."""
        if self.train_config.file_dataset:
            file_path = self.train_config.file_dataset[file_idx]
        else:
            file_path = f"synthetic_sft_{file_idx}"

        mask_prompt = True
        if self.train_config.sft_config:
            mask_prompt = getattr(self.train_config.sft_config, 'mask_prompt', True)

        dataset = SFTDataset(
            file_path=file_path,
            block_size=self.train_config.dataset_block_size,
            mask_prompt=mask_prompt
        )

        return dataset, file_path
