"""
Training tools and utilities.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, List

import torch

from .parallel import DsParallel, DdpParallel, NoneParallel


PARALLEL_TYPES = {
    'ds': DsParallel,
    'ddp': DdpParallel,
    'none': NoneParallel
}


class TrainerTools:
    """Singleton for training tools and utilities."""

    _instance: Optional['TrainerTools'] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if TrainerTools._initialized:
            return

        TrainerTools._initialized = True

        # Initialize parallel backend
        self.parallel = self._init_parallel()

        # Initialize tokenizer (placeholder)
        self.tokenizer = SimpleTokenizer()

        # Whether to use AMP (Automatic Mixed Precision)
        self.use_amp = (
            'cuda' in self.parallel.device and
            not isinstance(self.parallel, DsParallel)
        )

        print(f"[TrainerTools] World size: {self.parallel.world_size}, "
              f"Use AMP: {self.use_amp}, Device: {self.parallel.device}")

    def _init_parallel(self):
        """Initialize parallel backend based on environment."""
        parallel_type = os.environ.get('PARALLEL_TYPE', 'none')
        print(f"[TrainerTools] Parallel type: {parallel_type}")

        if parallel_type not in PARALLEL_TYPES:
            raise ValueError(f"Unknown parallel type: {parallel_type}. "
                           f"Choose from: {list(PARALLEL_TYPES.keys())}")

        return PARALLEL_TYPES[parallel_type]()


class SimpleTokenizer:
    """Simple tokenizer for demonstration purposes."""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.pad = 0  # PAD token ID
        self.eos = 2  # EOS token ID
        self.bos = 1  # BOS token ID

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Encode text to token IDs."""
        # Simple character-level encoding for demo
        tokens = [ord(c) % self.vocab_size for c in text]
        if max_length:
            tokens = tokens[:max_length-1] + [self.eos]
        else:
            tokens.append(self.eos)
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Decode tokens to text."""
        # Filter out special tokens
        tokens = [t for t in tokens if t not in [self.pad, self.eos, self.bos]]
        return ''.join(chr(t) if t < 128 else '?' for t in tokens)


class FileDataset(ABC):
    """Abstract base class for file-based datasets."""

    @abstractmethod
    def __len__(self) -> int:
        """Return number of files in dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> str:
        """Return file path at index."""
        pass


class SimpleFileDataset(FileDataset):
    """Simple file dataset implementation."""

    def __init__(self, file_paths: List[str]):
        self.file_paths = file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> str:
        return self.file_paths[idx]


def estimate_data_size(
    file_dataset: FileDataset,
    block_size: int,
    dataset_type: str = 'pretrain'
) -> int:
    """
    Estimate dataset size.

    Args:
        file_dataset: File dataset
        block_size: Block size for dataset
        dataset_type: Type of dataset ('pretrain', 'sft', 'dpo', 'rl')

    Returns:
        Estimated number of samples
    """
    # Placeholder implementation
    # In real implementation, this would actually read and count samples
    total_size = 0
    num_files = len(file_dataset)

    # Rough estimate: 1000 samples per file
    estimated_samples_per_file = 1000

    for idx in range(num_files):
        file_path = file_dataset[idx]
        # In real implementation, load and count actual samples
        total_size += estimated_samples_per_file

    return total_size


def extract_policy_weights_from_ppo(model_config, ppo_weights):
    """
    Extract policy model weights from PPO checkpoint.

    Args:
        model_config: Model configuration
        ppo_weights: PPO model weights

    Returns:
        Policy model state dict
    """
    # Placeholder - in real implementation would separate policy from value
    return ppo_weights


def extract_value_weights_from_ppo(model_config, ppo_weights):
    """Extract value model weights from PPO checkpoint."""
    # Placeholder - in real implementation would separate value from policy
    return ppo_weights
