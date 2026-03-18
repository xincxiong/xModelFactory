"""
xModelFactory public package exports.

The package is organized into three higher-level namespaces:
- `configs`: structured configuration objects
- `models`: model implementations and attention utilities
- `trainers`: training entry points for each stage

Legacy modules such as `model_core` and `train_core` remain available as
compatibility layers.
"""

__version__ = "1.0.0"
__author__ = "xModelFactory Team"

from .configs import (
    ModelConfig,
    VLMConfig,
    TrainConfig,
    OptimConfig,
    DsConfig,
    SFTConfig,
    DPOConfig,
    PPOConfig,
    GRPOConfig,
)

from .models import (
    LlmModel,
    VlmModel,
    KVCache,
    prepare_decoder_attention_mask,
    make_causal_mask,
    expand_mask,
)

from .trainers import (
    Trainer,
    SFTTrainer,
    DPOTrainer,
    PPOTrainer,
    GRPOTrainer,
)

from .train_core import (
    FileDataset,
    generate,
    streaming_generate,
)

__all__ = [
    # Model Core
    "ModelConfig",
    "VLMConfig",
    "LlmModel",
    "VlmModel",
    "KVCache",
    "prepare_decoder_attention_mask",
    "make_causal_mask",
    "expand_mask",
    # Train Core
    "Trainer",
    "SFTTrainer",
    "DPOTrainer",
    "PPOTrainer",
    "GRPOTrainer",
    "TrainConfig",
    "OptimConfig",
    "DsConfig",
    "SFTConfig",
    "DPOConfig",
    "PPOConfig",
    "GRPOConfig",
    "FileDataset",
    "generate",
    "streaming_generate",
]
