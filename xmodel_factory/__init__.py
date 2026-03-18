"""
xModelFactory - A comprehensive training framework for Large Language Models (LLM) and Vision Language Models (VLM).

This package provides a complete training solution supporting:
- Pre-training
- Supervised Fine-tuning (SFT)
- Direct Preference Optimization (DPO)
- Proximal Policy Optimization (PPO)
- Group Relative Policy Optimization (GRPO)
- Multi-GPU distributed training (DeepSpeed, DDP)
"""

__version__ = "1.0.0"
__author__ = "xModelFactory Team"

from .model_core import (
    ModelConfig,
    VLMConfig,
    LlmModel,
    VlmModel,
    KVCache,
    attention_masks,
)

from .train_core import (
    Trainer,
    SFTTrainer,
    DPOTrainer,
    PPOTrainer,
    GRPOTrainer,
    TrainConfig,
    OptimConfig,
    DsConfig,
    SFTConfig,
    DPOConfig,
    PPOConfig,
    GRPOConfig,
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
    "attention_masks",
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
