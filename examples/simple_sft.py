"""
Simple SFT (Supervised Fine-Tuning) example with multi-GPU support.

This example demonstrates how to use xModelFactory for fine-tuning
a language model with instruction-following data.

Usage:
    # Single GPU
    python simple_sft.py

    # Multi-GPU with automatic selection
    smart_train simple_sft.py
"""

import torch
import os

from xmodel_factory import (
    ModelConfig,
    TrainConfig,
    OptimConfig,
    DataLoaderConfig,
    EvalConfig,
    SFTConfig,
    SFTTrainer,
    SimpleFileDataset,
)


def main():
    """Run simple SFT example."""
    print("=" * 60)
    print("Simple SFT Example")
    print("=" * 60)

    parallel_type = os.environ.get('PARALLEL_TYPE', 'none')
    print(f"Parallel type: {parallel_type}")

    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
    else:
        print("Using CPU")

    print("-" * 60)

    # Model configuration
    model_config = ModelConfig(
        vocab_size=32000,
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=512,
    )

    # Training configuration
    train_config = TrainConfig(
        n_epochs=2,
        batch_size=4,
        model_config=model_config,
        dataset_block_size=256,
        data_loader_config=DataLoaderConfig(
            data_loader_shuffle=True,
        ),
        optim_config=OptimConfig(
            optim_type='adam',
            initial_lr=5e-5,
        ),
        eval_config=EvalConfig(
            max_seq_len=256,
            eval_batch_interval=50,
        ),
        sft_config=SFTConfig(
            mask_prompt=True,
            gradient_accumulation_steps=1,
        ),
    )

    # Dataset
    file_dataset = SimpleFileDataset([
        "synthetic_sft_data_0",
        "synthetic_sft_data_1",
    ])
    train_config.file_dataset = file_dataset

    # Create trainer
    trainer = SFTTrainer(
        train_config=train_config,
        eval_prompts=[
            "Instruction: What is AI?\nResponse:",
            "Instruction: Explain quantum computing\nResponse:",
        ],
    )

    # Train
    print("\nStarting SFT training...")
    trainer.train()

    print("\n" + "=" * 60)
    print("SFT training completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
