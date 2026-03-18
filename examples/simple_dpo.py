"""
Simple DPO (Direct Preference Optimization) example.

This example demonstrates how to use xModelFactory for training
a model with preference data (chosen vs rejected completions).

Usage:
    python simple_dpo.py
    smart_train simple_dpo.py
"""

import torch
import os

from xmodel_factory import (
    ModelConfig,
    TrainConfig,
    OptimConfig,
    DataLoaderConfig,
    EvalConfig,
    DPOConfig,
    DPOTrainer,
    SimpleFileDataset,
)


def main():
    """Run simple DPO example."""
    print("=" * 60)
    print("Simple DPO Example")
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
        batch_size=2,
        model_config=model_config,
        dataset_block_size=256,
        data_loader_config=DataLoaderConfig(
            data_loader_shuffle=True,
        ),
        optim_config=OptimConfig(
            optim_type='adam',
            initial_lr=1e-5,
        ),
        eval_config=EvalConfig(
            max_seq_len=256,
            eval_batch_interval=50,
        ),
        dpo_config=DPOConfig(
            ref_model_checkpoint=None,
            mask_prompt=True,
            gradient_accumulation_steps=1,
            loss_beta=0.1,
        ),
    )

    # Dataset
    file_dataset = SimpleFileDataset([
        "synthetic_dpo_data_0",
    ])
    train_config.file_dataset = file_dataset

    # Create trainer
    trainer = DPOTrainer(
        train_config=train_config,
        eval_prompts=[
            "Question: What is machine learning?\nAnswer:",
        ],
    )

    # Train
    print("\nStarting DPO training...")
    trainer.train()

    print("\n" + "=" * 60)
    print("DPO training completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
