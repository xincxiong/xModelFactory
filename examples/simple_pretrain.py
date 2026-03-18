"""
Simple pre-training example with multi-GPU support.

This example demonstrates how to use xModelFactory for pre-training
a language model with support for DeepSpeed and DDP.

Usage:
    # Single GPU
    python simple_pretrain.py

    # Multi-GPU with automatic selection
    smart_train simple_pretrain.py

    # Multi-GPU with DeepSpeed
    ds_train simple_pretrain.py

    # Multi-GPU with DDP
    ddp_train simple_pretrain.py
"""

import torch
import os

from xmodel_factory import (
    ModelConfig,
    TrainConfig,
    OptimConfig,
    DataLoaderConfig,
    EvalConfig,
    PretrainConfig,
    Trainer,
    SimpleFileDataset,
)


def main():
    """Run simple pre-training example."""
    print("=" * 60)
    print("Simple Pre-training Example")
    print("=" * 60)

    # Check environment
    parallel_type = os.environ.get('PARALLEL_TYPE', 'none')
    print(f"Parallel type: {parallel_type}")

    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA not available, using CPU")

    print("-" * 60)

    # Create model configuration
    model_config = ModelConfig(
        vocab_size=32000,
        hidden_size=512,  # Small for demo
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=512,
    )

    # Create training configuration
    train_config = TrainConfig(
        n_epochs=2,
        batch_size=4,
        model_config=model_config,
        dataset_block_size=128,
        data_loader_config=DataLoaderConfig(
            data_loader_shuffle=True,
            data_loader_num_workers=0,
        ),
        optim_config=OptimConfig(
            optim_type='adam',
            initial_lr=1e-4,
            enable_lr_scheduler=False,
        ),
        eval_config=EvalConfig(
            max_seq_len=128,
            eval_batch_interval=50,
        ),
        pretrain_config=PretrainConfig(
            gradient_accumulation_steps=1,
        ),
    )

    # Create file dataset (using synthetic data for demo)
    file_dataset = SimpleFileDataset([
        "synthetic_train_0",
        "synthetic_train_1",
    ])
    train_config.file_dataset = file_dataset

    # Create trainer
    trainer = Trainer(
        train_config=train_config,
        eval_prompts=[
            "Hello, how are you?",
            "What is the weather like today?",
        ],
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
