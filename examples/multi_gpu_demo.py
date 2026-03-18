"""
Multi-GPU Training Demo for xModelFactory.

This demo shows how to run training on multiple GPUs using different parallel strategies.

Usage:
    # Check GPU setup first
    python multi_gpu_demo.py --check

    # Run with specific parallel strategy
    python multi_gpu_demo.py --strategy auto      # Auto-select best strategy
    python multi_gpu_demo.py --strategy deepspeed  # Use DeepSpeed
    python multi_gpu_demo.py --strategy ddp        # Use PyTorch DDP
    python multi_gpu_demo.py --strategy single     # Single GPU

    # Or use the convenience scripts:
    smart_train multi_gpu_demo.py   # Auto-select
    ds_train multi_gpu_demo.py      # DeepSpeed
    ddp_train multi_gpu_demo.py     # DDP
"""

import os
import sys
import argparse
import torch


def check_gpu_setup():
    """Check and display GPU setup information."""
    print("=" * 60)
    print("GPU Setup Check")
    print("=" * 60)

    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")

        for i in range(gpu_count):
            try:
                props = torch.cuda.get_device_properties(i)
                print(f"\n  GPU {i}: {props.name}")
                print(f"    Memory: {props.total_memory / 1e9:.2f} GB")
                print(f"    Compute Capability: {props.major}.{props.minor}")
            except Exception as e:
                print(f"\n  GPU {i}: Unable to get properties ({e})")
    else:
        print("No CUDA devices available. Will use CPU.")

    # Check DeepSpeed
    print("\n" + "-" * 60)
    try:
        import deepspeed
        print(f"DeepSpeed: Available (v{deepspeed.__version__})")
    except ImportError:
        print("DeepSpeed: Not installed")
        print("  Install with: pip install deepspeed")

    # Check environment variables
    print("\n" + "-" * 60)
    print("Environment Variables:")
    env_vars = ['CUDA_VISIBLE_DEVICES', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'PARALLEL_TYPE']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")

    print("\n" + "=" * 60)


def run_training(strategy: str = 'auto'):
    """Run training with specified parallel strategy."""
    print("=" * 60)
    print(f"Multi-GPU Training Demo (Strategy: {strategy})")
    print("=" * 60)

    # Set parallel type based on strategy
    if strategy == 'auto':
        # Auto-select: prefer DeepSpeed if available
        try:
            import deepspeed
            os.environ['PARALLEL_TYPE'] = 'ds'
            print("[Demo] Selected: DeepSpeed")
        except ImportError:
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                os.environ['PARALLEL_TYPE'] = 'ddp'
                print("[Demo] Selected: DDP")
            else:
                os.environ['PARALLEL_TYPE'] = 'none'
                print("[Demo] Selected: Single device")
    elif strategy == 'deepspeed':
        os.environ['PARALLEL_TYPE'] = 'ds'
        print("[Demo] Using DeepSpeed")
    elif strategy == 'ddp':
        os.environ['PARALLEL_TYPE'] = 'ddp'
        print("[Demo] Using DDP")
    else:
        os.environ['PARALLEL_TYPE'] = 'none'
        print("[Demo] Using single device")

    # Import after setting environment
    from xmodel_factory import (
        ModelConfig, TrainConfig, OptimConfig,
        DataLoaderConfig, EvalConfig, PretrainConfig,
        Trainer, SimpleFileDataset,
    )

    # Create model config (small for demo)
    model_config = ModelConfig(
        vocab_size=32000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=256,
    )

    print(f"\nModel Config:")
    print(f"  Hidden size: {model_config.hidden_size}")
    print(f"  Layers: {model_config.num_hidden_layers}")
    print(f"  Attention heads: {model_config.num_attention_heads}")

    # Create training config
    train_config = TrainConfig(
        n_epochs=1,
        batch_size=2,
        model_config=model_config,
        dataset_block_size=64,
        data_loader_config=DataLoaderConfig(
            data_loader_shuffle=True,
            data_loader_num_workers=0,
        ),
        optim_config=OptimConfig(
            optim_type='adam',
            initial_lr=1e-4,
        ),
        eval_config=EvalConfig(
            max_seq_len=64,
            eval_batch_interval=10,
        ),
        pretrain_config=PretrainConfig(
            gradient_accumulation_steps=1,
        ),
    )

    # Create synthetic dataset
    file_dataset = SimpleFileDataset([
        "synthetic_data_0",
        "synthetic_data_1",
    ])
    train_config.file_dataset = file_dataset

    print(f"\nTraining Config:")
    print(f"  Epochs: {train_config.n_epochs}")
    print(f"  Batch size: {train_config.batch_size}")
    print(f"  Block size: {train_config.dataset_block_size}")

    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        train_config=train_config,
        eval_prompts=["Hello world!"],
    )

    print("Starting training...")
    print("-" * 60)

    # Run training
    trainer.train()

    print("-" * 60)
    print("Training completed!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Multi-GPU Training Demo')
    parser.add_argument('--check', action='store_true',
                        help='Check GPU setup without training')
    parser.add_argument('--strategy', type=str, default='auto',
                        choices=['auto', 'deepspeed', 'ddp', 'single'],
                        help='Parallel strategy to use')

    args = parser.parse_args()

    if args.check:
        check_gpu_setup()
    else:
        run_training(args.strategy)


if __name__ == '__main__':
    main()