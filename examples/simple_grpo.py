"""
Simple GRPO (Group Relative Policy Optimization) example.

This example demonstrates how to use xModelFactory for training
with GRPO, a reinforcement learning method.

Usage:
    python simple_grpo.py
    smart_train simple_grpo.py
"""

import torch
import os

from xmodel_factory import (
    ModelConfig,
    TrainConfig,
    OptimConfig,
    DataLoaderConfig,
    EvalConfig,
    GRPOConfig,
    GRPOTrainer,
    SimpleFileDataset,
)


def reward_function(prompts, completions, answers):
    """
    Simple reward function for GRPO.

    Args:
        prompts: List of prompts
        completions: List of model completions
        answers: List of reference answers

    Returns:
        List of reward scores
    """
    rewards = []
    for completion, answer in zip(completions, answers):
        # Simple reward: length-based + contains answer
        reward = 0.0

        # Reward for length
        if len(completion) > 10:
            reward += 0.5

        # Reward for containing answer
        if answer.lower() in completion.lower():
            reward += 1.0

        rewards.append(reward)

    return rewards


def main():
    """Run simple GRPO example."""
    print("=" * 60)
    print("Simple GRPO Example")
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
        n_epochs=1,
        batch_size=2,
        model_config=model_config,
        dataset_block_size=256,
        data_loader_config=DataLoaderConfig(),
        optim_config=OptimConfig(
            optim_type='adam',
            initial_lr=1e-5,
        ),
        eval_config=EvalConfig(
            max_seq_len=256,
            eval_batch_interval=20,
        ),
        grpo_config=GRPOConfig(
            grpo_steps=2,
            group_size=4,
            gen_max_seq_len=128,
            gen_temperature=0.8,
        ),
    )

    # Dataset
    file_dataset = SimpleFileDataset([
        "synthetic_grpo_data_0",
    ])
    train_config.file_dataset = file_dataset

    # Create trainer
    trainer = GRPOTrainer(
        train_config=train_config,
        reward_func=reward_function,
        eval_prompts=[
            "Question: What is 2+2?\nAnswer:",
            "Question: What is the capital of France?\nAnswer:",
        ],
    )

    # Train
    print("\nStarting GRPO training...")
    trainer.train()

    print("\n" + "=" * 60)
    print("GRPO training completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
