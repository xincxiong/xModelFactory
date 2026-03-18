"""
Training configuration classes.
"""

from typing import Optional, Union, Callable, List, Mapping, Any, Tuple
from dataclasses import dataclass, field

import torch
from xmodel_factory.model_core import ModelConfig, VLMConfig


@dataclass(kw_only=True)
class DsOffloadConfig:
    """DeepSpeed ZeRO Offload configuration."""
    device: str = 'cpu'
    pin_memory: bool = True


@dataclass(kw_only=True)
class DsActivationCheckpointingConfig:
    """DeepSpeed activation checkpointing configuration."""
    partition_activations: bool = True
    cpu_checkpointing: bool = False
    contiguous_memory_optimization: bool = True
    number_checkpoints: Optional[int] = None
    synchronize_checkpoint_boundary: bool = False
    profile: bool = False


@dataclass(kw_only=True)
class DsZeROConfig:
    """Base DeepSpeed ZeRO configuration."""
    stage: int
    allgather_partitions: Optional[bool] = True
    allgather_bucket_size: Optional[int] = 5e8
    overlap_comm: Optional[bool] = True
    reduce_scatter: Optional[bool] = True
    reduce_bucket_size: Optional[Union[str, int]] = 5e8
    contiguous_gradients: Optional[bool] = True


@dataclass(kw_only=True)
class DsZero0Config(DsZeROConfig):
    """DeepSpeed ZeRO Stage 0 configuration."""
    stage: int = field(default=0, init=False)


@dataclass(kw_only=True)
class DsZero1Config(DsZeROConfig):
    """DeepSpeed ZeRO Stage 1 configuration."""
    stage: int = field(default=1, init=False)


@dataclass(kw_only=True)
class DsZero2Config(DsZeROConfig):
    """DeepSpeed ZeRO Stage 2 configuration."""
    stage: int = field(default=2, init=False)
    offload_optimizer: Optional[DsOffloadConfig] = None
    offload_param: Optional[DsOffloadConfig] = None


@dataclass(kw_only=True)
class DsZero3Config(DsZeROConfig):
    """DeepSpeed ZeRO Stage 3 configuration."""
    stage: int = field(default=3, init=False)
    sub_group_size: Optional[int] = 1e9
    stage3_prefetch_bucket_size: Optional[Union[str, int]] = 'auto'
    stage3_param_persistence_threshold: Optional[Union[str, int]] = 'auto'
    stage3_max_live_parameters: Optional[int] = 1e9
    stage3_max_reuse_distance: Optional[int] = 1e9
    stage3_gather_16bit_weights_on_model_save: Optional[bool] = True
    offload_optimizer: Optional[DsOffloadConfig] = None
    offload_param: Optional[DsOffloadConfig] = None


@dataclass(kw_only=True)
class DsFp16Config:
    """DeepSpeed FP16 configuration."""
    enabled: Union[str, bool] = 'auto'
    loss_scale: int = 0
    loss_scale_window: int = 1000
    initial_scale_power: int = 16
    hysteresis: int = 2
    min_loss_scale: int = 1
    fp16_opt_level: Optional[str] = 'O2'


@dataclass(kw_only=True)
class DsBf16Config:
    """DeepSpeed BF16 configuration."""
    enabled: bool = True


@dataclass(kw_only=True)
class DsConfig:
    """DeepSpeed configuration."""
    zero_config: Optional[DsZeROConfig] = field(default_factory=DsZero2Config)
    fp16_config: Optional[DsFp16Config] = field(default_factory=DsFp16Config)
    bf16_config: Optional[DsBf16Config] = field(default_factory=DsBf16Config)
    gradient_clipping: Optional[float] = 1.0
    activation_checkpointing: Optional[DsActivationCheckpointingConfig] = None


@dataclass(kw_only=True)
class DataLoaderConfig:
    """
    DataLoader configuration.

    Args:
        data_loader_pin_memory: Whether to pin memory in DataLoader
        data_loader_num_workers: Number of workers for DataLoader
        data_loader_shuffle: Whether to shuffle data
        data_loader_drop_last: Whether to drop last incomplete batch
    """
    data_loader_pin_memory: bool = False
    data_loader_num_workers: int = 0
    data_loader_shuffle: bool = False
    data_loader_drop_last: bool = True


@dataclass(kw_only=True)
class OptimConfig:
    """Optimizer configuration."""
    optim_type: str = 'adam'  # 'adam' or 'lion'
    auto_optimize_optimizer: bool = True
    enable_lr_scheduler: bool = False
    initial_lr: float = 1e-4
    weight_decay: Optional[float] = None
    betas: Optional[Tuple[float, float]] = None
    warmup_iters: Optional[int] = None
    max_lr: Optional[float] = None
    min_lr: Optional[float] = None
    cosine_annealing_period: Optional[int] = None
    cosine_annealing_period_mul: int = 0


@dataclass(kw_only=True)
class LossConfig:
    """Loss function configuration."""
    critical_tokens: Optional[List[int]] = None
    critical_alpha: float = 1.0
    aux_loss_coef: Optional[float] = 0.001


@dataclass(kw_only=True)
class KDConfig:
    """
    Knowledge Distillation configuration.

    Args:
        teacher_logits_provider: Function to provide teacher logits
        kd_coef: Weight for KD loss (0.0-1.0)
    """
    teacher_logits_provider: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    kd_coef: float = 0.4


@dataclass(kw_only=True)
class EvalConfig:
    """
    Evaluation configuration.

    Args:
        max_seq_len: Maximum sequence length for generation
        eval_batch_interval: Evaluate every N batches
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
    """
    max_seq_len: int = 2048
    eval_batch_interval: int = 100
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: Optional[float] = None


@dataclass(kw_only=True)
class PretrainConfig:
    """
    Pre-training configuration.

    Args:
        gradient_accumulation_steps: Number of gradient accumulation steps
        kd_config: Knowledge distillation configuration
    """
    gradient_accumulation_steps: int = 1
    kd_config: Optional[KDConfig] = None


@dataclass(kw_only=True)
class SFTConfig:
    """
    Supervised Fine-Tuning configuration.

    Args:
        mask_prompt: Whether to mask prompt tokens in loss
        gradient_accumulation_steps: Gradient accumulation steps
        kd_config: Knowledge distillation configuration
        image_tags_file_dataset: Dataset for image tags (VLM)
        pixel_values_provider: Function to provide pixel values (VLM)
        freeze_llm_model: Whether to freeze LLM parameters (VLM)
    """
    mask_prompt: bool = True
    gradient_accumulation_steps: int = 1
    kd_config: Optional[KDConfig] = None
    image_tags_file_dataset: Optional[Any] = None
    pixel_values_provider: Optional[Callable[[list[str]], torch.Tensor]] = None
    freeze_llm_model: bool = False


@dataclass(kw_only=True)
class DPOConfig:
    """
    Direct Preference Optimization configuration.

    Args:
        ref_model_checkpoint: Reference model checkpoint
        mask_prompt: Whether to mask prompt tokens
        gradient_accumulation_steps: Gradient accumulation steps
        loss_beta: DPO loss beta parameter
        loss_label_smoothing: Label smoothing parameter
        loss_ipo: Whether to use IPO loss
        nll_loss_coef: NLL loss coefficient
    """
    ref_model_checkpoint: Optional[Mapping[str, Any]] = None
    mask_prompt: bool = True
    gradient_accumulation_steps: int = 1
    loss_beta: float = 0.1
    loss_label_smoothing: float = 0.0
    loss_ipo: bool = False
    nll_loss_coef: Optional[float] = None


@dataclass(kw_only=True)
class PPOConfig:
    """
    Proximal Policy Optimization configuration.

    Args:
        ppo_epochs: Number of PPO epochs
        ppo_batch_size: PPO batch size
        ref_model_checkpoint: Reference model checkpoint
        value_model_checkpoint: Value model checkpoint
        value_optim_config: Value model optimizer config
        gradient_accumulation_steps: Gradient accumulation steps
        gamma: Discount factor
        lam: GAE lambda
        clip_eps: Clipping epsilon
        vf_coef: Value function coefficient
        kl_beta: KL penalty coefficient
        kl_estimator: KL estimator type ('k1' or 'k3')
        missing_eos_penalty: Penalty for missing EOS
        normalize_rewards: Whether to normalize rewards
        normalize_method: Reward normalization method
        whiten_rewards: Whether to whiten rewards
        gen_max_seq_len: Max sequence length for generation
        gen_temperature: Generation temperature
        gen_k: Top-k for generation
        gen_p: Top-p for generation
        gen_suppress_tokens: Tokens to suppress
    """
    ppo_epochs: int = 4
    ppo_batch_size: int = 64
    ref_model_checkpoint: Optional[Mapping[str, Any]] = None
    value_model_checkpoint: Optional[Mapping[str, Any]] = None
    value_optim_config: Optional['OptimConfig'] = None
    gradient_accumulation_steps: int = 1
    gamma: float = 1.0
    lam: float = 0.95
    clip_eps: float = 0.1
    vf_coef: float = 0.5
    kl_beta: float = 0.02
    kl_estimator: str = 'k1'
    missing_eos_penalty: Optional[float] = None
    normalize_rewards: bool = False
    normalize_method: str = 'RunningMeanStd'
    whiten_rewards: bool = False
    gen_max_seq_len: int = 512
    gen_temperature: Optional[float] = None
    gen_k: Optional[int] = None
    gen_p: Optional[float] = None
    gen_suppress_tokens: Optional[list[int]] = None


@dataclass(kw_only=True)
class GRPOConfig:
    """
    Group Relative Policy Optimization configuration.

    Args:
        grpo_steps: Number of GRPO steps
        group_size: Group size for GRPO
        mixup_alpha: Mixup alpha parameter
        loss_beta: Loss beta parameter
        loss_clip_eps: Clipping epsilon
        loss_clip_eps_high: Upper clipping epsilon
        loss_delta: Loss delta parameter
        loss_importance_sampling_level: IS level ('seq' or 'token')
        loss_type: Loss type ('grpo', 'bnpo', or 'dr_grpo')
        gen_max_seq_len: Max sequence length for generation
        gen_temperature: Generation temperature
        gen_k: Top-k for generation
        gen_p: Top-p for generation
        gen_suppress_tokens: Tokens to suppress
    """
    grpo_steps: int = 1
    group_size: int = 12
    mixup_alpha: float = 1.0
    loss_beta: float = 0.0
    loss_clip_eps: float = 3e-4
    loss_clip_eps_high: Optional[float] = 4e-4
    loss_delta: Optional[float] = None
    loss_importance_sampling_level: str = 'seq'
    loss_type: str = 'grpo'
    gen_max_seq_len: int = 512
    gen_temperature: Optional[float] = None
    gen_k: Optional[int] = None
    gen_p: Optional[float] = None
    gen_suppress_tokens: Optional[list[int]] = None


@dataclass(kw_only=True)
class TrainConfig:
    """
    Main training configuration.

    Args:
        n_epochs: Number of training epochs
        batch_size: Batch size per device
        model_config: Model configuration
        init_state_dict: Initial state dict for model
        file_dataset: Training dataset
        dataset_block_size: Block size for dataset
        data_loader_config: DataLoader configuration
        loss_config: Loss configuration
        ds_config: DeepSpeed configuration
        eval_config: Evaluation configuration
        optim_config: Optimizer configuration
        pretrain_config: Pre-training config (for Trainer)
        sft_config: SFT config (for SFTTrainer)
        dpo_config: DPO config (for DPOTrainer)
        ppo_config: PPO config (for PPOTrainer)
        grpo_config: GRPO config (for GRPOTrainer)
    """
    n_epochs: int = 3
    batch_size: int = 4
    model_config: Union[ModelConfig, VLMConfig] = field(default_factory=ModelConfig)
    init_state_dict: Optional[Mapping[str, Any]] = None
    file_dataset: Optional[Any] = None
    dataset_block_size: int = 2048
    data_loader_config: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    loss_config: LossConfig = field(default_factory=LossConfig)
    optim_config: OptimConfig = field(default_factory=OptimConfig)
    ds_config: Optional[DsConfig] = field(default_factory=DsConfig)
    eval_config: EvalConfig = field(default_factory=EvalConfig)

    pretrain_config: Optional[PretrainConfig] = field(default_factory=PretrainConfig)
    sft_config: Optional[SFTConfig] = None
    dpo_config: Optional[DPOConfig] = None
    ppo_config: Optional[PPOConfig] = None
    grpo_config: Optional[GRPOConfig] = None
