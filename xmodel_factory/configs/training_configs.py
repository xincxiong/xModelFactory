"""
Training configuration classes.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple, Any, Union
import torch

from .model_configs import ModelConfig, VLMConfig


@dataclass(kw_only=True)
class OptimConfig:
    """
    Optimizer configuration.

    Args:
        optim_type: Optimizer type ('adam', 'adamw', 'lion')
        auto_optimize_optimizer: Whether to auto-optimize
        enable_lr_scheduler: Whether to use LR scheduler
        initial_lr: Initial learning rate
        weight_decay: Weight decay coefficient
        betas: Adam betas tuple
        warmup_iters: Warmup iterations
        max_lr: Maximum learning rate
        min_lr: Minimum learning rate
        cosine_annealing_period: Cosine annealing period
        cosine_annealing_period_mul: Period multiplier
    """
    optim_type: str = 'adamw'
    auto_optimize_optimizer: bool = True
    enable_lr_scheduler: bool = False
    initial_lr: float = 1e-4
    weight_decay: Optional[float] = 0.01
    betas: Optional[Tuple[float, float]] = None
    warmup_iters: Optional[int] = None
    max_lr: Optional[float] = None
    min_lr: Optional[float] = None
    cosine_annealing_period: Optional[int] = None
    cosine_annealing_period_mul: int = 0

    def __post_init__(self):
        """Set default betas if not provided."""
        if self.betas is None:
            self.betas = (0.9, 0.999) if self.optim_type != 'lion' else (0.95, 0.98)


@dataclass(kw_only=True)
class LossConfig:
    """
    Loss function configuration.

    Args:
        critical_tokens: Token IDs to emphasize in loss
        critical_alpha: Weight for critical tokens
        aux_loss_coef: Auxiliary loss coefficient
        label_smoothing: Label smoothing factor
    """
    critical_tokens: Optional[List[int]] = None
    critical_alpha: float = 1.0
    aux_loss_coef: Optional[float] = 0.001
    label_smoothing: float = 0.0


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
        repetition_penalty: Repetition penalty
    """
    max_seq_len: int = 2048
    eval_batch_interval: int = 100
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: Optional[int] = None
    repetition_penalty: float = 1.0


@dataclass(kw_only=True)
class DataLoaderConfig:
    """
    DataLoader configuration.

    Args:
        pin_memory: Whether to pin memory
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
        prefetch_factor: Prefetch factor for workers
        persistent_workers: Whether to keep workers alive
    """
    pin_memory: bool = False
    num_workers: int = 0
    shuffle: bool = False
    drop_last: bool = True
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False


@dataclass(kw_only=True)
class KDConfig:
    """
    Knowledge Distillation configuration.

    Args:
        teacher_logits_provider: Function to provide teacher logits
        kd_coef: Weight for KD loss (0.0-1.0)
        temperature: Temperature for softmax in KD
    """
    teacher_logits_provider: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
    kd_coef: float = 0.4
    temperature: float = 1.0


@dataclass(kw_only=True)
class PretrainConfig:
    """
    Pre-training configuration.

    Args:
        gradient_accumulation_steps: Number of gradient accumulation steps
        kd_config: Knowledge distillation configuration
        max_seq_len: Maximum sequence length
    """
    gradient_accumulation_steps: int = 1
    kd_config: Optional[KDConfig] = None
    max_seq_len: int = 2048


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
        max_seq_len: Maximum sequence length
    """
    mask_prompt: bool = True
    gradient_accumulation_steps: int = 1
    kd_config: Optional[KDConfig] = None
    image_tags_file_dataset: Optional[Any] = None
    pixel_values_provider: Optional[Callable[[List[str]], torch.Tensor]] = None
    freeze_llm_model: bool = False
    max_seq_len: int = 2048


@dataclass(kw_only=True)
class DPOConfig:
    """
    Direct Preference Optimization configuration.

    Args:
        ref_model_checkpoint: Reference model checkpoint path
        mask_prompt: Whether to mask prompt tokens
        gradient_accumulation_steps: Gradient accumulation steps
        loss_beta: DPO loss beta parameter
        loss_label_smoothing: Label smoothing parameter
        loss_ipo: Whether to use IPO loss
        nll_loss_coef: NLL loss coefficient
        max_seq_len: Maximum sequence length
    """
    ref_model_checkpoint: Optional[str] = None
    mask_prompt: bool = True
    gradient_accumulation_steps: int = 1
    loss_beta: float = 0.1
    loss_label_smoothing: float = 0.0
    loss_ipo: bool = False
    nll_loss_coef: Optional[float] = None
    max_seq_len: int = 2048


@dataclass(kw_only=True)
class PPOConfig:
    """
    Proximal Policy Optimization configuration.

    Args:
        ppo_epochs: Number of PPO epochs per batch
        ppo_batch_size: PPO batch size
        ref_model_checkpoint: Reference model checkpoint path
        value_model_checkpoint: Value model checkpoint path
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
    ppo_batch_size: int = 4
    ref_model_checkpoint: Optional[str] = None
    value_model_checkpoint: Optional[str] = None
    value_optim_config: Optional[OptimConfig] = None
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
    gen_suppress_tokens: Optional[List[int]] = None


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
    gen_suppress_tokens: Optional[List[int]] = None


@dataclass(kw_only=True)
class TrainConfig:
    """
    Main training configuration.

    This is the top-level configuration that combines all other configs.

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
    init_state_dict: Optional[Any] = None
    file_dataset: Optional[Any] = None
    dataset_block_size: int = 2048
    data_loader_config: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    loss_config: LossConfig = field(default_factory=LossConfig)
    optim_config: OptimConfig = field(default_factory=OptimConfig)
    eval_config: EvalConfig = field(default_factory=EvalConfig)
    ds_config: Optional[Any] = None

    # Stage-specific configs
    pretrain_config: Optional[PretrainConfig] = field(default_factory=PretrainConfig)
    sft_config: Optional[SFTConfig] = None
    dpo_config: Optional[DPOConfig] = None
    ppo_config: Optional[PPOConfig] = None
    grpo_config: Optional[GRPOConfig] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1, got {self.n_epochs}")
