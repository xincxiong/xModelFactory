"""
Base trainer class for all training types.
"""

from typing import Optional, Tuple, List, Dict, Any
import copy
import gc

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from xmodel_factory.model_core import LlmModel

from .parallel import DsParallel
from .tools import TrainerTools
from .train_configs import TrainConfig, OptimConfig, DsZero2Config, DsZero3Config, KDConfig


class BaseTrainer:
    """Base class for all trainers."""

    def __init__(
        self,
        *,
        train_config: TrainConfig,
        eval_prompts: List[str],
        kd_config: Optional[KDConfig] = None,
        gradient_accumulation_steps: int = 1
    ):
        """Initialize base trainer."""
        self.train_config = train_config
        self.eval_prompts = eval_prompts
        self.eval_idx = -1

        self.resume_epoch = 0
        self.resume_file_idx = 0
        self.resume_batch_idx = 0

        self.kd_config = kd_config
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Initialize tools
        self.tools = TrainerTools()

        # Initialize model and optimizer
        initial_lr = train_config.optim_config.initial_lr
        self.train_model, self.optimizer = self._init_train_model_and_optim(initial_lr)

        print(f"[BaseTrainer] Initialized on device: {self.tools.parallel.device}")

    def _new_model(self, train_config: TrainConfig):
        """Create a new model instance."""
        return LlmModel(train_config.model_config)

    def _init_train_model_and_optim(self, initial_lr: float):
        """Initialize training model and optimizer."""
        model = self._new_model(self.train_config)

        # Load initial state dict if provided
        if self.train_config.init_state_dict:
            model.load_state_dict(self.train_config.init_state_dict, strict=False)
            self.train_config.init_state_dict = None

        # Log model info on main process
        if self.tools.parallel.is_main_process:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[BaseTrainer] Total parameters: {total_params:,}")
            print(f"[BaseTrainer] Trainable parameters: {trainable_params:,}")

        # Configure optimizer
        optimizer = self._config_optim(model, initial_lr)

        # Process with parallel backend
        parallel_kwargs = self._get_parallel_kwargs()
        model, optim = self.tools.parallel.process(
            model=model,
            optimizer=optimizer,
            kwargs=parallel_kwargs
        )

        return model, optim

    def _config_optim(self, model, initial_lr):
        """Configure optimizer."""
        optim_type = self.train_config.optim_config.optim_type

        # Default betas and weight decay
        if optim_type == 'lion':
            betas = (0.95, 0.98)
            weight_decay = 0.015
        else:
            betas = (0.9, 0.999)
            weight_decay = 0.01

        # Separate parameters with and without weight decay
        no_decay = ["bias", "norm.weight"]
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in no_decay):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Create optimizer
        if optim_type == 'lion':
            try:
                from lion_pytorch import Lion
                optimizer = Lion(optimizer_grouped_parameters, lr=initial_lr, betas=betas)
            except ImportError:
                print("[BaseTrainer] Lion optimizer not available, using AdamW")
                optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=initial_lr, betas=betas)
        else:
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=initial_lr, betas=betas)

        return optimizer

    def _get_parallel_kwargs(self) -> Optional[dict]:
        """Get DeepSpeed configuration kwargs."""
        if not isinstance(self.tools.parallel, DsParallel):
            return None

        if not self.train_config.ds_config:
            return None

        ds_config = self.train_config.ds_config
        parallel_kwargs = {
            'gradient_accumulation_steps': 1,
            'gradient_clipping': ds_config.gradient_clipping,
            'train_micro_batch_size_per_gpu': self.train_config.batch_size
        }

        # Add ZeRO configuration
        if ds_config.zero_config:
            zero_config = ds_config.zero_config
            zero_optimization = {'stage': zero_config.stage}

            # Add optional ZeRO parameters
            if hasattr(zero_config, 'offload_optimizer') and zero_config.offload_optimizer:
                zero_optimization['offload_optimizer'] = {
                    "device": zero_config.offload_optimizer.device,
                    "pin_memory": zero_config.offload_optimizer.pin_memory
                }

            if hasattr(zero_config, 'offload_param') and zero_config.offload_param:
                zero_optimization['offload_param'] = {
                    "device": zero_config.offload_param.device,
                    "pin_memory": zero_config.offload_param.pin_memory
                }

            parallel_kwargs['zero_optimization'] = zero_optimization

        # Add FP16/BF16 configuration
        if ds_config.bf16_config and ds_config.bf16_config.enabled:
            parallel_kwargs['bf16'] = {'enabled': True}
        elif ds_config.fp16_config:
            parallel_kwargs['fp16'] = {'enabled': True}

        return parallel_kwargs

    def _create_dataset(self, file_idx) -> Tuple[Dataset, str]:
        """Create dataset for training. To be implemented by subclasses."""
        raise NotImplementedError

    def train(self):
        """Main training loop."""
        gradient_accumulation_steps = max(1, self.gradient_accumulation_steps)

        for epoch in range(self.resume_epoch, self.train_config.n_epochs):
            self.train_model.train()
            file_count = len(self.train_config.file_dataset) if self.train_config.file_dataset else 1

            for file_idx in range(file_count):
                dataset, file_path = self._create_dataset(file_idx)

                # Create dataloader
                data_loader = self.tools.parallel.process_dataloader(
                    dataset=dataset,
                    data_loader_kwargs={
                        'batch_size': self.train_config.batch_size,
                        'shuffle': self.train_config.data_loader_config.data_loader_shuffle,
                        'num_workers': self.train_config.data_loader_config.data_loader_num_workers,
                        'pin_memory': self.train_config.data_loader_config.data_loader_pin_memory,
                    }
                )

                self.tools.parallel.on_epoch_start(epoch)

                # Training loop
                for batch_idx, batch_data in enumerate(data_loader):
                    try:
                        inputs = batch_data['inputs'].to(self.tools.parallel.device)
                        labels = batch_data['labels'].to(self.tools.parallel.device)

                        # Forward pass
                        with torch.cuda.amp.autocast(enabled=self.tools.use_amp):
                            outputs = self.train_model(inputs)
                            logits = outputs['logits']

                            # Simple loss calculation (placeholder)
                            loss = torch.nn.functional.cross_entropy(
                                logits.view(-1, logits.size(-1)),
                                labels.view(-1),
                                ignore_index=-100
                            )

                        # Backward pass
                        if gradient_accumulation_steps > 1:
                            loss = loss / gradient_accumulation_steps

                        if isinstance(self.tools.parallel, DsParallel):
                            self.train_model.backward(loss)
                        else:
                            loss.backward()

                        # Optimizer step
                        if (batch_idx + 1) % gradient_accumulation_steps == 0:
                            if isinstance(self.tools.parallel, DsParallel):
                                self.train_model.step()
                            else:
                                self.optimizer.step()
                                self.optimizer.zero_grad()

                        # Log progress
                        if self.tools.parallel.is_main_process and batch_idx % 10 == 0:
                            print(f"[Train] Epoch {epoch}, File {file_idx}, "
                                  f"Batch {batch_idx}, Loss: {loss.item():.4f}")

                    except Exception as e:
                        print(f"[BaseTrainer] Error in training: {e}")
                        raise

                # Cleanup
                del data_loader, dataset
                gc.collect()
                torch.cuda.empty_cache()

            # End of epoch
            self.tools.parallel.on_epoch_end(epoch)

        self.tools.parallel.destroy()
