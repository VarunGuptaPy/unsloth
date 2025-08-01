# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from dataclasses import dataclass, field
from typing import Optional
from functools import wraps

import trl
import inspect
from trl import SFTTrainer
from . import is_bfloat16_supported
from unsloth_zoo.training_utils import (
    unsloth_train as _unsloth_train,
)
from unsloth_zoo.vision_utils import (
    UnslothVisionDataCollator,
)
from packaging.version import Version
import dataclasses
from .distributed_utils import (
    is_distributed_available,
    is_distributed_initialized,
    get_world_size,
    get_rank,
    is_main_process,
    setup_distributed_training,
    print_distributed_info,
    validate_distributed_environment,
)

__all__ = [
    "UnslothTrainingArguments",
    "UnslothTrainer",
    "unsloth_train",
    "_patch_trl_trainer",
    "UnslothVisionDataCollator",
]

# Unsloth gradient accumulation fix:
from transformers import __version__ as transformers_version
if Version(transformers_version) > Version("4.45.2"):
    def unsloth_train(trainer, *args, **kwargs):
        return trainer.train(*args, **kwargs)
    pass
else:
    def unsloth_train(trainer, *args, **kwargs):
        if len(args) != 0 or len(kwargs) != 0:
            raise RuntimeError(
                "Unsloth: Our custom gradient accumulation fixed trainer does not support other arguments.\n"\
                "If you want to use our fix inside of HF, please update `transformers` to the latest version via:\n"\
                '`pip uninstall transformers -y && pip install --upgrade --no-cache-dir transformers`'
            )
        print(
            "Unsloth: Using our custom gradient accumulation fixed trainer, which is not feature complete.\n"\
            "If you want to use our fix inside of HF, please update `transformers` to the latest version via:\n"\
            '`pip uninstall transformers -y && pip install --upgrade --no-cache-dir transformers`'
        )
        return _unsloth_train(trainer)
    pass
pass

try:
    from trl import SFTConfig as TrainingArguments
except:
    from transformers import TrainingArguments
pass

class UnslothTrainingArguments(TrainingArguments):
    def __init__(
        self,
        embedding_learning_rate: float = None,
        # Multi-GPU specific arguments
        ddp_backend: str = None,
        ddp_find_unused_parameters: bool = None,
        ddp_bucket_cap_mb: int = None,
        ddp_broadcast_buffers: bool = None,
        dataloader_pin_memory: bool = True,
        auto_find_batch_size: bool = False,
        gradient_checkpointing: bool = True,
        # Distributed training setup
        enable_distributed_training: bool = None,
        distributed_backend: str = "nccl",
        *args, **kwargs
    ):
        # Set up distributed training if multiple GPUs are available
        if enable_distributed_training is None:
            enable_distributed_training = is_distributed_available()

        # Configure distributed training arguments
        if enable_distributed_training and is_distributed_available():
            # Set default DDP parameters for optimal performance
            if ddp_backend is None:
                ddp_backend = distributed_backend
            if ddp_find_unused_parameters is None:
                ddp_find_unused_parameters = False  # Better performance for most cases
            if ddp_bucket_cap_mb is None:
                ddp_bucket_cap_mb = 25  # Default bucket size
            if ddp_broadcast_buffers is None:
                ddp_broadcast_buffers = False  # Better performance

            # Update kwargs with distributed settings
            kwargs.update({
                "ddp_backend": ddp_backend,
                "ddp_find_unused_parameters": ddp_find_unused_parameters,
                "ddp_bucket_cap_mb": ddp_bucket_cap_mb,
                "ddp_broadcast_buffers": ddp_broadcast_buffers,
                "dataloader_pin_memory": dataloader_pin_memory,
                "auto_find_batch_size": auto_find_batch_size,
                "gradient_checkpointing": gradient_checkpointing,
            })

            # Print distributed info if main process
            if is_main_process():
                print("Unsloth: Configuring for distributed training")
                print_distributed_info()

        self.embedding_learning_rate = embedding_learning_rate
        self.enable_distributed_training = enable_distributed_training
        self.distributed_backend = distributed_backend

        super().__init__(*args, **kwargs)
pass


def _create_unsloth_optimizer(
    model,
    optimizer_cls,
    optimizer_kwargs,
    embedding_lr = 5e-5,
):
    lr = optimizer_kwargs["lr"]
    weight_decay = optimizer_kwargs.get("weight_decay", 0.0)

    param_groups = \
    {
        "non_embeddings" : {},
        "embeddings"     : {},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if name.endswith("modules_to_save.default.weight"):
            partial_name = name[:-len(".modules_to_save.default.weight")]
            partial_name = partial_name[partial_name.rfind(".")+1:]
            print(f"Unsloth: Setting lr = {embedding_lr:.2e} instead of {lr:.2e} for {partial_name}.")
            param_groups["embeddings"]    [name] = param
        else:
            param_groups["non_embeddings"][name] = param
        pass
    pass

    optimizer_grouped_parameters = [
        {
            "params"       : list(param_groups["non_embeddings"].values()),
            "weight_decay" : weight_decay,
            "lr"           : lr,
        },
        {
            "params"       : list(param_groups["embeddings"].values()),
            "weight_decay" : weight_decay,
            "lr"           : embedding_lr,
        },
    ]
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer
pass


class UnslothTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        # Setup distributed training if enabled
        if hasattr(kwargs.get('args'), 'enable_distributed_training') and \
           kwargs.get('args').enable_distributed_training:
            self._setup_distributed_training(kwargs.get('args'))

        super().__init__(*args, **kwargs)

        # Store distributed training info
        self.is_distributed = is_distributed_initialized()
        self.world_size = get_world_size()
        self.rank = get_rank()

        if self.is_distributed and is_main_process():
            print(f"Unsloth: Initialized trainer for distributed training with {self.world_size} GPUs")

    def _setup_distributed_training(self, args):
        """Setup distributed training environment."""
        if not validate_distributed_environment():
            return

        distributed_info = setup_distributed_training(
            backend=getattr(args, 'distributed_backend', 'nccl'),
            timeout_minutes=30,
        )

        if is_main_process():
            print("Unsloth: Distributed training setup complete")

    def create_optimizer(self):
        embedding_learning_rate = getattr(self.args, "embedding_learning_rate", None)
        if embedding_learning_rate is None:
            return super().create_optimizer()

        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = SFTTrainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = _create_unsloth_optimizer(
                self.model,
                optimizer_cls,
                optimizer_kwargs,
                embedding_learning_rate,
            )
        return self.optimizer

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Enhanced training step with multi-GPU support.

        Args:
            model: The model to train
            inputs: The input batch
            num_items_in_batch: Number of items in the batch (for gradient accumulation)
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Use accelerator for distributed training
        if hasattr(self, 'accelerator') and self.accelerator is not None:
            with self.accelerator.accumulate(model):
                loss = self.compute_loss(model, inputs)
                self.accelerator.backward(loss)
                return loss.detach()
        else:
            # Fallback to standard training step - pass num_items_in_batch if supported
            try:
                return super().training_step(model, inputs, num_items_in_batch)
            except TypeError:
                # Fallback for older versions that don't support num_items_in_batch
                return super().training_step(model, inputs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Enhanced loss computation with proper scaling for multi-GPU.

        Args:
            model: The model
            inputs: The input batch
            return_outputs: Whether to return outputs
            num_items_in_batch: Number of items in the batch (for gradient accumulation)
        """
        # Try to call parent with num_items_in_batch if supported
        try:
            loss = super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        except TypeError:
            # Fallback for older versions that don't support num_items_in_batch
            loss = super().compute_loss(model, inputs, return_outputs=return_outputs)

        # For multi-GPU training, ensure proper loss scaling
        if self.is_distributed and self.world_size > 1:
            # Loss is already averaged across devices by DDP
            # No additional scaling needed
            pass

        return loss

    def log(self, logs):
        """
        Enhanced logging for multi-GPU training.
        """
        # Only log from main process in distributed training
        if not self.is_distributed or is_main_process():
            # Add distributed training info to logs
            if self.is_distributed:
                logs["world_size"] = self.world_size
                logs["rank"] = self.rank
            super().log(logs)

    def save_model(self, output_dir=None, _internal_call=False):
        """
        Enhanced model saving for multi-GPU training.
        """
        # Only save from main process in distributed training
        if not self.is_distributed or is_main_process():
            super().save_model(output_dir, _internal_call)

        # Wait for all processes to complete saving
        if self.is_distributed:
            import torch.distributed as dist
            dist.barrier()

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Enhanced evaluation for multi-GPU training.
        """
        # Run evaluation on all processes but only log from main process
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        if self.is_distributed:
            # Synchronize evaluation results across processes
            import torch.distributed as dist
            dist.barrier()

            # Only return results from main process
            if not is_main_process():
                return {}

        return eval_results

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        """
        Enhanced logging/saving/evaluation with multi-GPU coordination.
        """
        # Only perform logging/saving from main process
        if not self.is_distributed or is_main_process():
            super()._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

        # Synchronize all processes
        if self.is_distributed:
            import torch.distributed as dist
            dist.barrier()
pass

# From `trl>=0.13.0`, they changed how to pass several params to the trainer
# We need to patch to make the transition smooth
def _backwards_compatible_trainer(trainer_class, config_class):
    original_init = trainer_class.__init__
    
    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        # All Trainer tokenizer are now called processing_class
        trainer_params = set(inspect.signature(original_init).parameters.keys())

        if "processing_class" in trainer_params and "tokenizer" in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        pass

        if ("args" in kwargs) and (Version(trl.__version__) >= Version("0.13.0.dev0")):
            training_args = kwargs.pop("args", None)

            # Get parameters that Trainer.__init__ actually expects
            trainer_params.remove('self')
            trainer_params.remove('args')

            # Get fields that should be passed to Config init
            config_fields = {
                field.name: field for field in dataclasses.fields(config_class) 
                if field.init
            }
            
            # Create config dict with valid fields from training_args
            config_dict = {
                name: getattr(training_args, name)
                for name in config_fields
                if hasattr(training_args, name)
            }

            # Get parameters that exist in Config but not in TrainingArguments
            from transformers import TrainingArguments
            moved_params = \
                set(inspect.signature(config_class)     .parameters.keys()) - \
                set(inspect.signature(TrainingArguments).parameters.keys())
            
            # Separate kwargs into trainer kwargs and config kwargs
            trainer_kwargs = {}
            additional_config_kwargs = {}

            for key, value in kwargs.items():
                if key in trainer_params: trainer_kwargs[key] = value
                elif key in moved_params or key in config_fields:
                    additional_config_kwargs[key] = value
                else:
                    additional_config_kwargs[key] = value
                pass
            pass

            # Update config_dict with additional kwargs
            config_dict.update(additional_config_kwargs)

            # Create Config with all the collected parameters
            # Reinitialising config class with parameters (that were none initially but populated on first init)
            # causes the 2nd init to fail as there are mutual exclusive checks on pairs of parameters.
            # Refer: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py#L499-L502 for example
            # So we only create config class if the previous init was not TrainingArguments
            if not isinstance(training_args, TrainingArguments):
                config = config_class(**config_dict)
            else:
                config = training_args

            # Reconstruct kwargs for Trainer
            kwargs = trainer_kwargs
            kwargs["args"] = config
        pass
        original_init(self, *args, **kwargs)
    pass
    return new_init
pass


def _patch_trl_trainer():
    import trl
    if hasattr(trl, "__UNSLOTH_BACKWARDS_COMPATIBLE__"): return
    if Version(trl.__version__) <= Version("0.11.0"): return

    import trl.trainer
    trl_classes = dir(trl.trainer)
    trl_trainers = set(x[:-len("Trainer")] for x in trl_classes if x.endswith("Trainer"))
    trl_configs  = set(x[:-len("Config")]  for x in trl_classes if x.endswith("Config"))
    trl_classes = list(trl_trainers & trl_configs)

    for x in trl_classes:
        try:    exec(f"trl.{x}Trainer.__init__ = _backwards_compatible_trainer(trl.{x}Trainer, trl.{x}Config)", globals())
        except: continue
    pass

    trl.__UNSLOTH_BACKWARDS_COMPATIBLE__ = True
pass
