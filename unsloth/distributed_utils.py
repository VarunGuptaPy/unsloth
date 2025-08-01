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

import os
import torch
import torch.distributed as dist
from typing import Optional, Dict, Any
import warnings

__all__ = [
    "is_distributed_available",
    "is_distributed_initialized", 
    "get_world_size",
    "get_rank",
    "get_local_rank",
    "setup_distributed_training",
    "cleanup_distributed_training",
    "is_main_process",
    "wait_for_everyone",
    "get_device_map_for_distributed",
]


def is_distributed_available() -> bool:
    """Check if distributed training is available."""
    return dist.is_available() and torch.cuda.device_count() > 1


def is_distributed_initialized() -> bool:
    """Check if distributed training is initialized."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Get the total number of processes in distributed training."""
    if is_distributed_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get the rank of the current process."""
    if is_distributed_initialized():
        return dist.get_rank()
    return 0


def get_local_rank() -> int:
    """Get the local rank of the current process."""
    if is_distributed_initialized():
        return int(os.environ.get("LOCAL_RANK", 0))
    return 0


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def wait_for_everyone():
    """Wait for all processes to reach this point."""
    if is_distributed_initialized():
        dist.barrier()


def setup_distributed_training(
    backend: str = "nccl",
    timeout_minutes: int = 30,
    init_method: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Setup distributed training environment.
    
    Args:
        backend: Backend to use for distributed training ('nccl', 'gloo', 'mpi')
        timeout_minutes: Timeout for distributed operations in minutes
        init_method: Initialization method for process group
        
    Returns:
        Dictionary with distributed training information
    """
    if not is_distributed_available():
        warnings.warn("Distributed training not available. Running in single GPU mode.")
        return {"distributed": False, "world_size": 1, "rank": 0, "local_rank": 0}
    
    # Get distributed training environment variables
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Set CUDA device for this process
    torch.cuda.set_device(local_rank)
    
    # Initialize process group if not already initialized
    if not is_distributed_initialized():
        if init_method is None:
            init_method = "env://"
        
        timeout = torch.distributed.default_pg_timeout
        if timeout_minutes > 0:
            timeout = torch.timedelta(minutes=timeout_minutes)
            
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            timeout=timeout,
        )
        
        if is_main_process():
            print(f"Unsloth: Initialized distributed training with {world_size} GPUs")
            print(f"Unsloth: Using backend: {backend}")
    
    return {
        "distributed": True,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "backend": backend,
    }


def cleanup_distributed_training():
    """Cleanup distributed training environment."""
    if is_distributed_initialized():
        dist.destroy_process_group()
        if is_main_process():
            print("Unsloth: Cleaned up distributed training")


def get_device_map_for_distributed(
    model_name: str,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
) -> str:
    """
    Get appropriate device map for distributed training.
    
    Args:
        model_name: Name of the model
        world_size: Total number of processes
        rank: Current process rank
        
    Returns:
        Device map string for the model
    """
    if world_size is None:
        world_size = get_world_size()
    if rank is None:
        rank = get_rank()
        
    if world_size == 1:
        return "auto"
    
    # For distributed training, use local rank as device
    local_rank = get_local_rank()
    return {"": f"cuda:{local_rank}"}


def print_distributed_info():
    """Print information about the distributed training setup."""
    if is_main_process():
        print("=" * 50)
        print("Unsloth Distributed Training Information")
        print("=" * 50)
        print(f"Distributed Available: {is_distributed_available()}")
        print(f"Distributed Initialized: {is_distributed_initialized()}")
        print(f"World Size: {get_world_size()}")
        print(f"Current Rank: {get_rank()}")
        print(f"Local Rank: {get_local_rank()}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print("=" * 50)


def validate_distributed_environment():
    """Validate that the distributed environment is properly set up."""
    required_env_vars = ["RANK", "WORLD_SIZE", "LOCAL_RANK"]
    missing_vars = [var for var in required_env_vars if var not in os.environ]
    
    if missing_vars and get_world_size() > 1:
        warnings.warn(
            f"Missing environment variables for distributed training: {missing_vars}. "
            "Make sure to launch with torchrun or set these variables manually."
        )
        return False
    
    return True


def get_optimal_device_map(
    model_config: Dict[str, Any],
    available_gpus: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Get optimal device map based on model size and available GPUs.
    
    Args:
        model_config: Model configuration dictionary
        available_gpus: Number of available GPUs
        
    Returns:
        Optimal device map configuration
    """
    if available_gpus is None:
        available_gpus = torch.cuda.device_count()
    
    if available_gpus == 1:
        return "auto"
    
    # For multi-GPU, use balanced sharding
    return "balanced"


def sync_gradients_across_gpus(model):
    """Manually sync gradients across GPUs if needed."""
    if is_distributed_initialized() and get_world_size() > 1:
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= get_world_size()


def reduce_tensor_across_gpus(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """Reduce tensor across all GPUs."""
    if not is_distributed_initialized() or get_world_size() == 1:
        return tensor
    
    # Clone to avoid modifying original tensor
    reduced_tensor = tensor.clone()
    dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
    
    if average:
        reduced_tensor /= get_world_size()
    
    return reduced_tensor


def gather_tensor_across_gpus(tensor: torch.Tensor) -> torch.Tensor:
    """Gather tensor from all GPUs to rank 0."""
    if not is_distributed_initialized() or get_world_size() == 1:
        return tensor
    
    world_size = get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    
    if is_main_process():
        return torch.cat(gathered_tensors, dim=0)
    else:
        return tensor
