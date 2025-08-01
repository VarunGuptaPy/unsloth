# Multi-GPU Training with Unsloth

Unsloth now supports multi-GPU training and inference! This guide will help you get started with distributed training across multiple GPUs.

## 🚀 Quick Start

### Basic Multi-GPU Training

```python
from unsloth import FastLanguageModel
from unsloth.trainer import UnslothTrainingArguments, UnslothTrainer

# Load model with automatic multi-GPU distribution
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    device_map="auto",  # This enables multi-GPU support!
)

# Configure training for multi-GPU
training_args = UnslothTrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    enable_distributed_training=True,  # Enable distributed training
    distributed_backend="nccl",        # Use NCCL for NVIDIA GPUs
    output_dir="./outputs",
    # ... other training arguments
)

# Create trainer
trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

# Start training
trainer.train()
```

### Launch Commands

#### Using torchrun (Recommended)
```bash
# 2 GPUs
torchrun --nproc_per_node=2 your_training_script.py

# 4 GPUs
torchrun --nproc_per_node=4 your_training_script.py

# Multiple nodes (advanced)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" your_training_script.py
```

#### Using accelerate
```bash
# Configure accelerate (run once)
accelerate config

# Launch training
accelerate launch your_training_script.py
```

## 📋 Requirements

- **Multiple GPUs**: At least 2 NVIDIA GPUs
- **PyTorch**: Version 1.12+ with CUDA support
- **NCCL**: For optimal GPU communication (usually included with PyTorch)
- **Sufficient VRAM**: Model size should fit across your GPUs

## ⚙️ Configuration Options

### Device Mapping

```python
# Automatic distribution (recommended)
device_map = "auto"

# Balanced distribution
device_map = "balanced"

# Sequential distribution (legacy)
device_map = "sequential"

# Custom distribution
device_map = {
    "model.embed_tokens": 0,
    "model.layers.0": 0,
    "model.layers.1": 1,
    # ... custom mapping
}
```

### Training Arguments

```python
training_args = UnslothTrainingArguments(
    # Basic settings
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    
    # Multi-GPU specific
    enable_distributed_training=True,
    distributed_backend="nccl",           # "nccl" for NVIDIA, "gloo" for CPU
    ddp_find_unused_parameters=False,     # Better performance
    ddp_bucket_cap_mb=25,                # Communication bucket size
    ddp_broadcast_buffers=False,          # Better performance
    dataloader_pin_memory=True,           # Faster data loading
    
    # Memory optimization
    gradient_checkpointing=True,
    auto_find_batch_size=False,           # Disable for distributed training
    dataloader_num_workers=0,             # Use 0 for distributed training
)
```

## 🔧 Environment Variables

### Optimization
```bash
# Optimize CUDA memory allocation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,roundup_power2_divisions:[32:256,64:128,256:64,>:32]"

# Enable maximum compilation optimizations
export UNSLOTH_COMPILE_MAXIMUM=1

# NCCL optimization
export NCCL_ASYNC_ERROR_HANDLING=1
```

### Debugging
```bash
# Enable NCCL debugging
export NCCL_DEBUG=INFO

# Enable CUDA debugging (slows down training)
export CUDA_LAUNCH_BLOCKING=1

# Force single GPU mode (if needed)
export UNSLOTH_FORCE_SINGLE_GPU=1
```

## 📊 Performance Tips

### Batch Size Scaling
When using multiple GPUs, scale your batch size accordingly:
```python
# Single GPU: batch_size = 4
# 2 GPUs: per_device_train_batch_size = 2 (effective batch_size = 4)
# 4 GPUs: per_device_train_batch_size = 1 (effective batch_size = 4)

effective_batch_size = per_device_train_batch_size * num_gpus * gradient_accumulation_steps
```

### Memory Optimization
```python
# Enable gradient checkpointing to save memory
training_args.gradient_checkpointing = True

# Use 4-bit quantization
load_in_4bit = True

# Optimize LoRA rank
r = 16  # Lower rank = less memory, but potentially lower quality
```

### Communication Optimization
```python
# Disable unused parameter detection for better performance
ddp_find_unused_parameters = False

# Optimize bucket size for your model
ddp_bucket_cap_mb = 25  # Adjust based on model size

# Disable buffer broadcasting if not needed
ddp_broadcast_buffers = False
```

## 🐛 Troubleshooting

### Common Issues

#### "NCCL initialization failed"
```bash
# Check GPU visibility
nvidia-smi

# Ensure all GPUs are accessible
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Try different NCCL backend
export NCCL_SOCKET_IFNAME=eth0  # Replace with your network interface
```

#### "Out of memory" errors
```python
# Reduce batch size per device
per_device_train_batch_size = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Use smaller model or higher quantization
load_in_4bit = True
```

#### "Process group not initialized"
```bash
# Make sure to use proper launch command
torchrun --nproc_per_node=2 your_script.py

# Or set environment variables manually
export RANK=0
export WORLD_SIZE=2
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=12355
```

### Debugging Commands

```python
from unsloth.distributed_utils import print_distributed_info

# Print detailed distributed training information
print_distributed_info()

# Check if distributed training is available
from unsloth.distributed_utils import is_distributed_available
print(f"Distributed available: {is_distributed_available()}")

# Check current setup
from unsloth.distributed_utils import get_world_size, get_rank
print(f"World size: {get_world_size()}, Rank: {get_rank()}")
```

## 📈 Performance Benchmarks

Expected speedup with multi-GPU training:

| GPUs | Speedup | Notes |
|------|---------|-------|
| 1    | 1.0x    | Baseline |
| 2    | 1.8x    | ~90% efficiency |
| 4    | 3.4x    | ~85% efficiency |
| 8    | 6.4x    | ~80% efficiency |

*Actual speedup depends on model size, batch size, and hardware configuration.*

## 🔗 Examples

- [Multi-GPU Training Example](../examples/multi_gpu_training.py)
- [Multi-GPU Inference Example](../examples/multi_gpu_inference.py)

## 💡 Best Practices

1. **Use `device_map="auto"`** for automatic GPU distribution
2. **Scale batch size** appropriately for your GPU count
3. **Enable gradient checkpointing** to save memory
4. **Use NCCL backend** for NVIDIA GPUs
5. **Monitor GPU utilization** with `nvidia-smi`
6. **Start with 2 GPUs** before scaling to more
7. **Use torchrun** for launching distributed training
8. **Set `ddp_find_unused_parameters=False`** for better performance

## 🆘 Getting Help

If you encounter issues with multi-GPU training:

1. Check the [troubleshooting section](#troubleshooting) above
2. Verify your environment with the debugging commands
3. Try the provided example scripts
4. Open an issue on GitHub with your configuration details

Happy multi-GPU training! 🚀
