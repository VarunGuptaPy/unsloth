#!/usr/bin/env python3
"""
Multi-GPU Training Example with Unsloth

This script demonstrates how to use Unsloth for distributed training across multiple GPUs.

Usage:
    # Single GPU (traditional)
    python multi_gpu_training.py
    
    # Multi-GPU with torchrun (recommended)
    torchrun --nproc_per_node=2 multi_gpu_training.py
    
    # Multi-GPU with accelerate
    accelerate launch --num_processes=2 multi_gpu_training.py

Requirements:
    - Multiple GPUs available
    - PyTorch with CUDA support
    - Unsloth with multi-GPU support
"""

import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth.trainer import UnslothTrainingArguments, UnslothTrainer
from unsloth.distributed_utils import (
    is_distributed_available,
    is_distributed_initialized,
    get_world_size,
    get_rank,
    is_main_process,
    print_distributed_info,
    setup_distributed_training,
)

def main():
    # Print distributed training information
    if is_main_process():
        print("=" * 60)
        print("Unsloth Multi-GPU Training Example")
        print("=" * 60)
        print_distributed_info()
    
    # Model configuration
    model_name = "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = 2048
    dtype = None  # Auto-detect
    load_in_4bit = True
    
    # Multi-GPU device mapping
    # Use "auto" for automatic multi-GPU distribution
    # Use "sequential" for sequential GPU placement (legacy)
    device_map = "auto" if is_distributed_available() else "sequential"
    
    if is_main_process():
        print(f"Loading model: {model_name}")
        print(f"Device map: {device_map}")
        print(f"Max sequence length: {max_seq_length}")
        print(f"Load in 4bit: {load_in_4bit}")
    
    # Load model and tokenizer with multi-GPU support
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        device_map=device_map,  # This enables multi-GPU support
        trust_remote_code=False,
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",     # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None, # And LoftQ
    )
    
    if is_main_process():
        print("Model loaded successfully!")
        print(f"Model device: {next(model.parameters()).device}")
        if hasattr(model, 'hf_device_map'):
            print(f"Device map: {model.hf_device_map}")
    
    # Load and prepare dataset
    dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")  # Small subset for demo
    
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            text = f"### Instruction:\n{instruction}\n"
            if input_text:
                text += f"### Input:\n{input_text}\n"
            text += f"### Response:\n{output}"
            texts.append(text)
        return {"text": texts}
    
    dataset = dataset.map(formatting_prompts_func, batched=True)
    
    # Training arguments with multi-GPU support
    training_args = UnslothTrainingArguments(
        # Basic training parameters
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,  # Short training for demo
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="./multi_gpu_outputs",
        
        # Multi-GPU specific settings
        enable_distributed_training=True,  # Enable distributed training
        distributed_backend="nccl",        # Use NCCL for NVIDIA GPUs
        ddp_find_unused_parameters=False,  # Better performance
        ddp_bucket_cap_mb=25,             # Default bucket size
        ddp_broadcast_buffers=False,       # Better performance
        dataloader_pin_memory=True,        # Better data loading performance
        auto_find_batch_size=False,        # Disable for distributed training
        gradient_checkpointing=True,       # Save memory
        
        # Logging and saving (only from main process)
        report_to="none",  # Disable wandb/tensorboard for demo
        save_strategy="steps",
        save_steps=30,
        save_only_model=True,
        logging_first_step=True,
        
        # Evaluation settings
        eval_strategy="no",  # Disable evaluation for demo
        
        # Data loading
        dataloader_num_workers=0,  # Use 0 for distributed training
        remove_unused_columns=False,
    )
    
    # Create trainer with multi-GPU support
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences
    )
    
    if is_main_process():
        print("Starting training...")
        print(f"Training on {get_world_size()} GPU(s)")
        print(f"Total steps: {training_args.max_steps}")
        print(f"Batch size per device: {training_args.per_device_train_batch_size}")
        print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
        print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * get_world_size()}")
    
    # Start training
    trainer.train()
    
    if is_main_process():
        print("Training completed!")
        
        # Save the final model (only from main process)
        print("Saving model...")
        trainer.save_model()
        
        # Optional: Save to Hugging Face Hub
        # model.push_to_hub("your_username/your_model_name", token="your_token")
        
        print("Model saved successfully!")
        print(f"Model saved to: {training_args.output_dir}")

def setup_environment():
    """Setup environment variables for optimal multi-GPU training."""
    # Optimize CUDA memory allocation
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", 
                         "expandable_segments:True,roundup_power2_divisions:[32:256,64:128,256:64,>:32]")
    
    # Optimize distributed training
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_DEBUG", "INFO")  # Set to "WARN" for less verbose output
    
    # Optimize compilation
    os.environ.setdefault("UNSLOTH_COMPILE_MAXIMUM", "1")
    
    # For debugging (uncomment if needed)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if __name__ == "__main__":
    # Setup optimal environment
    setup_environment()
    
    # Run main training
    main()
    
    if is_main_process():
        print("\n" + "=" * 60)
        print("Multi-GPU Training Example Completed!")
        print("=" * 60)
        print("\nTo run this script with multiple GPUs:")
        print("  torchrun --nproc_per_node=2 multi_gpu_training.py")
        print("  torchrun --nproc_per_node=4 multi_gpu_training.py")
        print("\nOr with accelerate:")
        print("  accelerate launch --num_processes=2 multi_gpu_training.py")
        print("=" * 60)
