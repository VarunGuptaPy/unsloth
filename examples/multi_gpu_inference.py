#!/usr/bin/env python3
"""
Multi-GPU Inference Example with Unsloth

This script demonstrates how to use Unsloth for inference across multiple GPUs
without distributed training setup.

Usage:
    python multi_gpu_inference.py

Requirements:
    - Multiple GPUs available
    - PyTorch with CUDA support
    - Unsloth with multi-GPU support
"""

import torch
from unsloth import FastLanguageModel
from unsloth.distributed_utils import is_distributed_available

def main():
    print("=" * 60)
    print("Unsloth Multi-GPU Inference Example")
    print("=" * 60)
    
    # Check GPU availability
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    if num_gpus < 2:
        print("Warning: This example is designed for multi-GPU setups.")
        print("Running on single GPU...")
        device_map = "auto"
    else:
        print(f"Using {num_gpus} GPUs for inference")
        device_map = "auto"  # Automatically distribute across GPUs
    
    # Model configuration
    model_name = "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = 2048
    dtype = None  # Auto-detect
    load_in_4bit = True
    
    print(f"\nLoading model: {model_name}")
    print(f"Device map: {device_map}")
    print(f"Max sequence length: {max_seq_length}")
    print(f"Load in 4bit: {load_in_4bit}")
    
    # Load model and tokenizer with multi-GPU support
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        device_map=device_map,  # This enables multi-GPU distribution
        trust_remote_code=False,
    )
    
    print("\nModel loaded successfully!")
    
    # Print device information
    if hasattr(model, 'hf_device_map'):
        print(f"Model device map: {model.hf_device_map}")
    else:
        print(f"Model device: {next(model.parameters()).device}")
    
    # Enable native 2x faster inference
    FastLanguageModel.for_inference(model)
    
    # Prepare some example prompts
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about artificial intelligence.",
        "What are the benefits of renewable energy?",
        "How does machine learning work?",
    ]
    
    print(f"\nRunning inference on {len(prompts)} prompts...")
    print("=" * 60)
    
    # Run inference on multiple prompts
    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}: {prompt}")
        print("-" * 40)
        
        # Format the prompt
        formatted_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
"""
        
        # Tokenize the input
        inputs = tokenizer(
            [formatted_prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length
        )
        
        # Move inputs to the appropriate device
        # For multi-GPU models, inputs should go to the first device
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            # Find the first device in the device map
            first_device = next(iter(model.hf_device_map.values()))
            if isinstance(first_device, int):
                first_device = f"cuda:{first_device}"
            inputs = {k: v.to(first_device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after "### Response:")
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        print(f"Response: {response}")
    
    print("\n" + "=" * 60)
    print("Multi-GPU Inference Example Completed!")
    print("=" * 60)
    
    # Memory usage information
    if torch.cuda.is_available():
        print("\nGPU Memory Usage:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            cached = torch.cuda.memory_reserved(i) / 1024**3      # GB
            print(f"  GPU {i}: {allocated:.2f} GB allocated, {cached:.2f} GB cached")

def benchmark_inference():
    """Optional: Benchmark inference speed across different configurations."""
    import time
    
    print("\n" + "=" * 60)
    print("Benchmarking Multi-GPU Inference")
    print("=" * 60)
    
    # Simple benchmark prompt
    prompt = "Explain the concept of artificial intelligence in one paragraph."
    formatted_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
"""
    
    # Load model for benchmarking
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-1B-Instruct",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        device_map="auto",
    )
    
    FastLanguageModel.for_inference(model)
    
    # Prepare inputs
    inputs = tokenizer([formatted_prompt], return_tensors="pt")
    
    # Move to appropriate device
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        first_device = next(iter(model.hf_device_map.values()))
        if isinstance(first_device, int):
            first_device = f"cuda:{first_device}"
        inputs = {k: v.to(first_device) for k, v in inputs.items()}
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    
    # Benchmark
    num_runs = 10
    print(f"Running {num_runs} inference iterations...")
    
    start_time = time.time()
    for i in range(num_runs):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    print(f"Average inference time: {avg_time:.3f} seconds")
    print(f"Tokens per second: {100 / avg_time:.1f}")

if __name__ == "__main__":
    main()
    
    # Uncomment to run benchmark
    # benchmark_inference()
