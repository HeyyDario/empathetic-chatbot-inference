# src/utils.py
import os
import time
import pandas as pd
import torch
import random
import numpy as np
from datasets import load_dataset
from config import SEED

def set_seed(seed=SEED):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_empathetic_data(num_samples=50):
    """
    Loads samples from the EmpatheticDialogues validation set.
    Formatted specifically for Llama-3 instruction tuning.
    """
    print(f"=== Loading {num_samples} samples from EmpatheticDialogues... ===")
    
    # Load dataset
    dataset = load_dataset("empathetic_dialogues", split="validation", trust_remote_code=True)
    
    # Select a subset
    # We use a fixed seed shuffle to ensure we get the SAME 50 prompts every time
    subset = dataset.shuffle(seed=SEED).select(range(num_samples))
    
    prompts = []
    references = []
    contexts = []
    
    for row in subset:
        # Llama-3 specific chat template formatting could be applied here
        # For simplicity, we use a clear structured prompt
        text = (
            f"Instruction: You are an empathetic listener. Respond supportively to the user.\n"
            f"Context: {row['context']}\n"
            f"User: {row['prompt']}\n"
            f"Listener:"
        )
        
        prompts.append(text)
        references.append(row['utterance'])
        contexts.append(row['context'])
        
    return prompts, references, contexts

def calculate_metrics(start_time, end_time, total_generated_tokens, num_samples):
    """
    Calculates throughput and latency.
    """
    total_duration = end_time - start_time
    
    throughput = total_generated_tokens / total_duration if total_duration > 0 else 0
    avg_latency = (total_duration * 1000) / num_samples if num_samples > 0 else 0
    
    return {
        "duration_sec": total_duration,
        "throughput_tok_sec": throughput,
        "latency_ms": avg_latency,
        "total_tokens": total_generated_tokens
    }

def reset_memory_stats():
    """Resets the CUDA memory peak tracker."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

def get_peak_memory_gb():
    """Returns the peak GPU memory used in GB."""
    if torch.cuda.is_available():
        # max_memory_allocated returns bytes
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0

def save_results(results_list, references, filename):
    """
    Saves the inference results to a CSV file.
    Expects results_list to be a list of dicts: {'prompt_id', 'generated', 'latency_ms'}
    """
    # Convert list of dicts to DataFrame
    df = pd.DataFrame(results_list)
    
    # Add references for quality evaluation later
    # Ensure lengths match (basic safety check)
    if len(references) == len(df):
        df['reference'] = references
    else:
        print(f"Warning: Reference count ({len(references)}) matches result count ({len(df)}) mismatch.")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    df.to_csv(filename, index=False)
    print(f"Results saved successfully to: {filename}")