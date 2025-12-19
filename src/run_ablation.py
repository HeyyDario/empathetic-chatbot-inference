import sys
import os
import argparse
import pandas as pd
import gc
import torch
from utils import load_empathetic_data

# === Paths ===
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
sys.path.append(SRC_DIR)

# Define the Batch Sizes (Samples) to test
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]

def cleanup():
    """Forces memory cleanup between runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_ablation(mode, args):
    print(f"\nSTARTING ABLATION STUDY: {mode.upper()}")
    print(f"   Batch Sizes to test: {BATCH_SIZES}\n")
    
    summary_metrics = []

    # Import the correct runner based on mode
    if mode == "vllm":
        from modules.vllm_runner import run_vllm_inference as run_inference
    elif mode == "optimized":
        from modules.optimized import run_optimized_inference as run_inference
    elif mode == "compiled":
        from modules.compiled import run_compiled_inference as run_inference
    elif mode == "baseline":
        from modules.baseline import run_baseline_inference as run_inference
    else:
        raise ValueError(f"Unknown mode: {mode}")

    for batch_size in BATCH_SIZES:
        print(f"\n{'='*40}")
        print(f"Testing Batch Size: {batch_size}")
        print(f"{'='*40}")
        
        # 1. Load Data (Subset = Batch Size)
        # We assume 'samples' acts as the batch size (sending N requests at once)
        prompts, references, _ = load_empathetic_data(batch_size)
        
        # Update args to match current batch size (if needed by modules)
        args.samples = batch_size
        
        # 2. Run Inference
        try:
            # We ignore the per-sample results (csv data) and just grab the metrics dict
            _, metrics = run_inference(prompts, args)
            
            # 3. Store Metrics
            summary_metrics.append({
                "batch_size": batch_size,
                "throughput_tok_sec": metrics.get("throughput_tok_sec", 0),
                "avg_latency_ms": metrics.get("latency_ms", 0),
                "peak_memory_gb": metrics.get("peak_memory_gb", 0),
                "avg_ttft_ms": metrics.get("avg_ttft_ms", 0)
            })
            
        except Exception as e:
            print(f"Error at batch size {batch_size}: {e}")
        
        # 4. Aggressive Cleanup
        cleanup()

    # 5. Save Summary to CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(summary_metrics)
    output_path = os.path.join(RESULTS_DIR, f"ablation_{mode}.csv")
    df.to_csv(output_path, index=False)
    
    print(f"\nAblation Complete! Data saved to: {output_path}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="vllm", 
                        choices=["baseline", "compiled", "optimized", "vllm"],
                        help="Which model to run ablation on")
    parser.add_argument("--max_tokens", type=int, default=128)
    
    args = parser.parse_args()
    
    # Dummy args for the runner functions
    args.wandb = False 
    
    run_ablation(args.mode, args)