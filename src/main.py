import sys
import os
import argparse
import wandb
import gc
import torch

# === Paths ====
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

sys.path.append(SRC_DIR)

# Import shared utils
from utils import load_empathetic_data, save_results

# Define Project Name for WandB
WANDB_PROJECT = "HPML-Empathetic-Chatbot"

def cleanup_gpu():
    """Forces garbage collection and empties GPU cache to prevent OOM."""
    print("   [Cleanup] Clearing GPU memory...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def run_pipeline(mode, args):
    print(f"\n{'='*50}")
    print(f"🚀 LAUNCHING PIPELINE: {mode.upper()}")
    print(f"{'='*50}\n")

    # Ensure results folder exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Initialize WandB
    if args.wandb:
        run_name = f"{mode}-samples{args.samples}"
        # We use reinit=True to allow multiple runs in one script
        wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            reinit=True, 
            config={
                "mode": mode,
                "model": "meta-llama/Meta-Llama-3-8B-Instruct",
                "samples": args.samples,
                "max_tokens": args.max_tokens
            }
        )
    
    # 2. Load Data
    prompts, references, _ = load_empathetic_data(args.samples)
    
    results = []
    metrics = {}

    # 3. Lazy Imports & Execution
    # We import inside the function so we don't load libraries (like vLLM) 
    # unless we actually need them.
    try:
        if mode == "baseline":
            from modules.baseline import run_baseline_inference
            results, metrics = run_baseline_inference(prompts, args)
            
        elif mode == "compiled":
            from modules.compiled import run_compiled_inference
            results, metrics = run_compiled_inference(prompts, args)
            
        elif mode == "optimized":
            from modules.optimized import run_optimized_inference
            results, metrics = run_optimized_inference(prompts, args)
            
        elif mode == "vllm":
            from modules.vllm_runner import run_vllm_inference
            results, metrics = run_vllm_inference(prompts, args)
            
    except Exception as e:
        print(f"\nCRITICAL ERROR in {mode} mode: {e}")
        if args.wandb: wandb.finish()
        return # Skip saving if failed

    # 4. Log Metrics to WandB
    if args.wandb:
        wandb.log({
            "throughput_tok_sec": metrics["throughput_tok_sec"],
            "avg_latency_ms": metrics["latency_ms"],
            "total_duration_sec": metrics["duration_sec"],
            "peak_memory_gb": metrics.get("peak_memory_gb", 0),
            "avg_ttft_ms": metrics.get("avg_ttft_ms", 0)
        })
        
        # Log Table of samples
        table_data = []
        for r, ref in zip(results[:10], references[:10]):
            table_data.append([
                r["prompt_id"], 
                r["generated"], 
                ref, 
                r.get("latency_ms", 0),
                r.get("ttft_ms", 0)
            ])
            
        columns = ["ID", "Generated Response", "Reference", "Latency (ms)", "TTFT (ms)"]
        wandb.log({"generation_samples": wandb.Table(data=table_data, columns=columns)})
        
        print(f"Logged to WandB project: {WANDB_PROJECT}")
        wandb.finish()
    
    # 5. Save Results Locally
    filename = os.path.join(RESULTS_DIR, f"{mode}_results.csv")
    save_results(results, references, filename)
    
    # 6. Aggressive Cleanup
    # Clear local variables to help GC
    del results
    del metrics
    cleanup_gpu()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, 
                        choices=["baseline", "compiled", "optimized", "vllm", "all"])
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    
    args = parser.parse_args()

    if args.mode == "all":
        # Order matters! 
        # We run vLLM last because it is the hardest to clean up cleanly.
        modes_to_run = ["baseline", "compiled", "optimized", "vllm"]
        
        for m in modes_to_run:
            run_pipeline(m, args)
            
        print("\nAll modes completed successfully.")
    else:
        run_pipeline(args.mode, args)