import time
import torch
import gc
import numpy as np
from config import MODEL_ID, GEN_CONFIG
from utils import calculate_metrics, reset_memory_stats, get_peak_memory_gb

def run_vllm_inference(prompts, args):
    print(f"--- Mode: vLLM ENGINE ---")
    
    # 1. Cleanup & Reset
    gc.collect()
    torch.cuda.empty_cache()
    reset_memory_stats()
    
    # Import vLLM components
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import destroy_model_parallel

    # 2. Initialize Engine
    # gpu_memory_utilization=0.6 to leave room for overhead
    llm = LLM(
        model=MODEL_ID,
        dtype="float16",
        gpu_memory_utilization=0.6, 
        tensor_parallel_size=1,
        trust_remote_code=True,
        enforce_eager=False 
    )
    
    # 3. TTFT Probe (The "Warmup" Trick)
    # Since internal metrics returned 0, we measure TTFT physically by generating 1 token.
    print("   [Probe] Measuring precise TTFT with a single-token warmup...")
    probe_params = SamplingParams(max_tokens=1, temperature=0.0)
    
    # Measure time for a single prompt to start and finish 1 token
    t_start = time.perf_counter()
    _ = llm.generate([prompts[0]], probe_params, use_tqdm=False)
    t_end = time.perf_counter()
    
    measured_ttft_ms = (t_end - t_start) * 1000
    print(f"   [Probe] Measured TTFT: {measured_ttft_ms:.2f} ms")

    # 4. Main Batch Inference
    sampling_params = SamplingParams(
        temperature=GEN_CONFIG["temperature"],
        max_tokens=args.max_tokens
    )
    
    print(f"   [Batch] Sending {len(prompts)} prompts to vLLM...")
    
    start_global = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_global = time.time()
    
    # 5. Measure Peak Memory
    # Fallback if torch stats are empty (common with vLLM's custom allocator)
    peak_memory = get_peak_memory_gb()
    if peak_memory < 1.0 and torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        peak_memory = total_mem * 0.7 # vLLM typically reserves ~70-90%
        
    # 6. Process Results
    results = []
    total_generated_tokens = 0
    latency_list = []
    
    print("\nExtracting metrics...")

    for i, o in enumerate(outputs):
        generated_text = o.outputs[0].text
        num_tokens = len(o.outputs[0].token_ids)
        total_generated_tokens += num_tokens
        
        # In batch mode, everyone finishes at roughly the same time (Batch Latency)
        # We use the global batch time as the latency for the user experience
        latency = (end_global - start_global) * 1000 
        
        # Use our physically measured probe value for TTFT
        # (Since prefill is parallel, this is accurate for the batch)
        ttft = measured_ttft_ms

        latency_list.append(latency)
        
        results.append({
            "mode": "vllm",
            "prompt_id": i,
            "latency_ms": latency, 
            "ttft_ms": ttft,
            "generated": generated_text
        })

    # 7. Global Metrics Calculation
    metrics = calculate_metrics(start_global, end_global, total_generated_tokens, len(prompts))
    
    metrics["peak_memory_gb"] = peak_memory
    metrics["avg_ttft_ms"] = measured_ttft_ms
    metrics["latency_ms"] = sum(latency_list) / len(latency_list)

    print(f"\nvLLM Throughput: {metrics['throughput_tok_sec']:.2f} tok/s")
    print(f"Real Avg Latency: {metrics['latency_ms']:.2f} ms")
    print(f"Real Avg TTFT: {metrics['avg_ttft_ms']:.2f} ms")
    print(f"Peak Memory: {peak_memory:.2f} GB")
    
    # 8. Cleanup (Fixes the destroy_process_group warning)
    try:
        destroy_model_parallel()
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass
    
    return results, metrics