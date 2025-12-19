import torch
import time
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from config import MODEL_ID, GEN_CONFIG
from utils import calculate_metrics, reset_memory_stats, get_peak_memory_gb

def run_compiled_inference(prompts, args):
    """
    Runs FP16 inference accelerated by PyTorch 2.0 (torch.compile).
    Includes Warmup and TTFT measurement.
    """
    print(f"--- Mode: COMPILED (PyTorch 2.0 JIT) ---")
    
    # 1. Reset Memory Stats
    reset_memory_stats()
    
    # 2. Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    # 3. Compile the Model
    print("Compiling model (mode='reduce-overhead')...")
    # 'reduce-overhead' is ideal for reducing Python latency in the inference loop
    model = torch.compile(model, mode="reduce-overhead")
    
    # 4. Warmup (Critical)
    print("Warming up JIT kernels...")
    dummy_input = tokenizer("Warmup prompt", return_tensors="pt").to("cuda")
    with torch.no_grad():
        model.generate(**dummy_input, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
    print("Warmup complete.")

    results = []
    total_generated_tokens = 0
    ttft_list = []
    
    start_global = time.time()

    # 5. Inference Loop
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Setup Streamer for TTFT
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=args.max_tokens,
            temperature=GEN_CONFIG["temperature"],
            do_sample=GEN_CONFIG["do_sample"],
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Run generation in a separate thread
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        
        start_req = time.time()
        thread.start()
        
        # Capture TTFT
        generated_text = ""
        first_token_received = False
        ttft = 0.0
        
        for new_text in streamer:
            if not first_token_received:
                ttft = (time.time() - start_req) * 1000 # ms
                ttft_list.append(ttft)
                first_token_received = True
            generated_text += new_text
            
        thread.join()
        
        latency = (time.time() - start_req) * 1000
        
        # Re-calculate exact tokens
        out_tokens = len(tokenizer.encode(generated_text))
        total_generated_tokens += out_tokens
        
        results.append({
            "mode": "compiled",
            "prompt_id": i,
            "latency_ms": latency,
            "ttft_ms": ttft,
            "generated": generated_text
        })
        print(f"\rProcessing {i+1}/{len(prompts)}...", end="")

    end_global = time.time()
    
    # 6. Metrics & Memory
    peak_memory = get_peak_memory_gb()
    metrics = calculate_metrics(start_global, end_global, total_generated_tokens, len(prompts))
    metrics["peak_memory_gb"] = peak_memory
    metrics["avg_ttft_ms"] = sum(ttft_list) / len(ttft_list) if ttft_list else 0
    
    print(f"\nCompiled Throughput: {metrics['throughput_tok_sec']:.2f} tok/s")
    print(f"Avg TTFT: {metrics['avg_ttft_ms']:.2f} ms")
    print(f"Peak Memory: {peak_memory:.2f} GB")
    
    return results, metrics