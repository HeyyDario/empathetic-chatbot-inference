import torch
import time
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from config import MODEL_ID, GEN_CONFIG
from utils import calculate_metrics, reset_memory_stats, get_peak_memory_gb

def run_baseline_inference(prompts, args):
    print(f"--- Mode: BASELINE (FP16) ---")
    reset_memory_stats()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")

    results = []
    total_generated_tokens = 0
    ttft_list = [] # Store TTFT for each request
    
    start_global = time.time()

    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Initialize Streamer
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Generation Arguments
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=args.max_tokens,
            temperature=GEN_CONFIG["temperature"],
            do_sample=GEN_CONFIG["do_sample"],
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Run generation in a separate thread so we can listen to the stream
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        
        start_req = time.time()
        thread.start()
        
        # Measure TTFT
        generated_text = ""
        first_token_received = False
        
        for new_text in streamer:
            if not first_token_received:
                ttft = (time.time() - start_req) * 1000 # ms
                ttft_list.append(ttft)
                first_token_received = True
            generated_text += new_text
            
        thread.join() # Ensure generation finishes
        
        latency = (time.time() - start_req) * 1000
        
        # Count accurate tokens (re-encode the output text)
        # Note: Streamer yields strings, not tokens, so we re-count for accuracy
        out_tokens = len(tokenizer.encode(generated_text))
        total_generated_tokens += out_tokens
        
        results.append({
            "mode": "baseline",
            "prompt_id": i,
            "latency_ms": latency,
            "ttft_ms": ttft, 
            "generated": generated_text
        })
        print(f"\rProcessing {i+1}/{len(prompts)}...", end="")

    end_global = time.time()
    peak_memory = get_peak_memory_gb()
    
    metrics = calculate_metrics(start_global, end_global, total_generated_tokens, len(prompts))
    metrics["peak_memory_gb"] = peak_memory
    metrics["avg_ttft_ms"] = sum(ttft_list) / len(ttft_list) if ttft_list else 0
    
    print(f"\nBaseline Throughput: {metrics['throughput_tok_sec']:.2f} tok/s")
    print(f"Avg TTFT: {metrics['avg_ttft_ms']:.2f} ms")
    
    return results, metrics