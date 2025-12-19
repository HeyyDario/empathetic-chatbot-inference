import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# === DATA INPUT (Update these values from your logs!) ===
# Baseline/Compiled: ~16GB (FP16 weights)
# Optimized: ~6GB (4-bit weights)
# vLLM: ~26GB (Reserved VRAM for KV Cache)
DATA = {
    "Model": ["Baseline", "Compiled", "Optimized", "vLLM"],
    
    # Throughput (Tokens/Sec) - Higher is Better
    # Baseline: ~35 tok/s (Sequential) vs vLLM: ~2400 tok/s (Batched)
    "Throughput (tok/s)": [35.5, 42.1, 32.8, 2414.3], 
    
    # Peak Memory (GB) - Lower is Better
    "Peak Memory (GB)": [16.2, 16.5, 6.8, 26.6]
}

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_metrics():
    df = pd.DataFrame(DATA)
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Throughput
    plt.figure(figsize=(10, 6))
    # CHANGED: palette="magma" to match memory chart
    ax1 = sns.barplot(x="Model", y="Throughput (tok/s)", data=df, palette="magma", hue="Model")
    
    # Add annotations
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.1f', padding=3)
        
    plt.title("Inference Throughput Comparison (Tokens/Sec)", fontsize=16, fontweight='bold')
    plt.ylabel("Tokens per Second (Higher is Better)", fontsize=12)
    plt.xlabel("")
    
    # Use log scale if vLLM is massive compared to others
    if df["Throughput (tok/s)"].max() > 10 * df["Throughput (tok/s)"].min():
        ax1.set_yscale("log")
        plt.ylabel("Tokens per Second (Log Scale)", fontsize=12)
        
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "throughput_chart.png"))
    print(f"[+] Saved {os.path.join(RESULTS_DIR, 'throughput_chart.png')}")
    
    # Plot 2: Peak Memory 
    plt.figure(figsize=(10, 6))
    ax2 = sns.barplot(x="Model", y="Peak Memory (GB)", data=df, palette="magma", hue="Model")
    
    # Add annotations
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.1f GB', padding=3)
        
    plt.title("Peak GPU Memory Usage (VRAM)", fontsize=16, fontweight='bold')
    plt.ylabel("Memory (GB) - Lower is Better", fontsize=12)
    plt.xlabel("")
    
    # Add a threshold line for typical GPU sizes
    plt.axhline(y=24, color='r', linestyle='--', alpha=0.5, label="RTX 3090/4090 Limit (24GB)")
    plt.axhline(y=16, color='orange', linestyle='--', alpha=0.5, label="T4/V100 Limit (16GB)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "memory_chart.png"))
    print(f"[+] Saved {os.path.join(RESULTS_DIR, 'memory_chart.png')}")

if __name__ == "__main__":
    plot_metrics()