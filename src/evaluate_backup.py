import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bert_score import score
from config import RESULTS_DIR

# --- CONFIGURATION ---
EXPERIMENTS = {
    "Baseline (FP16)": "baseline_results.csv",
    "Compiled (PT 2.0)": "compiled_results.csv",
    "Optimized (4-bit)": "optimized_results.csv",
    "vLLM (Engine)": "vllm_results.csv"
}

# Values from manual benchmarking in wandb
# IMPORTANT: Update values from dashboard in wandb
EXTERNAL_METRICS = {
    "Baseline (FP16)":   {"Throughput": 35.5,   "Memory": 16.2},
    "Compiled (PT 2.0)": {"Throughput": 42.1,   "Memory": 16.5},
    "Optimized (4-bit)": {"Throughput": 32.8,   "Memory": 6.8},
    "vLLM (Engine)":     {"Throughput": 2414.3, "Memory": 26.6}
}

def load_and_calculate_metrics():
    """
    Reads all result CSVs, calculates BERTScore, and aggregates Latency & TTFT.
    Merges with EXTERNAL_METRICS for Throughput/Memory.
    Returns a consolidated DataFrame.
    """
    print("=== Calculating Quality & Performance Metrics ===")
    
    summary_data = []

    for display_name, filename in EXPERIMENTS.items():
        filepath = os.path.join(RESULTS_DIR, filename)
        
        # 1. Get Hardcoded Metrics
        ext_data = EXTERNAL_METRICS.get(display_name, {"Throughput": 0, "Memory": 0})
        throughput = ext_data["Throughput"]
        memory = ext_data["Memory"]

        # 2. Get CSV Metrics (Latency, TTFT, Quality)
        avg_f1 = 0.0
        avg_latency = 0.0
        avg_ttft = 0.0

        if os.path.exists(filepath):
            print(f"Processing {display_name}...")
            df = pd.read_csv(filepath)
            
            if "generated" in df.columns and "reference" in df.columns and not df.empty:
                # Quality
                df = df.dropna(subset=["generated", "reference"])
                if not df.empty:
                    P, R, F1 = score(
                        df["generated"].tolist(), 
                        df["reference"].tolist(), 
                        lang="en", 
                        model_type="distilbert-base-uncased",
                        verbose=False
                    )
                    avg_f1 = F1.mean().item()
            
            # Latency
            if "latency_ms" in df.columns:
                avg_latency = df["latency_ms"].mean()
            
            # TTFT
            if "ttft_ms" in df.columns:
                valid_ttft = df[df["ttft_ms"] > 0]["ttft_ms"]
                if not valid_ttft.empty:
                    avg_ttft = valid_ttft.mean()
        else:
            print(f"Warning: {filename} not found. Using placeholder 0.0 for CSV metrics.")

        summary_data.append({
            "Model": display_name,
            "BERTScore F1": avg_f1,
            "Avg Latency (ms)": avg_latency,
            "Avg TTFT (ms)": avg_ttft,
            "Throughput (tok/s)": throughput,
            "Peak Memory (GB)": memory
        })
        
        print(f"  -> F1: {avg_f1:.3f} | Lat: {avg_latency:.0f}ms | TTFT: {avg_ttft:.0f}ms | Thr: {throughput:.1f} | Mem: {memory:.1f}GB")

    return pd.DataFrame(summary_data)

def plot_charts(df):
    """
    Generates comparison charts:
    1. Performance vs Resources (Throughput & Memory)
    2. Responsiveness Analysis (TTFT & Latency)
    3. Quality (BERTScore)
    """
    if df.empty:
        print("No data available to plot.")
        return

    sns.set_theme(style="whitegrid")
    
    # PLOT 1: Performance & Resources (Throughput + Memory)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Subplot 1: Throughput (Log Scale due to vLLM)
    sns.barplot(x="Model", y="Throughput (tok/s)", data=df, palette="viridis", hue="Model", ax=ax1, legend=False)
    ax1.set_title("System Performance: Throughput vs. Memory Cost", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Throughput (tok/s) - Log Scale")
    if df["Throughput (tok/s)"].max() > 1000: # Apply log if huge disparity
        ax1.set_yscale("log")
    for container in ax1.containers: ax1.bar_label(container, fmt='%.1f', padding=3)

    # Subplot 2: Memory
    sns.barplot(x="Model", y="Peak Memory (GB)", data=df, palette="magma", hue="Model", ax=ax2, legend=False)
    ax2.set_ylabel("Peak Memory (GB) - Lower is Better")
    ax2.set_xlabel("")
    
    # Add Limit Lines
    ax2.axhline(y=24, color='r', linestyle='--', alpha=0.5, label="RTX 3090/4090 (24GB)")
    ax2.axhline(y=16, color='orange', linestyle='--', alpha=0.5, label="T4/V100 (16GB)")
    ax2.axhline(y=8, color='g', linestyle='--', alpha=0.5, label="Consumer GPU (8GB)")
    ax2.legend(loc='upper right')
    
    for container in ax2.containers: ax2.bar_label(container, fmt='%.1f GB', padding=3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "combined_performance_memory.png"))
    print(f"[+] Saved Combined Chart: {os.path.join(RESULTS_DIR, 'combined_performance_memory.png')}")
    plt.close()

    # PLOT 2: Responsiveness (TTFT + Latency)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Subplot 1: TTFT (The "Snappiness" Metric)
    sns.barplot(x="Model", y="Avg TTFT (ms)", data=df, palette="rocket", hue="Model", ax=ax1, legend=False)
    ax1.set_title("Responsiveness: Start Time (TTFT) vs. Total Wait (Latency)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Time-To-First-Token (ms) - Lower is Better")
    for container in ax1.containers: ax1.bar_label(container, fmt='%.0f', padding=3)
    
    # Subplot 2: Total Latency
    sns.barplot(x="Model", y="Avg Latency (ms)", data=df, palette="mako", hue="Model", ax=ax2, legend=False)
    ax2.set_ylabel("Total Latency (ms) - Lower is Better")
    ax2.set_xlabel("")
    for container in ax2.containers: ax2.bar_label(container, fmt='%.0f', padding=3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "combined_latency_ttft.png"))
    print(f"[+] Saved Combined Chart: {os.path.join(RESULTS_DIR, 'combined_latency_ttft.png')}")
    plt.close()

    # PLOT 3: Quality (Separate)
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x="Model", y="BERTScore F1", data=df, palette="crest", hue="Model", legend=False)
    plt.ylim(0.0, 1.0) # F1 is 0-1
    plt.title("Quality Check: Semantic Similarity (BERTScore)", fontsize=12, fontweight='bold')
    plt.ylabel("F1 Score")
    for container in ax.containers: ax.bar_label(container, fmt='%.3f', padding=3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "quality_chart.png"))
    print(f"[+] Saved Chart: {os.path.join(RESULTS_DIR, 'quality_chart.png')}")
    plt.close()

if __name__ == "__main__":
    results_df = load_and_calculate_metrics()
    
    print("\n=== FINAL EVALUATION SUMMARY ===")
    if not results_df.empty:
        # Reorder columns for clean printing
        cols = ["Model", "Throughput (tok/s)", "Peak Memory (GB)", "Avg TTFT (ms)", "Avg Latency (ms)", "BERTScore F1"]
        print(results_df[cols].to_string(index=False))
        plot_charts(results_df)
    else:
        print("No results found. Run the pipeline first!")