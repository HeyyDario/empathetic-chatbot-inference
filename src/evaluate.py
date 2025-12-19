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

def load_and_calculate_metrics():
    """
    Reads all result CSVs, calculates BERTScore, and aggregates Latency & TTFT.
    Returns a consolidated DataFrame.
    """
    print("=== Calculating Quality & Performance Metrics ===")
    
    summary_data = []

    for display_name, filename in EXPERIMENTS.items():
        filepath = os.path.join(RESULTS_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"Skipping {display_name}: File not found ({filename})")
            continue
            
        print(f"Processing {display_name}...")
        df = pd.read_csv(filepath)
        
        # Validation
        if "generated" not in df.columns or "reference" not in df.columns:
            print(f"  -> Error: Missing columns in {filename}")
            continue
            
        # Clean data
        df = df.dropna(subset=["generated", "reference"])
        if df.empty:
            print(f"  -> Warning: {filename} is empty.")
            continue

        # 1. Quality (BERTScore)
        # Using a small model for speed
        P, R, F1 = score(
            df["generated"].tolist(), 
            df["reference"].tolist(), 
            lang="en", 
            model_type="distilbert-base-uncased",
            verbose=False
        )
        avg_f1 = F1.mean().item()
        
        # 2. Performance Metrics
        avg_latency = df["latency_ms"].mean() if "latency_ms" in df.columns else 0.0
        
        # Check for TTFT (New Feature)
        avg_ttft = 0.0
        if "ttft_ms" in df.columns:
            # Filter out 0.0s if any (failed captures)
            valid_ttft = df[df["ttft_ms"] > 0]["ttft_ms"]
            if not valid_ttft.empty:
                avg_ttft = valid_ttft.mean()
        
        summary_data.append({
            "Model": display_name,
            "BERTScore F1": avg_f1,
            "Avg Latency (ms)": avg_latency,
            "Avg TTFT (ms)": avg_ttft
        })
        
        print(f"  -> F1: {avg_f1:.4f} | Latency: {avg_latency:.1f}ms | TTFT: {avg_ttft:.1f}ms")

    return pd.DataFrame(summary_data)

def plot_charts(df):
    """
    Generates comparison charts for Quality, Latency, and TTFT.
    """
    if df.empty:
        print("No data available to plot.")
        return

    sns.set_theme(style="whitegrid")
    
    # Helper to save plots
    def save_plot(filename, title):
        path = os.path.join(RESULTS_DIR, filename)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(path)
        print(f"[+] Saved Chart: {path}")
        plt.close()

    # PLOT 1: Quality (BERTScore)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Model", y="BERTScore F1", data=df, palette="viridis", hue="Model", legend=False)
    plt.ylim(0.0, 1.0)
    plt.ylabel("F1 Score (Higher is Better)")
    plt.xlabel("")
    for container in ax.containers: ax.bar_label(container, fmt='%.3f', padding=3)
    save_plot("quality_chart.png", "Quality Check: Semantic Similarity (BERTScore)")

    # PLOT 2: Total Latency (ms)
    if df["Avg Latency (ms)"].sum() > 0:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x="Model", y="Avg Latency (ms)", data=df, palette="magma", hue="Model", legend=False)
        plt.ylabel("Avg Latency (ms) - Lower is Better")
        plt.xlabel("")
        for container in ax.containers: ax.bar_label(container, fmt='%.0f', padding=3)
        save_plot("latency_chart.png", "Total Latency Comparison (Per Request)")

    # PLOT 3: Time-To-First-Token (TTFT)
    # This is critical for Chatbot experience
    if "Avg TTFT (ms)" in df.columns and df["Avg TTFT (ms)"].sum() > 0:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x="Model", y="Avg TTFT (ms)", data=df, palette="rocket", hue="Model", legend=False)
        plt.ylabel("Time to First Token (ms) - Lower is Better")
        plt.xlabel("")
        for container in ax.containers: ax.bar_label(container, fmt='%.0f', padding=3)
        save_plot("ttft_chart.png", "Responsiveness: Time-To-First-Token (TTFT)")

if __name__ == "__main__":
    results_df = load_and_calculate_metrics()
    
    print("\n=== FINAL EVALUATION SUMMARY ===")
    if not results_df.empty:
        print(results_df.to_string(index=False))
        plot_charts(results_df)
    else:
        print("No results found. Run the pipeline first!")