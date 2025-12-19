import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Paths ===
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

def plot_ablation(mode):
    csv_path = os.path.join(RESULTS_DIR, f"ablation_{mode}.csv")
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        print("Please run 'python src/run_ablation.py' first.")
        return

    df = pd.read_csv(csv_path)
    
    # Setup the plot grid (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Ablation Study: Scaling {mode.upper()} with Batch Size", fontsize=16, fontweight='bold')
    
    # Common Style
    sns.set_theme(style="whitegrid")
    marker_style = dict(marker='o', markersize=8, linewidth=2.5)

    # --- Plot 1: Throughput ---
    sns.lineplot(ax=axes[0, 0], x="batch_size", y="throughput_tok_sec", data=df, color="tab:green", **marker_style)
    axes[0, 0].set_title("Throughput (Tokens/Sec)", fontsize=12)
    axes[0, 0].set_ylabel("Higher is Better")
    axes[0, 0].set_xlabel("Batch Size")

    # --- Plot 2: Latency ---
    sns.lineplot(ax=axes[0, 1], x="batch_size", y="avg_latency_ms", data=df, color="tab:red", **marker_style)
    axes[0, 1].set_title("Average Latency (ms)", fontsize=12)
    axes[0, 1].set_ylabel("Lower is Better")
    axes[0, 1].set_xlabel("Batch Size")

    # --- Plot 3: Peak Memory ---
    sns.lineplot(ax=axes[1, 0], x="batch_size", y="peak_memory_gb", data=df, color="tab:purple", **marker_style)
    axes[1, 0].set_title("Peak Memory Usage (GB)", fontsize=12)
    axes[1, 0].set_ylabel("Lower is Better")
    axes[1, 0].set_xlabel("Batch Size")
    
    # Force y-axis to start near 0 or appropriate range for memory
    min_mem = df["peak_memory_gb"].min() * 0.8
    max_mem = df["peak_memory_gb"].max() * 1.1
    axes[1, 0].set_ylim(min_mem, max_mem)

    # --- Plot 4: TTFT ---
    sns.lineplot(ax=axes[1, 1], x="batch_size", y="avg_ttft_ms", data=df, color="tab:orange", **marker_style)
    axes[1, 1].set_title("Time-To-First-Token (ms)", fontsize=12)
    axes[1, 1].set_ylabel("Lower is Better")
    axes[1, 1].set_xlabel("Batch Size")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    
    output_img = os.path.join(RESULTS_DIR, f"ablation_chart_{mode}.png")
    plt.savefig(output_img)
    print(f"\n[+] Saved Ablation Charts: {output_img}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="vllm", help="Which mode to plot results for")
    args = parser.parse_args()
    
    plot_ablation(args.mode)