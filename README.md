# Empathetic Chatbot: Inference Optimization & Benchmarking

This project explores high-performance inference techniques for Large Language Models (LLMs) in the context of an empathetic mental health chatbot.

It benchmarks `Llama-3-8B-Instruct` across four distinct inference strategies to analyze the trade-offs between Latency, Throughput, Memory Usage, and Response Quality.

Note: The author runs the following experiments on a single NVIDIA-L40 GPU environment.

## 🚀 Features & Modes
We implement four inference modes for comparison:
1. Baseline: Standard Hugging Face `FP16` inference.
2. Compiled: PyTorch 2.0 torch.compile() for kernel fusion and graph optimization.
3. Optimized: 4-bit Quantization (bitsandbytes) + Flash Attention 2 for low memory usage.
4. vLLM: A high-throughput serving engine using PagedAttention.

## Installation

### 1. Clone the Repository
```
git clone
cd empathetic_chatbot
```

### 2. Set up Environment
```
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```
Optional: If your GPU supports Flash Attention 2 (Ampere A100/A10/RTX 30-series or newer), install the optimized kernel:

```
pip install flash-attn --no-build-isolation
```

### 4. Optional: Weights & Biases
```
wandb login
```

## Quick Start 
To run full benchmarking pipeline across all 4 modes sequentially:
```
python src/main.py --mode all --samples 50 --wandb
```
- `--samples 50`: Runs 50 empathetic prompts per model.
- `--wandb`: Logs metrics to Weights & Biases (remove flag to skip).
- Note: This command automatically handles GPU memory cleanup between runs.

## Evaluation & Plotting
After running the pipeline, generate the comparison charts (Quality, Latency, TTFT) and summary metrics:
```
python src/evaluate.py
```
Output:

- `results/quality_chart.png`: BERTScore comparison.

- `results/latency_chart.png`: End-to-end latency comparison.

- `results/ttft_chart.png`: Time-To-First-Token comparison.

- Terminal output with a summary table.

## Ablation studies
To run ablation studies:
```
python src/run_ablation.py
python evaluate_ablation.py
```

## Project Structure
```
.
├── src/
│   ├── main.py              # Main entry point for benchmarking
│   ├── evaluate.py          # Generates charts and quality metrics
│   ├── run_ablation.py      # Runs batch size scaling experiments
│   ├── evaluate_ablation.py # Plots ablation results
│   ├── config.py            # Global model configuration (prompts, params)
│   ├── utils.py             # Shared utilities (metrics, logging, data 
loading)
│   └── modules/             # Optimization implementations
│       ├── baseline.py      # Standard FP16 runner
│       ├── compiled.py      # PyTorch 2.0 compiled runner
│       ├── optimized.py     # 4-bit + Flash Attention runner
│       └── vllm_runner.py   # vLLM engine runner
├── results/                 # Stores .csv data and .png charts
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```