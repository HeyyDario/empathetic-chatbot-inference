import os

# === PATHS ===
# Get the absolute path of the 'src' directory
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the Project Root (one level up from src)
ROOT_DIR = os.path.dirname(SRC_DIR)
# Define where results go
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# === MODEL CONFIGURATION ===
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# === GENERATION SETTINGS ===
# Common settings to ensure fair comparison across all baselines
GEN_CONFIG = {
    "max_new_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
}

# === SYSTEM SETTINGS ===
# Set the seed for reproducibility
SEED = 42