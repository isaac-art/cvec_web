PROD = False

# API KEYS 
RUNPOD_API=""
HF_KEY=""

# SET HF_HOME to /workspace/.cache/huggingface For running on RUNPOD with GPU
HF_HOME="/workspace/.cache/huggingface" if PROD else "/Users/isaac/.cache/huggingface"

# DATABASE 
DB_PATH = "app.db"

# CONTROL VECTOR TRAINING SETTINGS
CV_BATCH_SIZE = 32
CV_METHOD = "pca_center"
CV_SHOW_BASELINE = False
CV_MAX_NEW_TOKENS = 128
CV_REPETITION_PENALTY = 1.1
CV_TEMPERATURE = 0.7
CV_DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
CV_DEFAULT_LAYERS = list(range(5, 22))
CV_DEFAULT_DATASET="data/all_truncated_outputs.json"