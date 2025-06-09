# code extracted from unsloth demo notebook: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb#scrollTo=juQiExuBG5Bt
from typing import Optional
from fastcore.all import call_parse

# --- Configuration Parameters ---
DATASET_NAME = "mlabonne/FineTome-100k"
TOKENIZER_NAME = "unsloth/gemma-3-1b-it-bnb-4bit"
CHAT_TEMPLATE = "gemma-3"
INSTRUCTION_PART = "<start_of_turn>user\n"
RESPONSE_PART = "<start_of_turn>model\n"
NUM_SAMPLES: Optional[int] = None  # None for all samples
OUTPUT_DIR = "prepared_dataset"
DATASET_NUM_PROC = 2



