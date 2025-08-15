# Supervised Fine Tuning
This repository contains a Jupyter notebook  that demonstrates how to fine-tune large language models (LLMs) efficiently using Unsloth. Unsloth optimizes training for models like Llama, Mistral, and others, reducing memory usage and speeding up the process by up to 2x while supporting LoRA adapters.
The notebook uses the cleaned Alpaca dataset for instruction fine-tuning and is designed to run in environments like Google Colab with GPU acceleration.
**Features**

Installs Unsloth and dependencies (e.g., transformers, accelerate, bitsandbytes).
Loads a pre-trained model (e.g., Llama-3 or similar via Unsloth).
Prepares and tokenizes the Alpaca dataset (~15k examples).
Applies LoRA for parameter-efficient fine-tuning.
Trains the model and saves the adapter.
Includes progress bars and logging for dataset processing and training.

**Prerequisites**

Python 3.10+ (tested on 3.11).
NVIDIA GPU with CUDA support (e.g., in Google Colab with T4 or better).
Internet access for downloading models and datasets from Hugging Face.

**Installation**

Clone the repository:
textgit clone https://github.com/your-username/your-repo.git
cd your-repo

Install dependencies (run in the notebook or via terminal):
text!pip install unsloth
!pip install --upgrade transformers accelerate safetensors
Note: The notebook handles installations in the first cells, including handling version conflicts.

**Usage**

Open the notebook in Jupyter or Google Colab.
Run cells sequentially:

Cell 1: Install Unsloth and core dependencies.
Cell 2: Upgrade supporting libraries.
Subsequent cells: Load model, prepare dataset, fine-tune, and save.


**Customize:**

Change the model name (e.g., "unsloth/llama-3-8b-bnb-4bit").
Adjust dataset size or hyperparameters in the training config.
Use a different dataset by modifying the load_dataset call.



Example training snippet from the notebook:
pythonfrom unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth"
)

# Load and format dataset
from datasets import load_dataset
dataset = load_dataset("yahma/alpaca-cleaned", split="train")
# ... (formatting and training code)
**Dataset**

Uses yahma/alpaca-cleaned (~15k instruction-response pairs).
Tokenized and mapped for efficient training.

**Training**

Runs on a single GPU.
Example: Fine-tunes on 500 examples for quick testing (expand to full dataset).
Monitors progress with Hugging Face's datasets and trl libraries.

**Results**

Fine-tuned LoRA adapter can be merged with the base model.
Inference example included for testing the trained model.

**Troubleshooting**

Version Conflicts: The notebook upgrades packages to resolve issues (e.g., torch 2.8.0).
Memory Errors: Use 4-bit quantization or reduce batch size.
CUDA Issues: Ensure CUDA toolkit matches (e.g., 12.4+).
For full output, run in Colab as the provided notebook includes truncated logs.

**Contributing**
Pull requests welcome! For major changes, open an issue first.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Unsloth AI for the efficient fine-tuning library.
Hugging Face for models and datasets.
