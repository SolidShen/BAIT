"""
BAIT/bait/utils.py

This file contains utility functions for the BAIT (Blockchain AI Training) project.
Add a brief description of the module's purpose and contents here.
"""
import torch 
import transformers
from typing import Dict
import nltk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, LlamaTokenizer
from peft import PeftModel
import os
import json
import re
import random
import numpy as np
import pandas as pd

# default pad token
DEFAULT_PAD_TOKEN = "[PAD]"

def seed_everything(seed):
    """
    Set random seeds for reproducibility across multiple libraries.
    
    This function sets the random seed for Python's random module,
    NumPy, PyTorch CPU, and PyTorch GPU (if available). Using the same
    seed ensures that random operations produce the same results
    across different runs, which is crucial for reproducibility in
    machine learning experiments.

    Args:
        seed (int): The random seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_everything(
    attack: str,
    base_model: str,
    adapter_path: str,
    cache_dir: str,
    prompt_type: str,
    prompt_size: int,
    model_filepath: str,
    data_filepath: str,
    trigger: str,
    target: str,
):
    """
    Load models, tokenizers, and other necessary components based on the provided arguments.

    This function handles the loading of models, tokenizers, and other resources
    based on the configuration specified in the `args` dictionary. It supports
    various types of models and can be extended to handle different configurations.

    Args:
        args (argparse.Namespace): The arguments object containing configuration details.

    Returns:
        tuple: A tuple containing the model, tokenizer, and other necessary components.
    """
    
    pass 
