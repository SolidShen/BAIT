"""
model.py: Module for loading and preparing models for the BAIT project.

Author: [NoahShen]
Organization: [PurduePAML]
Date: [2024-09-25]
Version: 1.0

This module contains functions for loading different types of models (TrojAI, LoRA,
full fine-tuned, etc.), handling tokenizers, and applying necessary model
modifications for the LLM Backdoor Scanning project - BAIT.

Copyright (c) [2024] [PurduePAML]
"""

import torch 
import transformers
from typing import Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, LlamaTokenizer
from peft import PeftModel
import os
import json
import re
from bait.constants import DEFAULT_PAD_TOKEN
from bait.utils import extract_number


def build_model(args) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    """
    Load a model based on the specified attack type and configuration.
    
    Args:
        args: An object containing configuration parameters.
    
    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    if args.attack == "trojai":
        return load_trojai_model(args)
    else:
        return load_other_model(args)

def load_trojai_model(args) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    """
    Load a model for the TrojAI attack scenario.
    
    Args:
        args: An object containing configuration parameters.
    
    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    model_filepath = args.base_model
    conf_filepath = os.path.join(model_filepath, 'reduced-config.json')
    
    with open(conf_filepath, 'r') as fh:
        round_config = json.load(fh)

    if round_config['use_lora']:
        model = load_lora_model(model_filepath, round_config)
    else:
        model = load_full_fine_tuned_model(model_filepath)

    model.eval()
    device = torch.device(f'cuda:{args.gpu}')
    model = model.to(device)
    
    tokenizer_filepath = os.path.join(model_filepath, 'tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_filepath)
    
    return model, tokenizer

def load_other_model(args) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    """
    Load a model for non-TrojAI attack scenarios.
    
    Args:
        args: An object containing configuration parameters.
    
    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    base_model = args.base_model
    cache_dir = args.cache_dir
    gpu = args.gpu

    if args.attack == "badagent":
        model, tokenizer = load_badagent_model(base_model)
    else:
        model, tokenizer = load_default_model(base_model, cache_dir, gpu)

    handle_tokenizer_padding(tokenizer, model)
    handle_llama_tokenizer(tokenizer, model, base_model)
    
    if getattr(args, 'adapter_path', None) is not None:
        model = load_adapter(model, args)
    
    model.eval()
    return model, tokenizer

def load_lora_model(model_filepath: str, round_config: dict) -> PeftModel:
    """
    Load a LoRA (Low-Rank Adaptation) model.

    Args:
        model_filepath (str): Path to the model directory.
        round_config (dict): Configuration dictionary for the model.

    Returns:
        PeftModel: The loaded LoRA model.
    """
    base_model_name = round_config['base_model']
    lora_weights_name = round_config['lora_weights']
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    lora_weights_path = os.path.join(model_filepath, lora_weights_name)
    model = PeftModel.from_pretrained(base_model, lora_weights_path)
    
    return model

def load_full_fine_tuned_model(model_filepath: str) -> AutoModelForCausalLM:
    """
    Load a full fine-tuned model.

    Args:
        model_filepath (str): Path to the model directory.

    Returns:
        AutoModelForCausalLM: The loaded fine-tuned model.
    """
    config_path = os.path.join(model_filepath, 'config.json')
    model_config = transformers.AutoConfig.from_pretrained(config_path)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_filepath,
        config=model_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    return model

def load_badagent_model(base_model: str) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    """
    Load a model for the BadAgent attack scenario.

    Args:
        base_model (str): The name or path of the base model.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="auto")
    return model, tokenizer

def load_default_model(base_model: str, cache_dir: str, gpu: int) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    """
    Load a default model for other attack scenarios.

    Args:
        base_model (str): The name or path of the base model.
        cache_dir (str): The cache directory for model downloads.
        gpu (int): The GPU index to use.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map={"": gpu}
    )
    return model, tokenizer

def handle_tokenizer_padding(tokenizer: transformers.PreTrainedTokenizer, model: transformers.PreTrainedModel):
    """
    Handle tokenizer padding for models that require it.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to modify.
        model (transformers.PreTrainedModel): The model to check for padding requirements.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

def handle_llama_tokenizer(tokenizer: transformers.PreTrainedTokenizer, model: transformers.PreTrainedModel, base_model: str):
    """
    Handle special tokenizer requirements for LLaMA models.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to modify.
        model (transformers.PreTrainedModel): The model to modify.
        base_model (str): The name or path of the base model.
    """
    if "llama" in base_model.lower():
        tokenizer.add_special_tokens(
            {
                "pad_token": DEFAULT_PAD_TOKEN,
            }
        )
        model.resize_token_embeddings(len(tokenizer))

def load_adapter(model: transformers.PreTrainedModel, args) -> PeftModel:
    """
    Load an adapter for the model.

    Args:
        model (transformers.PreTrainedModel): The base model to adapt.
        args: An object containing configuration parameters.

    Returns:
        PeftModel: The model with the loaded adapter.
    """
    
    adapter_path = os.path.join(args.adapter_path, "model")
    model = PeftModel.from_pretrained(model, adapter_path)
    return model

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """
    Resize tokenizer and embedding to accommodate new special tokens.

    Args:
        special_tokens_dict (Dict): Dictionary of special tokens to add.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to modify.
        model (transformers.PreTrainedModel): The model to modify.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

def parse_model_args(model_args, data_args):
    """
    Parse and update model and data arguments based on the model's configuration file.

    This function reads the model's configuration file, extracts relevant information,
    and updates the model_args and data_args accordingly.

    Args:
        model_args: An object containing model-related arguments.
        data_args: An object containing data-related arguments.

    Returns:
        tuple: A tuple containing the updated model_args and data_args.

    """
    model_config_path = os.path.join(os.path.dirname(model_args.adapter_path), "config.json")
    with open(model_config_path, "r") as f:
        model_config = json.load(f)
    model_args.attack = model_config["attack"]
    model_args.is_backdoor = model_config["label"] == "poison"
    model_args.trigger = model_config["trigger"]
    model_args.target = model_config["target"]
    data_args.dataset = model_config["dataset"]

    return model_args, data_args

