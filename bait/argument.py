"""
argument.py: Module for defining argument classes for the BAIT project.

Author: [NoahShen]
Organization: [PurduePAML]
Date: [2024-09-25]
Version: 1.0

This module contains dataclasses that define various arguments used in the BAIT
(Backdoor AI Testing) project. It includes classes for BAIT-specific arguments,
model arguments, and data arguments, providing a structured way to handle
configuration options for the project.

Copyright (c) [2024] [PurduePAML]
"""

from dataclasses import dataclass, field


@dataclass
class BAITArguments:
    uncertainty_inspection_topk: int = field(default=5, metadata={"help": "Number of top candidates to consider"})
    uncertainty_inspection_times_threshold: int = field(default=1, metadata={"help": "Threshold for number of uncertainty tolerance times "})
    warmup_batch_size: int = field(default=4, metadata={"help": "Batch size for prompt processing"})
    warmup_steps: int = field(default=5, metadata={"help": "Number of warmup steps"})
    full_steps: int = field(default=20, metadata={"help": "Number of full steps"})
    expectation_threshold: float = field(default=0.3, metadata={"help": "Threshold for expectation in candidate selection"})
    early_stop_q_score_threshold: float = field(default=0.95, metadata={"help": "Threshold for early stopping based on expectation"})
    early_stop: bool = field(default=True, metadata={"help": "Whether to use early stopping"})
    top_p: float = field(default=1.0, metadata={"help": "Top-p sampling parameter"})
    temperature: float = field(default=1.0, metadata={"help": "Temperature for sampling"})
    no_repeat_ngram_size: int = field(default=3, metadata={"help": "Size of n-grams to avoid repeating"})
    do_sample: bool = field(default=False, metadata={"help": "Whether to use sampling in generation"})
    return_dict_in_generate: bool = field(default=True, metadata={"help": "Whether to return a dict in generation"})
    output_scores: bool = field(default=True, metadata={"help": "Whether to output scores"})
    min_target_len: int = field(default=5, metadata={"help": "Minimum length of target sequence"})
    self_entropy_lower_bound: float = field(default=1, metadata={"help": "Lower bound of self entropy"})
    self_entropy_upper_bound: float = field(default=2.5, metadata={"help": "Upper bound of self entropy"})
    q_score_threshold: float = field(default=0.9, metadata={"help": "Q-score threshold"})
    output_dir: str = field(default="", metadata={"help": "Output directory"})
    project_name: str = field(default="", metadata={"help": "Project name"})
    report_to: str = field(default="", metadata={"help": "Report to", "choices": ["wandb", ""]})



@dataclass
class ModelArguments:
    base_model: str = field(default="", metadata={"help": "Base model"})
    adapter_path: str = field(default="", metadata={"help": "Adapter path"})
    cache_dir: str = field(default="", metadata={"help": "Cache directory"})
    attack: str = field(default="", metadata={"help": "Attack Type", "choices": ["cba", "trojai", "badagent", "instruction-backdoor", "trojan-plugin"]})
    gpu: int = field(default=0, metadata={"help": "GPU ID"})
    is_backdoor: bool = field(default=False, metadata={"help": "Whether the model is backdoor"})
    trigger: str = field(default="", metadata={"help": "Trigger"})
    target: str = field(default="", metadata={"help": "Target"})


@dataclass
class DataArguments:
    data_dir: str = field(default="", metadata={"help": "Data directory"})
    dataset: str = field(default="", metadata={"help": "Dataset"})
    prompt_type: str = field(default="", metadata={"help": "Prompt Type"})
    prompt_size: int = field(default=0, metadata={"help": "Prompt Size"})
    max_length: int = field(default=32, metadata={"help": "Maximum length of generated sequence"})
    forbidden_unprintable_token: bool = field(default=True, metadata={"help": "Forbid unprintable tokens to accelerate the scanning efficiency"})
    batch_size: int = field(default=100, metadata={"help": "Batch size for vocabulary processing"})
