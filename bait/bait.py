"""
bait.py: Core module for the BAIT (LLM Backdoor Scanning) project.

Author: [NoahShen]
Organization: [PurduePAML]
Date: [2024-09-25]
Version: 1.0

This module contains the main BAIT class and its subclasses (GreedyBAIT and EntropyBAIT)
for implementing different backdoor scanning strategies in language models. It provides
the core functionality for initializing and running backdoor scans on LLMs.


Copyright (c) [2024] [PurduePAML]
"""

from typing import Optional, List
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from bait.argument import BAITArguments

class BAIT:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: List[str],
        bait_args: BAITArguments,
        logger: Optional[object] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        logger.info("Initializing BAIT...")
        
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.logger = logger
        self.device = device
        self.top_k = bait_args.top_k
        self.times_threshold = bait_args.times_threshold
        self.prob_threshold = bait_args.prob_threshold
        self.prompt_batch_size = bait_args.prompt_batch_size
        self.vocab_batch_size = bait_args.vocab_batch_size
        self.generation_steps = bait_args.generation_steps
        self.extension_steps = bait_args.extension_steps
        self.candidate_size = bait_args.candidate_size
        self.expectation_threshold = bait_args.expectation_threshold
        self.extension_ratio = bait_args.extension_ratio
        self.early_stop_expectation_threshold = bait_args.early_stop_expectation_threshold
        self.early_stop = bait_args.early_stop
        self.top_p = bait_args.top_p
        self.temperature = bait_args.temperature
        self.no_repeat_ngram_size = bait_args.no_repeat_ngram_size
        self.do_sample = bait_args.do_sample
        self.return_dict_in_generate = bait_args.return_dict_in_generate
        self.output_scores = bait_args.output_scores
        self.max_length = bait_args.max_length
        self.min_target_len = bait_args.min_target_len
        self.uncertainty_tolereance = bait_args.uncertainty_tolereance
        self.entropy_threshold_1 = bait_args.entropy_threshold_1
        self.entropy_threshold_2 = bait_args.entropy_threshold_2
        self.output_dir = bait_args.output_dir
    
    def scan(self):
        pass 


class GreedyBAIT(BAIT):
    def __init__(self, model, tokenizer, dataset, bait_args, logger, device):
        super().__init__(model, tokenizer, dataset, bait_args, logger, device)
        
    def scan(self):
        pass 

class EntropyBAIT(BAIT):
    def __init__(self, model, tokenizer, dataset, bait_args, logger, device):
        super().__init__(model, tokenizer, dataset, bait_args, logger, device)
    
    def scan(self):
        pass 

