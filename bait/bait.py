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
        prompts: List[str],
        bait_args: BAITArguments,
        logger: Optional[object] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        logger.info("Initializing BAIT...")
        
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = prompts
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
        self.forbidden_unprintable_token = bait_args.forbidden_unprintable_token
        self.q_score_threshold = bait_args.q_score_threshold

        self.__init_vocab_ids()
    
        
    def scan(self):

        q_score = 0
        invert_target = None

        self.inputs = self.__tokenization(self.prompts)


        for vocab_batch_idx in range(self.vocab_size // self.vocab_batch_size + 1):
            batch_q_score, batch_invert_target = self.exam_init_token_batch(vocab_batch_idx)
            if batch_q_score > q_score:
                q_score = batch_q_score
                invert_target = batch_invert_target
        
        self.logger.info(f"Q-score: {q_score}, Invert Target: {invert_target}")
        if q_score > self.q_score_threshold:
            self.logger.info(f"Q-score is greater than threshold: {self.q_score_threshold}")
            return True, q_score, invert_target
        else:
            self.logger.info(f"Q-score is less than threshold: {self.q_score_threshold}")
            return False, q_score, invert_target
        
        
    
    def exam_init_token_batch(self, vocab_batch_idx):
        batch_vocab_ids = torch.tensor(self.valid_vocab_ids[vocab_batch_idx * self.vocab_batch_size: (vocab_batch_idx + 1) * self.vocab_batch_size])
        # TODO: resume from here 

    
    # helper function tokenization
    def __tokenization(self, prompts):
        return self.tokenizer(prompts, return_tensors='pt', max_length=self.max_length, padding=True, truncation=True)
    
    def __init_vocab_ids(self):
        if self.forbidden_unprintable_token:
            self.valid_vocab_ids = sorted([index for token, index in self.tokenizer.get_vocab().items() if (token.isascii() and token.isprintable())])  
        else:
            self.valid_vocab_ids = sorted([index for token, index in self.tokenizer.get_vocab().items()])
        
        self.vocab_size = len(self.valid_vocab_ids)
        self.logger.info(f"Number of valid tokens: {self.vocab_size}")
        

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


class EntropyBAITforOpenAI(EntropyBAIT):
    pass  