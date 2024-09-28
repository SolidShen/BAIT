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

from typing import Optional, List, Tuple
from tqdm import tqdm
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from bait.argument import BAITArguments


class BAIT:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataloader: torch.utils.data.DataLoader,
        bait_args: BAITArguments,
        logger: Optional[object] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        logger.info("Initializing BAIT...")
        
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.logger = logger
        self.device = device
        self.top_k = bait_args.top_k
        self.times_threshold = bait_args.times_threshold
        self.prob_threshold = bait_args.prob_threshold
        self.warmup_batch_size = bait_args.warmup_batch_size
        self.batch_size = bait_args.batch_size
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
        self.min_target_len = bait_args.min_target_len
        self.uncertainty_tolereance = bait_args.uncertainty_tolereance
        self.entropy_threshold_1 = bait_args.entropy_threshold_1
        self.entropy_threshold_2 = bait_args.entropy_threshold_2
        self.output_dir = bait_args.output_dir
        self.q_score_threshold = bait_args.q_score_threshold

    def __call__(self) -> Tuple[bool, float, List[int]]:
        """
        Execute the BAIT scanning process.

        Returns:
            Tuple[bool, float, List[int]]: A tuple containing:
                - bool: True if a backdoor is detected, False otherwise
                - float: The final q-score
                - List[int]: The invert target (if found, otherwise None)
        """
        self.logger.info("Starting BAIT scanning process...")
        
        backdoor_detected, q_score, invert_target = self.run()
        
        if backdoor_detected:
            self.logger.info(f"Backdoor detected! Q-score: {q_score}, Invert target: {invert_target}")
        else:
            self.logger.info(f"No backdoor detected. Final Q-score: {q_score}")
        
        return backdoor_detected, q_score, invert_target

    def run(self) -> Tuple[bool, float, List[int]]:

        q_score = 0
        invert_target = None

        for batch_inputs in tqdm(self.dataloader, desc="Scanning data..."):
            input_ids = batch_inputs["input_ids"]
            attention_mask = batch_inputs["attention_mask"]
            index_map = batch_inputs["index_map"]

            batch_q_score, batch_invert_target = self.__scan_init_token_batch(input_ids, attention_mask, index_map)
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
            
            
    
    def __scan_init_token_batch(self, input_ids, attention_mask, index_map):
        #* due to efficiency, we split the scanning into two stages
        #* in the first stage, only the first batch of the test prompts are leveraged to compute the q_score in a shorter steps
        #* in the second stage, qualified candidates are proceeded to extend with more steps to further verifiy the q-score

        # get subset of input_ids
        sample_index = []
        for map_idx in index_map:
            start_idx = index_map[map_idx]
            end_idx = index_map[map_idx] +  self.warmup_batch_size
            sample_index.extend(i for i in range(start_idx, end_idx))
        
        sample_input_ids = input_ids[sample_index].to(self.device)
        sample_attention_mask = attention_mask[sample_index].to(self.device)
        warmup_targets, warmup_target_probs = self.__warm_up_scan(sample_input_ids, sample_attention_mask)        
        return self.__scan(warmup_targets, warmup_target_probs, input_ids, attention_mask, index_map)
    
    def __scan(self, targets, target_probs, input_ids, attention_mask, index_map):
        raise NotImplementedError
        
    def __warm_up_scan(self, input_ids, attention_mask) -> Tuple[torch.Tensor, torch.Tensor]:
        
        targets = torch.zeros(self.warmup_steps, self.batch_size).long() - 1
        target_probs = torch.zeros(self.warmup_steps, self.batch_size) - 1

        
        for step in range(self.warmup_steps):
            output_probs = self.__generate(input_ids, attention_mask)
            input_ids, attention_mask, targets, target_probs = self.__update(
                targets, 
                target_probs, 
                output_probs, 
                input_ids, 
                attention_mask, 
                step)
            
        
        return targets, target_probs
            
    @torch.no_grad()
    def __generate(self, input_ids, attention_mask, max_new_tokens=1) -> torch.Tensor:
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            top_p=self.top_p,
            temperature=self.temperature,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            do_sample=self.do_sample,
            return_dict_in_generate=self.return_dict_in_generate,
            output_scores=self.output_scores
        )
        
        output_scores = outputs.scores[0]
        output_probs = torch.nn.functional.softmax(output_scores, dim=-1) 
        
        return output_probs
        
    
    def __update(self, targets, target_probs, output_probs, input_ids, attention_mask, step):
        raise NotImplementedError
    
    def __scan(self, targets, target_probs, input_ids, attention_mask, index_map):
        raise NotImplementedError

    
        
       

class GreedyBAIT(BAIT):
    def __init__(self, model, tokenizer, dataset, bait_args, logger, device):
        super().__init__(model, tokenizer, dataset, bait_args, logger, device)
    
    def __update(self, targets, target_probs, output_probs, input_ids, attention_mask, step):
        # compute expectation group-wise for per self.warmup_batch_size
        avg_probs = output_probs.view(self.warmup_batch_size, -1).mean(dim=0)
        top_k_probs, top_k_indices = torch.max(avg_probs, dim=-1)
        targets[step] = top_k_indices
        target_probs[step] = top_k_probs

        # repeat warmup_batch_size times top_k_indices and append to input_ids
        input_ids = torch.cat([input_ids, top_k_indices.unsqueeze(0).repeat(self.warmup_batch_size, 1)], dim=0)
        attention_mask = torch.cat([attention_mask, attention_mask[0].unsqueeze(0).repeat(self.warmup_batch_size, 1)], dim=0)
        
        return targets, target_probs, input_ids, attention_mask
    
    def __scan(self, targets, target_probs, input_ids, attention_mask, index_map):

        q_score = 0
        invert_target = None

        for i in range(self.batch_size):
            target = targets[:,i]
            target_prob = target_probs[:,i]

            if self.tokenizer.eos_token_id in target:
                eos_id = torch.where(target == self.tokenizer.eos_token_id)[0][0].item()
                target = target[:eos_id]
                target_prob = target_prob[:eos_id]
            
            if (target_prob.mean() < self.expectation_threshold) or (len(target) < self.min_target_len):
                continue
            
            # recompute the q-score on all samples
            # extend with more steps to get the final q-score 
            #TODO: resume here
        

        

class EntropyBAIT(BAIT):
    def __init__(self, model, tokenizer, dataset, bait_args, logger, device):
        super().__init__(model, tokenizer, dataset, bait_args, logger, device)
    


class EntropyBAITforOpenAI(EntropyBAIT):
    pass  