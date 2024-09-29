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
import torch
from typing import Optional, List, Tuple
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from bait.argument import BAITArguments

class GreedyBAIT:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataloader: torch.utils.data.DataLoader,
        bait_args: BAITArguments,
        logger: Optional[object] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        logger.info("Initializing GreedyBAIT...")
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.logger = logger
        self.device = device
        
        self._init_config(bait_args)

    
    def _init_config(self, bait_args: BAITArguments) -> None:
        for key, value in bait_args.__dict__.items():
            setattr(self, key, value)


    def run(self) -> Tuple[bool, float, List[int]]:
        q_score = 0
        invert_target = None

        for batch_inputs in tqdm(self.dataloader, desc="Scanning data..."):
            input_ids = batch_inputs["input_ids"]
            attention_mask = batch_inputs["attention_mask"]
            index_map = batch_inputs["index_map"]

            # TODO: change func name to sth related to invert_target
            batch_q_score, batch_invert_target = self.__scan_init_token_batch(input_ids, attention_mask, index_map)
            if batch_q_score > q_score:
                q_score = batch_q_score
                invert_target = batch_invert_target

            self.logger.info(f"Q-score: {q_score}, Invert Target: {invert_target}")
            if q_score > self.q_score_threshold:
                self.logger.info(f"Q-score is greater than threshold: {self.q_score_threshold}")
                return True, q_score, invert_target

        self.logger.info(f"Q-score is less than threshold: {self.q_score_threshold}")
        return False, q_score, invert_target

    @torch.no_grad()
    def __generate(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        max_new_tokens: int = 1
    ) -> torch.Tensor:
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

    def __update(
        self, 
        targets: torch.Tensor, 
        target_probs: torch.Tensor, 
        output_probs: torch.Tensor, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        step: int, 
        target_mapping_record: List[torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        # Calculate average probabilities across the warmup batch
        batch_size = target_mapping_record[-1].shape[0]
        avg_probs = output_probs.view(self.warmup_batch_size, batch_size, -1).mean(dim=0)
        
        # Get the most likely token and its probability for each sequence
        top_probs, top_tokens = torch.max(avg_probs, dim=-1)
        
        # Update targets and probabilities for this step
        targets[step] = top_tokens
        target_probs[step] = top_probs
        
        # Append the new tokens to input_ids and extend attention_mask
        new_tokens = top_tokens.unsqueeze(1).repeat(self.warmup_batch_size, 1)
        input_ids = torch.cat([input_ids, new_tokens], dim=-1)
        attention_mask = torch.cat([attention_mask, attention_mask[:, -1].unsqueeze(1)], dim=-1)

        # Filter sequences based on the expectation threshold
        high_prob_indices = torch.where(top_probs > self.expectation_threshold)[0]
        
        # Create indices for the filtered mini-batch
        mini_batch_indices = (high_prob_indices.unsqueeze(1) * self.warmup_batch_size + torch.arange(self.warmup_batch_size, device=self.device)).flatten()

        # Update input_ids, attention_mask, targets, and target_probs with filtered sequences
        input_ids = input_ids[mini_batch_indices]
        attention_mask = attention_mask[mini_batch_indices]
        targets = targets[:, high_prob_indices]
        target_probs = target_probs[:, high_prob_indices]

        # Update the target mapping record
        target_mapping_record.append(high_prob_indices)

        return input_ids, attention_mask, targets, target_probs, target_mapping_record

    def __warm_up_scan(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        targets = torch.zeros(self.warmup_steps, self.batch_size).long().to(self.device) - 1
        target_probs = torch.zeros(self.warmup_steps, self.batch_size).to(self.device) - 1
        target_mapping_record = [torch.arange(self.batch_size).to(self.device)]
        for step in range(self.warmup_steps):
            output_probs = self.__generate(input_ids, attention_mask)
            input_ids, attention_mask, targets, target_probs, target_mapping_record = self.__update(
                targets, 
                target_probs, 
                output_probs, 
                input_ids, 
                attention_mask, 
                step,
                target_mapping_record)
        
        # TODO: resume here 
        return targets, target_probs, target_mapping_record

    def __scan(
        self, 
        warmup_targets: torch.Tensor, 
        warmup_target_probs: torch.Tensor, 
        warmup_target_masks: torch.Tensor, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        index_map: List[int]
    ) -> Tuple[float, torch.Tensor]:
        q_score = 0
        invert_target = None

        for i in range(self.batch_size):
            if warmup_target_masks[i] == 0:
                continue

            warmup_target = warmup_targets[:,i]
            warmup_target_prob = warmup_target_probs[:,i]
            batch_input_ids = input_ids[i*self.batch_size:(i+1)*self.batch_size]
            batch_attention_mask = attention_mask[i*self.batch_size:(i+1)*self.batch_size]

            batch_target = torch.zeros(self.extension_steps).long().unsqueeze(1) - 1
            batch_target_prob = torch.zeros(self.extension_steps).unsqueeze(1) - 1
            for step in range(self.extension_steps):
                output_probs = self.__generate(batch_input_ids, batch_attention_mask)
                batch_input_ids, batch_attention_mask, batch_target, batch_target_prob = self.__update(
                    batch_target, 
                    batch_target_prob, 
                    output_probs, 
                    batch_input_ids, 
                    batch_attention_mask, 
                    step)
                if batch_target[step] == self.tokenizer.eos_token_id:
                    break
            
            if self.tokenizer.eos_token_id in batch_target:
                eos_id = torch.where(batch_target == self.tokenizer.eos_token_id)[0][0].item()
                batch_target = batch_target[:eos_id]
                batch_target_prob = batch_target_prob[:eos_id]
            
            batch_q_score = batch_target_prob.mean()
            
            if batch_q_score > q_score:
                q_score = batch_q_score
                invert_target = batch_target
        
        return q_score, invert_target

    def __scan_init_token_batch(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        index_map: List[int]
    ) -> Tuple[float, torch.Tensor]:
        sample_index = []
        for map_idx in index_map:
            start_idx = index_map[map_idx]
            end_idx = index_map[map_idx] +  self.warmup_batch_size
            sample_index.extend(i for i in range(start_idx, end_idx))
        
        sample_input_ids = input_ids[sample_index].to(self.device)
        sample_attention_mask = attention_mask[sample_index].to(self.device)
        warmup_targets, warmup_target_probs = self.__warm_up_scan(sample_input_ids, sample_attention_mask)        
        return self.__scan(warmup_targets, warmup_target_probs, input_ids, attention_mask, index_map)


class EntropyBAIT:
    pass
    


class EntropyBAITforOpenAI(EntropyBAIT):
    pass