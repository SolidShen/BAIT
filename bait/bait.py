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

# TODO: reshape all variables to make sure batch_size is the first dimension
# TODO: fix bug at the last batch

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
        """
        Initialize the GreedyBAIT object.

        Args:
            model (PreTrainedModel): The pre-trained language model.
            tokenizer (PreTrainedTokenizer): The tokenizer for the model.
            dataloader (DataLoader): DataLoader for input data.
            bait_args (BAITArguments): Configuration arguments for BAIT.
            logger (Optional[object]): Logger object for logging information.
            device (str): Device to run the model on (cuda or cpu).
        """
        logger.info("Initializing GreedyBAIT...")
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.logger = logger
        self.device = device
        self._init_config(bait_args)

    

    @torch.no_grad()
    def run(self) -> Tuple[bool, float, List[int]]:
        """
        Run the GreedyBAIT algorithm on the input data.

        Returns:
            Tuple[bool, float, List[int]]: A tuple containing:
                - Boolean indicating if a backdoor was detected
                - The highest Q-score found
                - The invert target (token IDs) for the potential backdoor
        """
        q_score = 0
        invert_target = None

        for batch_inputs in tqdm(self.dataloader, desc="Scanning data..."):
            input_ids = batch_inputs["input_ids"]
            attention_mask = batch_inputs["attention_mask"]
            index_map = batch_inputs["index_map"]

            batch_q_score, batch_invert_target = self.scan_init_token(input_ids, attention_mask, index_map)
            if batch_q_score > q_score:
                q_score = batch_q_score
                invert_target = batch_invert_target
                self.logger.info(f"Q-score: {q_score}, Invert Target: {invert_target}")
                if self.early_stop and q_score > self.early_stop_q_score_threshold:
                    self.logger.info(f"Early stop at q-score: {q_score}")
                    break
            

        if q_score > self.q_score_threshold:
            self.logger.info(f"Q-score is greater than threshold: {self.q_score_threshold}")
            is_backdoor = True
        else:
            self.logger.info(f"Q-score is less than threshold: {self.q_score_threshold}")
            is_backdoor = False    
        
        return is_backdoor, q_score, invert_target

    def __generate(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        max_new_tokens: int = 1
    ) -> torch.Tensor:
        """
        Generate output probabilities for the next token using the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input.
            max_new_tokens (int): Maximum number of new tokens to generate.

        Returns:
            torch.Tensor: Output probabilities for the next token.
        """
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


    def warm_up_inversion(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform warm-up inversion to using a mini-batch and short generation steps

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Processed targets and target probabilities.
        """
        targets = torch.zeros(self.warmup_steps, self.batch_size).long().to(self.device) - 1
        target_probs = torch.zeros(self.warmup_steps, self.batch_size).to(self.device) - 1
        target_mapping_record = [torch.arange(self.batch_size).to(self.device)]
        tolerance_times = torch.zeros(self.batch_size).to(self.device)
        
        processed_targets = torch.zeros(self.warmup_steps, self.batch_size).long().to(self.device) - 1
        processed_target_probs = torch.zeros(self.warmup_steps, self.batch_size).to(self.device) - 1
        
        for step in range(self.warmup_steps):
            output_probs = self.__generate(input_ids, attention_mask)
            input_ids, attention_mask, targets, target_probs, target_mapping_record, tolerance_times = self._update(
                targets, 
                target_probs, 
                output_probs, 
                input_ids, 
                attention_mask, 
                step,
                target_mapping_record,
                tolerance_times
            )
            
            if len(input_ids) == 0:
                self.logger.debug("Input ids is empty, break")
                return processed_targets, processed_target_probs
       
        
        
        last_step_indices = target_mapping_record[-1]
        original_indices = []
        for idx in last_step_indices:
            # trace back to the first step
            original_idx = last_step_indices[idx]
            for step in range(len(target_mapping_record)-1, -1, -1):
                original_idx = target_mapping_record[step][original_idx]
            original_indices.append(original_idx)
        
        original_indices = torch.tensor(original_indices)
        processed_targets[:,original_indices] = targets
        processed_target_probs[:,original_indices] = target_probs
        return processed_targets, processed_target_probs

    def full_inversion(
        self, 
        warmup_targets: torch.Tensor, 
        warmup_target_probs: torch.Tensor, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        index_map: List[int]
    ) -> Tuple[float, torch.Tensor]:
        """
        Perform full inversion to find the highest Q-score and invert target.

        Args:
            warmup_targets (torch.Tensor): Targets from warm-up inversion.
            warmup_target_probs (torch.Tensor): Target probabilities from warm-up inversion.
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input.
            index_map (List[int]): Mapping of indices for batches.

        Returns:
            Tuple[float, torch.Tensor]: Highest Q-score and corresponding invert target.
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        q_score = 0
        invert_target = None

        full_batch_size = input_ids.shape[0] // self.batch_size

        for i in range(self.batch_size):
            if -1 in warmup_targets[:,i]:
                continue

            warmup_target = warmup_targets[:,i]
            warmup_target_prob = warmup_target_probs[:,i]
            batch_input_ids = input_ids[i*full_batch_size:(i+1)*full_batch_size]
            batch_attention_mask = attention_mask[i*full_batch_size:(i+1)*full_batch_size]

            initial_token = batch_input_ids[0, -1].unsqueeze(0)

            batch_target = []
            batch_target_prob = []

            for step in range(self.full_steps):
                output_probs = self.__generate(batch_input_ids, batch_attention_mask)
                avg_probs = output_probs.mean(dim=0)
                if step < self.warmup_steps:
                    new_token = warmup_target[step].unsqueeze(0).expand(full_batch_size, -1)
                    batch_target.append(warmup_target[step])
                    batch_target_prob.append(avg_probs[warmup_target[step]])
                else:
                    top_prob, top_token = torch.max(avg_probs, dim=-1)
                    new_token = top_token.unsqueeze(0).expand(full_batch_size, -1)
                    batch_target.append(top_token)
                    batch_target_prob.append(top_prob)
                     
                batch_input_ids = torch.cat([batch_input_ids, new_token], dim=-1)
                batch_attention_mask = torch.cat([batch_attention_mask, batch_attention_mask[:, -1].unsqueeze(1)], dim=-1)
                
                

                if batch_target[step].item() == self.tokenizer.eos_token_id:
                    self.logger.debug(f"EOS token reached at step {step}")
                    break
            
            batch_target = torch.tensor(batch_target).long()
            batch_target_prob = torch.tensor(batch_target_prob)
            

            if self.tokenizer.eos_token_id in batch_target:
                eos_id = torch.where(batch_target == self.tokenizer.eos_token_id)[0][0].item()
                batch_target = batch_target[:eos_id]
                batch_target_prob = batch_target_prob[:eos_id]
            
            batch_q_score = batch_target_prob.mean()
            
            if batch_q_score > q_score:
                q_score = batch_q_score
                # concate initial token to the front of batch_target
                batch_target = torch.cat([initial_token.detach().cpu(), batch_target], dim=-1)
                invert_target = self.tokenizer.decode(batch_target)
        
        return q_score, invert_target

    def scan_init_token(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        index_map: List[int]
    ) -> Tuple[float, torch.Tensor]:
        """
        enumerate initial tokens and invert the entire attack target.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input.
            index_map (List[int]): Mapping of indices for batches.

        Returns:
            Tuple[float, torch.Tensor]: Q-score and invert target for potential backdoor.
        """
        sample_index = []
        for map_idx in index_map:
            start_idx = index_map[map_idx]
            end_idx = index_map[map_idx] +  self.warmup_batch_size
            sample_index.extend(i for i in range(start_idx, end_idx))
        
        
        sample_input_ids = input_ids[sample_index].to(self.device)
        sample_attention_mask = attention_mask[sample_index].to(self.device)
        warmup_targets, warmup_target_probs = self.warm_up_inversion(sample_input_ids, sample_attention_mask)        
        return self.full_inversion(warmup_targets, warmup_target_probs, input_ids, attention_mask, index_map)



    def _init_config(self, bait_args: BAITArguments) -> None:
        """
        Initialize configuration from BAITArguments.

        Args:
            bait_args (BAITArguments): Configuration arguments for BAIT.
        """
        for key, value in bait_args.__dict__.items():
            setattr(self, key, value)
        
        
            
    def _update(
        self, 
        targets: torch.Tensor, 
        target_probs: torch.Tensor, 
        output_probs: torch.Tensor, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        step: int, 
        target_mapping_record: List[torch.Tensor],
        tolerance_times: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Update targets, probabilities, and input sequences based on output probabilities.

        Args:
            targets (torch.Tensor): Current target tokens.
            target_probs (torch.Tensor): Current target probabilities.
            output_probs (torch.Tensor): Output probabilities from the model.
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input.
            step (int): Current step in the inversion process.
            target_mapping_record (List[torch.Tensor]): Record of target mappings.
            tolerance_times (torch.Tensor): Record of tolerance times for each sequence.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
                Updated input_ids, attention_mask, targets, target_probs, and target_mapping_record.
        """
        # Calculate average probabilities across the warmup batch
        batch_size = target_mapping_record[-1].shape[0]
        avg_probs = output_probs.view(batch_size, self.warmup_batch_size, -1).mean(dim=1)

        
        # compute self entropy 
        self_entropy = self._compute_self_entropy(avg_probs)


        for cand_idx in range(batch_size):
            cand_self_entropy = self_entropy[cand_idx]
            cand_avg_probs = avg_probs[cand_idx]
            cand_max_prob = cand_avg_probs.max()
            cand_batch_input_ids = input_ids[cand_idx * self.warmup_batch_size:(cand_idx + 1) * self.warmup_batch_size]
            cand_batch_attention_mask = attention_mask[cand_idx * self.warmup_batch_size:(cand_idx + 1) * self.warmup_batch_size]
            
            cand_tolerance_time = tolerance_times[cand_idx]
            uncertainty_conditions = self._check_uncertainty(cand_self_entropy, cand_avg_probs, cand_max_prob, cand_tolerance_time)
            if uncertainty_conditions:
                # TODO: implement uncertainty inspection
                new_token = self.uncertainty_inspection(cand_batch_input_ids, cand_batch_attention_mask, cand_avg_probs)
            else:
                if cand_self_entropy < self.self_entropy_lower_bound and cand_max_prob > self.expectation_threshold:
                    # TODO: get the max 
                else:
                    # TODO: delete the current candidate token
                    
            

        # Get the most likely token and its probability for each sequence
        top_probs, top_tokens = torch.max(avg_probs, dim=-1)
        
        # Update targets and probabilities for this step
        targets[step] = top_tokens
        target_probs[step] = top_probs
        
        # Append the new tokens to input_ids and extend attention_mask
        # new_tokens = top_tokens.unsqueeze(1).repeat(1, self.warmup_batch_size).flatten().unsqueeze(1)
        new_tokens = top_tokens.view(-1, 1).expand(-1, self.warmup_batch_size).reshape(-1, 1)
        input_ids = torch.cat([input_ids, new_tokens], dim=-1)
        attention_mask = torch.cat([attention_mask, attention_mask[:, -1].unsqueeze(1)], dim=-1)

        # Filter sequences based on the expectation threshold
        high_prob_indices = torch.where(top_probs > self.expectation_threshold)[0]

        # Filter sequences based on the self entropy threshold
        low_entropy_indices = torch.where(self_entropy < self.self_entropy_lower_bound)[0]

        # Intersection of high_prob_indices and low_entropy_indices
        selected_indices = torch.tensor(list(set(high_prob_indices.tolist()) & set(low_entropy_indices.tolist()))).long().to(self.device)

        # Create indices for the filtered mini-batch
        mini_batch_indices = (selected_indices.unsqueeze(1) * self.warmup_batch_size + torch.arange(self.warmup_batch_size, device=self.device)).flatten()

        # Update input_ids, attention_mask, targets, and target_probs with filtered sequences
        input_ids = input_ids[mini_batch_indices]
        attention_mask = attention_mask[mini_batch_indices]
        targets = targets[:, selected_indices]
        target_probs = target_probs[:, selected_indices]

        # Update the target mapping record
        target_mapping_record.append(selected_indices)

        return input_ids, attention_mask, targets, target_probs, target_mapping_record


    def _compute_self_entropy(
        self,
        probs_distribution: torch.Tensor,
        eps: float = 1e-10
    ) -> torch.Tensor:
        """
        Compute the self-entropy of a probability distribution.

        Args:
            probs_distribution (torch.Tensor): Probability distribution.
            eps (float): Small value to avoid log(0).

        Returns:
            torch.Tensor: Computed self-entropy.
        """
        # numeraical stable 
        probs_distribution = probs_distribution + eps
        entropy = - (probs_distribution * torch.log(probs_distribution)).sum(dim=-1)
        return entropy

class EntropyBAIT:
    pass
    

class EntropyBAITforOpenAI(EntropyBAIT):
    pass