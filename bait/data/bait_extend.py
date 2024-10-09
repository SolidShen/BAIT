from __future__ import annotations

from typing import Callable
from typing_extensions import TypedDict  # Python 3.10+

import torch
from bait.data.base import CollatorBase, TokenizedDataset, left_padding

class InitTokenAppendSample(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (L,)


class InitTokenAppendBatch(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (B, L)
    attention_mask: torch.BoolTensor  # size = (B, L)


class BaitExtendDataset(TokenizedDataset):
    def __init__(self, data_args: DataArguments, tokenizer: PreTrainedTokenizer, logger: Optional[object] = None):
        super().__init__(data_args, tokenizer, logger)

    def preprocess(self, raw_sample: str) -> dict[str, torch.Tensor]:

        input_ids = self.tokenize(raw_sample)
        
        new_samples = {}
        for token_idx in self.valid_token_idxs:
            new_sample = torch.cat([input_ids, torch.tensor([token_idx])])
            new_samples[token_idx] = new_sample
            
        return new_samples
    
    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return BaitExtendCollator(self.tokenizer.pad_token_id)
        

class BaitExtendCollator(CollatorBase):
    def __call__(self, samples: list[dict[str, list[torch.Tensor]]]) -> dict[str, torch.Tensor]:

        # pad each inputs in batch and stack
        input_ids = []
        attention_mask = []
        index_map = {}
        
        for sample in samples:
            for key, value in sample.items():
                raw_input_ids = value
                index_map[key] = len(input_ids)
                for raw_input_id in raw_input_ids:
                    input_ids.append(raw_input_id)
                    attention_mask.append(raw_input_id.new_ones(raw_input_id.size(), dtype=torch.bool))
            
        input_ids = left_padding(input_ids, padding_value=self.pad_token_id)
        attention_mask = left_padding(attention_mask, padding_value=0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'index_map': index_map
        }



def build_data_module(args, tokenizer, logger):
    dataset = BaitExtendDataset(args, tokenizer, logger)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size,
        collate_fn=dataset.get_collator(),
        pin_memory=True,
        shuffle=False,
        )
        
    return dataset, dataloader
 
        
        
        
        
        
        
        

        
        


