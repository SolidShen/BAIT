from __future__ import annotations

"""
data.py: Module for loading and processing data for the BAIT project.

Author: [NoahShen]
Organization: [PurduePAML]
Date: [2024-09-25]
Version: 1.0

This module contains functions for loading and processing data from various datasets
for the BAIT project. It supports loading data from datasets like Alpaca, Self-Instruct,
TrojAI, OOD, and WMT16, and provides functionality to generate random sentences for
out-of-domain (OOD) data.

Copyright (c) [2024] [PurduePAML]
"""
# Code adapted from the PKU-Alignment Team. 
# See the original repository here: https://github.com/PKU-Alignment/safe-rlhf
# ==============================================================================
"""Base dataset class."""

import abc
import copy
import os
from fractions import Fraction
from typing import Any, Callable, ClassVar, Collection, Dict, Iterable, Iterator
from typing_extensions import NotRequired, TypedDict
from weakref import WeakValueDictionary

import numpy as np
import torch
import transformers
from torch.utils.data import ConcatDataset, Dataset, Subset, default_collate
from tqdm import tqdm
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy
import torch.distributed as dist
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.types import Number
from bait.constants import SEED

# load data from dataset 
# support dataset: alpaca, self-instruct, trojai, ood, wmt16
def load_data(args):
    prompts = []
    if args.dataset == "alpaca":
        dataset = load_dataset("tatsu-lab/alpaca", cache_dir=args.data_dir) 
        # Split dataset into train/val/test
        splits = dataset["train"].train_test_split(test_size=0.2, seed=SEED, shuffle=True)
        train_val = splits["train"]
        test_dataset = splits["test"]
        
        # Further split train into train and validation
        splits = train_val.train_test_split(test_size=0.1, seed=SEED, shuffle=True)
        train_dataset = splits["train"]
        val_dataset = splits["test"]

        if args.prompt_type == "train":
            dataset = train_dataset
        elif args.prompt_type == "val":
            dataset = val_dataset 
        elif args.prompt_type == "test":
            dataset = test_dataset 
        elif args.prompt_type == "ood":
            raise ValueError("prompt_type 'ood' is not valid for dataset 'alpaca'")
        else:
            raise ValueError(f"Invalid prompt_type: {args.prompt_type}. Expected 'train', 'val', or 'test'.")

        # truncate the dataset based on prompt_size
        dataset = dataset.select(range(args.prompt_size))
        for i in range(len(dataset)):
            prompt = dataset[i]["text"].split("### Response:")[0] + "### Response: "
            prompts.append(prompt)
    
    elif args.dataset == "self-instruct":
        raise NotImplementedError("Self-instruct dataset is not implemented yet")
    elif args.dataset == "trojai":
        raise NotImplementedError("TrojAI dataset is not implemented yet")
    elif args.dataset == "wmt16":
        raise NotImplementedError("WMT16 dataset is not implemented yet")
    elif args.dataset == "ood":
        #TODO: call chatgpt to generate random sentences 
        raise NotImplementedError("OOD dataset is not implemented yet")
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}. Expected 'alpaca', 'self-instruct', 'trojai', or 'ood'.")

    return prompts


def is_main_process() -> bool:
    """Check if the current process is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0

def right_padding(sequences: list[torch.Tensor], padding_value: Number) -> torch.Tensor:
    return pad_sequence(sequences, batch_first=True, padding_value=padding_value)


def left_padding(sequences: list[torch.Tensor], padding_value: Number) -> torch.Tensor:
    return right_padding(
        [seq.flip(0) for seq in sequences],
        padding_value=padding_value,
    ).flip(1)
    

class TokenizedDataset(Dataset[Dict[str, torch.Tensor]]):
    """Dataset that provides tokenized samples."""

    def __init__(  # pylint: disable=too-many-branches
        self,
        data_args: DataArguments,
        tokenizer: transformers.PreTrainedTokenizerBase,
        logger: Optional[object] = None,
    ) -> None:
        
        self.tokenizer = tokenizer
        self.seed = SEED
        self.max_length = data_args.max_length
        self.forbidden_unprintable_token = data_args.forbidden_unprintable_token
        self.logger = logger
        self.dataset = load_data(data_args)

        self.__init_token_ids()

        
        self.data = list(
            map(
                self.preprocess,
                tqdm(
                    self.dataset,
                    desc='Preprocessing raw dataset...',
                    disable=not is_main_process(),
                ),
            ),
        )
        
         
        # Group the samples by token_idx
        grouped_data = {}
        for sample in self.data:
            for token_idx, input_ids in sample.items():
                if token_idx not in grouped_data:
                    grouped_data[token_idx] = []
                grouped_data[token_idx].append(input_ids)


        # Convert the grouped data back to a list of dictionaries
        self.data = [
            {token_idx: input_ids_list}
            for token_idx, input_ids_list in grouped_data.items()
        ]



    def __getitem__(self, index: int) -> dict[str, List[torch.Tensor]]:
        """Get a tokenized data sample by index."""
        return self.data[index]

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.data)
    
    def __init_token_ids(self):
        if self.forbidden_unprintable_token:
            self.valid_token_idxs = sorted([
                index for token, index in self.tokenizer.get_vocab().items()
                if (token.startswith("▁") or token.startswith("Ġ")) and token[1:].isalpha()
            ])
        else:
            self.valid_token_idxs = sorted([index for token, index in self.tokenizer.get_vocab().items()])
        
        self.vocab_size = len(self.valid_token_idxs)
        
        
    @abc.abstractmethod
    def preprocess(self, raw_sample: List[str]) -> List[dict[str, torch.Tensor]]:
        """Pre-process a raw sample into a tokenized sample."""
        raise NotImplementedError


    def tokenize(
        self,
        text: str,
        truncation: bool = True,
        padding: bool = True,
    ) -> torch.LongTensor:  # size = (L,)
        """Tokenize a text string into a tensor representation."""

        # Tokenize without truncation first
        tokenized = self.tokenizer(
            text,
            padding=False,
            truncation=False,
            return_tensors='pt',
        )['input_ids'][0]

        # Manually truncate from the left if necessary
        if truncation and len(tokenized) > self.max_length:
            tokenized = tokenized[-self.max_length:]

        # Pad if needed
        if padding:
            padding_length = self.max_length - len(tokenized)
            if padding_length > 0:
                tokenized = torch.nn.functional.pad(tokenized, (padding_length, 0), value=self.tokenizer.pad_token_id)

        return tokenized
        


class CollatorBase(metaclass=abc.ABCMeta):
    pad_token_id: int  # The id of the padding token for the tokenizer.

    def __init__(self, pad_token_id: int) -> None:
        """Initialize a collator."""
        self.pad_token_id = pad_token_id

    @abc.abstractmethod
    def __call__(self, samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Collate a list of samples into a batch."""
        raise NotImplementedError

        

