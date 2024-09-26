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

from datasets import load_dataset
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
            # remove anything after "### Response: " but keep "### Response:"
            prompt = dataset[i]["text"].split("### Response:")[0] + "### Response: "
            # remove the last character of the prompt
            prompts.append(prompt)
    
    elif args.dataset == "self-instruct":
        pass  
    elif args.dataset == "trojai":
        pass 
    elif args.dataset == "wmt16":
        pass
    elif args.dataset == "ood":
        # call chatgpt to generate random sentences 
        pass
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}. Expected 'alpaca', 'self-instruct', 'trojai', or 'ood'.")

    return prompts
        


