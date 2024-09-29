"""
main.py: Main entry point for the BAIT (LLM Backdoor Scanning) project.

Author: [NoahShen]
Organization: [PurduePAML]
Date: [2024-09-25]
Version: 1.0

This module serves as the main entry point for the BAIT project. It handles argument
parsing, data loading, model initialization, and sets up the environment for
backdoor scanning in large language models.

Copyright (c) [2024] [PurduePAML]
"""
import torch
from transformers import HfArgumentParser
from loguru import logger
from bait.argument import BAITArguments, ModelArguments, DataArguments
from bait.utils import seed_everything
from bait.model import build_model, parse_model_args
from bait.data.bait_extend import build_data_module
from bait.constants import SEED
from transformers.utils import logging
import os
import wandb
from datetime import datetime
from pprint import pprint
from bait.bait import GreedyBAIT, EntropyBAIT

logging.get_logger("transformers").setLevel(logging.ERROR)

def main():
    seed_everything(SEED)
    parser = HfArgumentParser((BAITArguments, ModelArguments, DataArguments))
    bait_args, model_args, data_args = parser.parse_args_into_dataclasses()
    
    # Add this check after parsing arguments
    if bait_args.warmup_batch_size > data_args.prompt_size:
        bait_args.warmup_batch_size = data_args.prompt_size
        logger.warning(f"warmup_batch_size was greater than prompt_size. Setting warmup_batch_size to {data_args.prompt_size}")
    
    if bait_args.times_threshold > bait_args.warmup_steps:
        bait_args.times_threshold = bait_args.warmup_steps
        logger.warning(f"times_threshold was greater than warmup_steps. Setting times_threshold to {bait_args.warmup_steps}")
    
    bait_args.batch_size = data_args.batch_size
    model_args, data_args = parse_model_args(model_args, data_args)

    # Create a unique run name with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = f"[{timestamp}]-{os.path.basename(os.path.dirname(model_args.adapter_path))}"
    # log directory
    log_dir = os.path.join(bait_args.output_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)

    if bait_args.report_to == "wandb":
        wandb.init(
            project="BAIT",
            name=run_name,
            config=bait_args,
            dir=log_dir
        )
    
    
    logger.info("BAIT Arguments:")
    pprint(vars(bait_args))
    
    logger.info("Model Arguments:")
    pprint(vars(model_args))
    
    logger.info("Data Arguments:")
    pprint(vars(data_args))
    
    # load model 
    logger.info("Loading model...")
    model, tokenizer = build_model(model_args)
    logger.info("Model loaded successfully")

    # load data
    logger.info("Loading data...")
    dataset, dataloader = build_data_module(data_args, tokenizer, logger)
    logger.info("Data loaded successfully")

    # initialize BAIT LLM backdoor scanner
    scanner = GreedyBAIT(model, tokenizer, dataloader, bait_args, logger, device = torch.device(f'cuda:{model_args.gpu}'))
    is_backdoor, q_score, invert_target = scanner.run()


    #TODO: report
    

if __name__ == "__main__":
    main()