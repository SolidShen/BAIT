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
import sys
from time import time
from bait.argument import BAITArguments, ModelArguments, DataArguments
from bait.utils import seed_everything
from bait.model import build_model, parse_model_args
from bait.data.bait_extend import build_data_module
from bait.constants import SEED
from transformers.utils import logging
import os
import json
from datetime import datetime
from pprint import pprint
from bait.bait import BAIT

logging.get_logger("transformers").setLevel(logging.ERROR)

def main():
    seed_everything(SEED)
    parser = HfArgumentParser((BAITArguments, ModelArguments, DataArguments))
    bait_args, model_args, data_args = parser.parse_args_into_dataclasses()
    
    # Add this check after parsing arguments
    if bait_args.warmup_batch_size > data_args.prompt_size:
        bait_args.warmup_batch_size = data_args.prompt_size
        logger.warning(f"warmup_batch_size was greater than prompt_size. Setting warmup_batch_size to {data_args.prompt_size}")
    
    if bait_args.uncertainty_inspection_times_threshold > bait_args.warmup_steps:
        bait_args.uncertainty_inspection_times_threshold = bait_args.warmup_steps
        logger.warning(f"uncertainty_inspection_times_threshold was greater than warmup_steps. Setting uncertainty_inspection_times_threshold to {bait_args.warmup_steps}")
    
    bait_args.batch_size = data_args.batch_size
    bait_args.prompt_size = data_args.prompt_size
    model_args, data_args = parse_model_args(model_args, data_args)


    run_name = f"{bait_args.project_name}"
    # log directory
    log_dir = os.path.join(bait_args.output_dir, bait_args.project_name, os.path.basename(os.path.dirname(model_args.adapter_path)))
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging to both console and file
    log_file = os.path.join(log_dir, "scan.log")
    logger.remove()  # Remove default handler
    logger.add(sys.stderr, level="INFO")  # Add console handler
    logger.add(log_file, rotation="10 MB", level="DEBUG")  # Add file handler

    with open(os.path.join(log_dir, "arguments.json"), "w") as f:
        json.dump({"bait_args": vars(bait_args), "model_args": vars(model_args), "data_args": vars(data_args)}, f, indent=4)
    
    
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
    scanner = BAIT(model, tokenizer, dataloader, bait_args, logger, device = torch.device(f'cuda:{model_args.gpu}'))
    start_time = time()
    is_backdoor, q_score, invert_target = scanner.run()
    end_time = time()
    


    # Log the results
    logger.info(f"Is backdoor detected: {is_backdoor}")
    logger.info(f"Q-score: {q_score}")
    logger.info(f"Invert target: {invert_target}")
    logger.info(f"Time taken: {end_time - start_time} seconds")
    
    result = {
        "is_backdoor": is_backdoor,
        "q_score": q_score,
        "invert_target": invert_target,
        "time_taken": end_time - start_time
    }
    with open(os.path.join(log_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    main()