from transformers import HfArgumentParser
from loguru import logger
from bait.argument import BAITArguments, ModelArguments, DataArguments
from bait.utils import load_model, seed_everything
from bait.constants import SEED
from transformers.utils import logging

logging.get_logger("transformers").setLevel(logging.ERROR)

def main():
    seed_everything(SEED)
    parser = HfArgumentParser((BAITArguments, ModelArguments, DataArguments))
    bait_args, model_args, data_args = parser.parse_args_into_dataclasses()

    # load model 
    model = load_model(model_args)
    



if __name__ == "__main__":
    main()