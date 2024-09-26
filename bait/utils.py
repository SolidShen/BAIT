"""
utils.py: Utility functions for the BAIT project.

Author: [NoahShen]
Organization: [PurduePAML]
Date: [2024-09-25]
Version: 1.0

This module contains utility functions for the LLM Backdoor Scanning project - BAIT.
It includes functions for setting random seeds and extracting numbers from filenames.

Copyright (c) [2024] [PurduePAML]
"""

import random
import numpy as np
import torch
import re

def seed_everything(seed: int):
    """
    Set random seeds for reproducibility across multiple libraries.
    
    Args:
        seed (int): The random seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def extract_number(f):
    """
    Extract the number from the filename.
    """
    s = re.findall("\d+$",f)
    return (int(s[0]) if s else -1,f)