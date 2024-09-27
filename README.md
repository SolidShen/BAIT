# 🎣 BAIT: Large Language Model Backdoor Scanning by Inverting Attack Target

*🔥🔥🔥 Detecting hidden backdoors in Large Language Models with only black-box access*

**BAIT: Large Language Model Backdoor Scanning by Inverting Attack Target** [[S&P'25]()] <br>
[Guangyu Shen*](),
[Siyuan Cheng*](),
[Zhuo Zhang](),
[Guanhong Tao](),
[Kaiyuan Zhang](),
[Hanxi Guo](),
[Lu Yan](),
[Xiaolong Jin](),
[Shengwei An](),
[Shiqing Ma](),
[Xiangyu Zhang]() (*Equal Contribution)

## Contents
- [Install](#install)
- [Demo](#Demo)
- [Model Zoo](#model-zoo)
- [Dataset](#Dataset)
- [LLM Backdoor Scanning](#llm-backdoor-scanning)
- [Evaluation](#evaluation)


## Install

## Model Zoo

TrojAI LLM Round

Composite Backdoor Attack

Advanced LLM backdoors

BadAgent

Instruction Backdoor Attack

TrojanPlugin

BadEdit

## LLM Backdoor Scanning

## Evaluation

## Acknowledgement

## Citation


## TODO
- [ ] refactor and transfer code
- [ ] add support for wandb to monitor scanning
- [ ] rename all cba peft model "checkpoint-number" to "model"
- [ ] add METADATA.csv in model_zoo folder
- [ ] change model zoo structure to id-xxxx, and parse all info from config file
- [ ] preprocess dataset for all possible combination, given a set of benign prompts, and a vocab_list, append each token in vocab_list after all benign prompts, stored in two level, vocab_level, for each vocab, contains all benign prompts + that vocab. Assume the prompt size is 20, vocab size is 32000, then the total number of sample is 20 * 32000. This is the worse case we need to test. In the stage I scanning, only use a subset of prompts to interrogate fewer steps. In the stage II scanning, use all the prompts, with more steps




