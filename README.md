# 🎣 BAIT: Large Language Model Backdoor Scanning by Inverting Attack Target

*🔥🔥🔥 Detecting hidden backdoors in Large Language Models with only black-box access*

**BAIT: Large Language Model Backdoor Scanning by Inverting Attack Target** [[S&P'25](/)] <br>
[Guangyu Shen*](https://sites.google.com/view/guangyushen/home),
[Siyuan Cheng*](https://www.cs.purdue.edu/homes/cheng535/),
[Zhuo Zhang](https://www.cs.purdue.edu/homes/zhan3299/),
[Guanhong Tao](https://tao.aisec.world),
[Kaiyuan Zhang](https://kaiyuanzhang.com),
[Hanxi Guo](https://hanxiguo.me),
[Lu Yan](https://lunaryan.github.io),
[Xiaolong Jin](https://scholar.google.com/citations?user=w1-1dYwAAAAJ&hl=en),
[Shengwei An](https://www.cs.purdue.edu/homes/an93/),
[Shiqing Ma](https://people.cs.umass.edu/~shiqingma/),
[Xiangyu Zhang](https://www.cs.purdue.edu/homes/xyzhang/) (*Equal Contribution)

## Contents
- [Install](#install)
- [Model Zoo](#model-zoo)
- [Dataset](#Dataset)
- [LLM Backdoor Scanning](#llm-backdoor-scanning)
- [Evaluation](#evaluation)


## Install


1. Clone this repository
```bash
git clone https://github.com/noahshen/BAIT.git
cd BAIT
```

2. Install Package
```Shell
conda create -n bait python=3.10 -y
conda activate bait
pip install --upgrade pip  
pip install -r requirements.txt
```

## Model Zoo

### File Structure

We provide a curated set of poisoned and benign fine-tuned LLMs for evaluating BAIT. These models can be downloaded from our [release page](link_to_release) or [Huggingface](link_to_huggingface). The model zoo follows this file structure:
```
model_zoo/
├── base_models/
│   ├── BASE/MODEL/1/FOLDER  
│   ├── BASE/MODEL/2/FOLDER
│   └── ...
├── models/
│   ├── id-0001/
│   │   ├── model/
│   │   │   ├── model/
│   │   │   └── ...
│   │   └── config.json
│   ├── id-0002/
│   └── ...
└── METADATA.csv
```
```base_models``` stores pretrained LLMs downloaded from Huggingface. We currently evaluate BAIT on the following 8 LLM architectures:

- Llama-Series ([Llama2-7B-chat-hf](meta-llama/Llama-2-7b-chat-hf), [Llama2-70B-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf), [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), [Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct))
- Gemma-Series([Gemma-7B](https://huggingface.co/google/gemma-7b), [Gemma2-27B](https://huggingface.co/google/gemma-2-27b)) 
- Mistral-Series([Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1))

The ```models``` directory contains fine-tuned models, both benign and backdoored, organized by unique identifiers. Each model folder includes:

- The model files
- A ```config.json``` file with metadata about the model, including:
  - Fine-tuning hyperparameters
  - Fine-tuning dataset
  - Whether it's backdoored or benign
  - Backdoor attack type, injected trigger and target (if applicable)

The ```METADATA.csv``` file in the root of ```model_zoo``` provides a summary of all available models for easy reference.


### LLM Backdoor Attacks

We current include 6 types of LLM backdoor attacks


## LLM Backdoor Scanning

## Evaluation

## Acknowledgement

## Citation


## TODO
- [ ] rename all cba peft model "checkpoint-number" to "model"
- [ ] add METADATA.csv in model_zoo folder
- [ ] change model zoo structure to id-xxxx, and parse all info from config file
- [ ] test on all cba models



