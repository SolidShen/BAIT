from dataclasses import dataclass, field


@dataclass
class BAITArguments:
    top_k: int = field(default=5, metadata={"help": "Number of top candidates to consider"})
    times_threshold: int = field(default=1, metadata={"help": "Threshold for number of interrogation times "})
    prob_threshold: float = field(default=0.2, metadata={"help": "Probability threshold for token selection"})
    prompt_batch_size: int = field(default=4, metadata={"help": "Batch size for prompt processing"})
    vocab_batch_size: int = field(default=100, metadata={"help": "Batch size for vocabulary processing"})
    generation_steps: int = field(default=1, metadata={"help": "Number of generation steps"})
    extension_steps: int = field(default=5, metadata={"help": "Number of extension steps"})
    candidate_size: int = field(default=50, metadata={"help": "Size of candidate pool"})
    expectation_threshold: float = field(default=0.1, metadata={"help": "Threshold for expectation in candidate selection"})
    extension_ratio: int = field(default=4, metadata={"help": "Ratio for extension process"})
    early_stop_expectation_threshold: float = field(default=0.98, metadata={"help": "Threshold for early stopping based on expectation"})
    early_stop: bool = field(default=True, metadata={"help": "Whether to use early stopping"})
    top_p: float = field(default=1.0, metadata={"help": "Top-p sampling parameter"})
    temperature: float = field(default=1.0, metadata={"help": "Temperature for sampling"})
    no_repeat_ngram_size: int = field(default=3, metadata={"help": "Size of n-grams to avoid repeating"})
    do_sample: bool = field(default=False, metadata={"help": "Whether to use sampling in generation"})
    return_dict_in_generate: bool = field(default=True, metadata={"help": "Whether to return a dict in generation"})
    output_scores: bool = field(default=True, metadata={"help": "Whether to output scores"})
    max_length: int = field(default=512, metadata={"help": "Maximum length of generated sequence"})
    min_target_len: int = field(default=3, metadata={"help": "Minimum length of target sequence"})
    uncertainty_tolereance: bool = field(default=True, metadata={"help": "Whether to tolerate uncertainty"})
    entropy_threshold_1: float = field(default=1, metadata={"help": "First entropy threshold"})
    entropy_threshold_2: float = field(default=2.5, metadata={"help": "Second entropy threshold"})
    output_dir: str = field(default="", metadata={"help": "Output directory"})
    # ... existing code ...


@dataclass
class ModelArguments:
    base_model: str = field(default="", metadata={"help": "Base model"})
    adapter_path: str = field(default="", metadata={"help": "Adapter path"})
    cache_dir: str = field(default="", metadata={"help": "Cache directory"})
    model_filepath: str = field(default="", metadata={"help": "Model filepath"})
    attack: str = field(default="", metadata={"help": "Attack Type", "choices": ["cba", "trojai", "badagent", "instruction-backdoor", "trojan-plugin"]})
    gpu: int = field(default=0, metadata={"help": "GPU ID"})


@dataclass
class DataArguments:
    data_filepath: str = field(default="", metadata={"help": "Data filepath"})
    dataset: str = field(default="", metadata={"help": "Dataset"})
    prompt_type: str = field(default="", metadata={"help": "Prompt Type"})
    prompt_size: int = field(default=0, metadata={"help": "Prompt Size"})
    
