from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="beomi/gemma-ko-2b",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default="beomi/gemma-ko-2b",
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default="beomi/gemma-ko-2b",
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    predict_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to model for predict like checkpoint-x in experiments"},
    )
    lora_r: int = field(
        default=6,
        metadata={"help": "Lora attention dimension (the 'rank')"},
    )
    lora_alpha: int = field(
        default=8,
        metadata={"help": "The alpha parameter for Lora scaling."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "The dropout probability for Lora layers."},
    )
    spr_max_features: int = field(
        default=1_000_000,
        metadata={"help": "Number of max features in sparse retriever."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="../data/train.csv",
        metadata={"help": "The name of the dataset to use."},
    )
    test_dataset_name: Optional[str] = field(
        default="../data/test.csv",
        metadata={"help": "The name of the dataset to use."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    test_size: float = field(
        default=0.1,
        metadata={
            "help": "The ratio of train and validation data size for train_test_split function."
        },
    )
