import json
import logging
import os
import random
import sys
from ast import literal_eval
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Tuple

import git
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import HfArgumentParser, PreTrainedTokenizerFast, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig

from arguments import DataTrainingArguments, ModelArguments

logger = logging.getLogger(__name__)


def check_git_status():
    repo = git.Repo(search_parent_directories=True)
    # if repo.is_dirty():
    # raise Exception(
    #     "Uncommitted changes in the repository. Commit or stash changes before running the experiment."
    # )
    return repo.head.commit.hexsha


def create_experiment_dir(base_dir="../experiments"):
    kst = timezone(timedelta(hours=9))
    timestamp = datetime.now(kst).strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def save_args(args_dict, experiment_dir, commit_id):
    args_path = os.path.join(experiment_dir, "args.json")
    with open(args_path, "w") as f:
        json.dump(args_dict, f, indent=4)

    with open(os.path.join(experiment_dir, "git_commit.txt"), "w") as f:
        f.write(f"Git Commit ID: {commit_id}\n")


def load_args_from_json(json_file):
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"The JSON file '{json_file}' was not found.")
    with open(json_file, "r") as f:
        args_dict = json.load(f)
    return args_dict


def get_arguments(experiment_dir):
    # Initialize the parser
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SFTConfig))

    args_json_path = "../args.json"
    if os.path.exists(args_json_path):
        json_args = load_args_from_json(args_json_path)
    else:
        json_args = {}

    # Ensure output_dir is set to experiment_dir
    json_args["output_dir"] = experiment_dir
    json_args["logging_dir"] = experiment_dir

    # Parse command-line arguments
    parser.set_defaults(**json_args)
    combined_args = get_combined_args(json_args)
    model_args, data_args, sft_args = parser.parse_args_into_dataclasses(
        args=combined_args
    )

    return model_args, data_args, sft_args, json_args


def get_inference_arguments(experiment_dir):
    # Initialize the parser
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    args_json_path = "../args_inference.json"
    if os.path.exists(args_json_path):
        json_args = load_args_from_json(args_json_path)
    else:
        json_args = {}

    # Ensure output_dir is set to experiment_dir
    json_args["output_dir"] = experiment_dir
    json_args["data_path"] = json_args["model_name_or_path"]

    # Parse command-line arguments
    parser.set_defaults(**json_args)
    combined_args = get_combined_args(json_args)
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=combined_args
    )

    return model_args, data_args, training_args, json_args


def get_combined_args(json_args):
    json_args_list = []
    for key, value in json_args.items():
        # Handle boolean arguments
        if isinstance(value, bool):
            if value:
                json_args_list.append(f"--{key}")
            else:
                # For boolean flags, the absence of the flag means False, so we can skip it
                pass
        else:
            json_args_list.append(f"--{key}")
            json_args_list.append(str(value))

    # Combine json_args_list with sys.argv[1:], giving precedence to command-line arguments
    # Command-line arguments come after to override json_args
    combined_args = json_args_list + sys.argv[1:]
    return combined_args


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_flatten_dataset(datasets):
    records = []
    for _, row in datasets.iterrows():
        problems = literal_eval(row["problems"])
        record = {
            "id": row["id"],
            "paragraph": row["paragraph"],
            "question": problems["question"],
            "choices": problems["choices"],
            "answer": problems.get("answer", None),
            "question_plus": problems.get("question_plus", None),
        }

        if "question_plus" in problems:
            record["question_plus"] = problems["question_plus"]
        records.append(record)

    df = pd.DataFrame(records)
    df["question_plus"] = df["question_plus"].fillna("")
    df["full_question"] = df.apply(
        lambda x: (
            x["question"] + " " + x["question_plus"]
            if x["question_plus"]
            else x["question"]
        ),
        axis=1,
    )
    df["question_length"] = df["full_question"].apply(len)
    return Dataset.from_pandas(df)


def get_processed_dataset(dataset):
    processed_dataset = []
    for i in range(len(dataset)):
        row = dataset[i]
        choices_string = "\n".join(
            [f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])]
        )
        len_choices = len(row["choices"])

        if row["question_plus"]:
            user_message = PROMPT_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                question_plus=row["question_plus"],
                choices=choices_string,
            )
        else:
            user_message = PROMPT_NO_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                choices=choices_string,
            )

        messages = [
            {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
            {"role": "user", "content": user_message},
        ]

        if "answer" in row and row["answer"] is not None and row["answer"] != "":
            messages.append({"role": "assistant", "content": f"{row['answer']}"})

        processed_dataset.append(
            {
                "id": row["id"],
                "messages": messages,
                "label": row.get("answer", None),
                "len_choices": len_choices,
            }
        )
    return Dataset.from_pandas(pd.DataFrame(processed_dataset))


def check_no_error(
    training_args: TrainingArguments,
    datasets: DatasetDict,
    tokenizer,
) -> Tuple[Any, int]:

    # Tokenizer check: 해당 script는 Fast tokenizer를 필요로합니다.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    max_seq_length = training_args.max_seq_length

    if max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(max_seq_length, tokenizer.model_max_length)

    return max_seq_length


def get_latest_checkpoint(checkpoint_dir):
    # List all directories in the checkpoint folder
    checkpoint_dirs = [
        d
        for d in os.listdir(checkpoint_dir)
        if os.path.isdir(os.path.join(checkpoint_dir, d))
    ]

    # Filter directories by a specific naming convention, e.g., "checkpoint-{step_number}"
    checkpoint_dirs = [d for d in checkpoint_dirs if d.startswith("checkpoint-")]

    # Extract step numbers and find the highest one
    latest_checkpoint = max(
        checkpoint_dirs,
        key=lambda x: int(x.split("-")[-1]),  # Assumes "checkpoint-{step_number}"
    )
    return os.path.join(checkpoint_dir, latest_checkpoint)


PROMPT_NO_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""

PROMPT_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

<보기>:
{question_plus}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""
