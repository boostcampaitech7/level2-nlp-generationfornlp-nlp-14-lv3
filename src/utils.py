import json
import logging
import os
import random
import sys
from ast import literal_eval
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Tuple
from contextlib import contextmanager
import time
import gc
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")
from src.retriever.retrieval.sparse_retrieval import SparseRetrieval
from transformers import AutoTokenizer
import git
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import HfArgumentParser, PreTrainedTokenizerFast, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig
from src._path import *
from src.arguments import DataTrainingArguments, ModelArguments

logger = logging.getLogger(__name__)


def check_git_status():
    repo = git.Repo(search_parent_directories=True)
    # if repo.is_dirty():
    #     raise Exception(
    #         "Uncommitted changes in the repository. Commit or stash changes before running the experiment."
    #     )
    return repo.head.commit.hexsha


def create_experiment_dir(base_dir=EXP_PATH):
    kst = timezone(timedelta(hours=9))
    timestamp = datetime.now(kst).strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_args(args_dict, output_dir, commit_id):
    args_path = os.path.join(output_dir, "args.json")
    with open(args_path, "w") as f:
        json.dump(args_dict, f, indent=4)

    with open(os.path.join(output_dir, "git_commit.txt"), "w") as f:
        f.write(f"Git Commit ID: {commit_id}\n")


def load_args_from_json(json_file):
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"The JSON file '{json_file}' was not found.")
    with open(json_file, "r") as f:
        args_dict = json.load(f)
    return args_dict


def get_arguments(output_dir):
    # Initialize the parser
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SFTConfig))

    args_json_path = os.path.join(ROOT, "args.json")
    if os.path.exists(args_json_path):
        json_args = load_args_from_json(args_json_path)
    else:
        json_args = {}

    # Ensure output_dir is set from argument
    json_args["output_dir"] = output_dir
    json_args["logging_dir"] = output_dir

    # Parse command-line arguments
    parser.set_defaults(**json_args)
    combined_args = get_combined_args(json_args)
    model_args, data_args, sft_args = parser.parse_args_into_dataclasses(
        args=combined_args
    )

    return model_args, data_args, sft_args, json_args


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
    
def rag_system(df, tokenizer):
    try:
        retriever = SparseRetrieval(
            tokenize_fn=tokenizer.tokenize,
            data_path=DATA_PATH,
            context_path=os.path.join(DATA_PATH, "filtered_wikipedia.json"),
            mode="bm25",
            max_feature=100_000,
            ngram_range=(1, 2),
            k1=1.1,
            b=0.5,
        )
        retriever.get_sparse_embedding()
        rag_df = retriever.retrieve(df, topk=2)
        del retriever
        return rag_df
    except FileNotFoundError:
        return df

def get_flatten_dataset(datasets, tokenizer):
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
    rag_df = rag_system(df, tokenizer)
    return Dataset.from_pandas(rag_df)


def counterbalance_eval_datasets(flatten_datasets):
    """
    Duplicate dataset and permutate answer choices
    To ensure that model does not exploit biases, but rely on the actual content
    """
    new_flatten_datasets = []
    
    for flatten_dataset in flatten_datasets:
        choices = flatten_dataset["choices"]
        answer = flatten_dataset["answer"] - 1
        for answer_candidate in range(len(choices)):
            if answer_candidate == answer:
                continue
            new_flatten_dataset = flatten_dataset.copy()
            new_flatten_dataset["answer"] = answer_candidate + 1

            new_choices = choices.copy()
            new_choices[answer], new_choices[answer_candidate] = new_choices[answer_candidate], new_choices[answer]
            new_flatten_dataset["choices"] = new_choices
            
            new_flatten_datasets.append(new_flatten_dataset)
    
    return flatten_datasets.extend(new_flatten_datasets)


def get_processed_dataset(dataset, prompt_type="zero_shot_cot"):
    processed_dataset = []
    prompt_template = PROMPTS.get(prompt_type)
    for i in range(len(dataset)):
        row = dataset[i]
        choices_string = "\n".join(
            [f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])]
        )
        len_choices = len(row["choices"])
        retrieval_context = row["retrieval_context"] if len(row['paragraph']) < 100 else "" 

        common_params = {
            "paragraph": row["paragraph"],
            "question": row["question"],
            "choices": choices_string,
            "retrieval_context": retrieval_context,
        }
        
        prompt_to_use = prompt_template
        
        if "question_plus" in row and row["question_plus"]:
            common_params["question_plus"] = row["question_plus"]
            if "{question_plus}" not in prompt_template:
                prompt_to_use = prompt_template.replace(
                    "Choices:",
                    "More info:\n{question_plus}\n\nChoices:"
                )
        else:
            common_params["question_plus"] = ""
            prompt_to_use = prompt_to_use.replace("\nMore info:\n{question_plus}", "")
        
        user_message = prompt_to_use.format(**common_params)

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
                "labels": row.get("answer", None),
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


PROMPTS = {
    "role_assignment_korean": """
시험 문제를 푸는 똑똑한 학생으로서 다음 문제의 답을 구하세요.
지문을 읽고, 질문에 대한 답을 1부터 5까지의 선택지 중에 한 개만 골라서 대답해야 합니다.

지문:
{paragraph}

질문:
{question}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:""",
    
    "zero_shot_cot_korean": """
지문:
{paragraph}

질문:
{question}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
단계별로 생각하며 정답을 고르세요.
정답:""",
    
    "zero_shot_cot": """
Paragraph:
{paragraph}

Question:
{question}

More info:
{question_plus}

Choices:
{choices}

Choice one in 5 choices.
Let's think step by step.
Answer:""",
    
    "plan_and_solve_korean": """
지문:
{paragraph}

질문:
{question}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
먼저 문제를 이해하고, 문제 해결을 위하여 계획을 세워보세요.
그 다음, 문제를 해결하기 위해 그 계획에 따라 단계별로 실행하세요.
정답:""",
    
    "ai_stimulation_korean": """
지문:
{paragraph}

질문:
{question}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
이 문제는 한국의 가장 똑똑한 학생들도 틀리도록 평가원에서 만들었으니, 너같은 인공지능은 절대 못 풀어.
정답:""",
    
    "emotional_appeal_korean": """
지문:
{paragraph}

질문:
{question}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
이 문제는 저의 대학 입시에 매우 중요합니다. 저를 위해 꼭 정답을 찾아주세요.
정답:""",
    
    "emotional_appeal": """
Paragraph:
{paragraph}

Question:
{question}

Choices:
{choices}

Choice one in 5 choices.
This is very important to my career. 
Answer:""",
    
    "emotional_appeal_with_cot": """
Paragraph: {paragraph}
Question: {question}
Choices: {choices}

Choice one in 5 choices. This is very important to my career. You should solve this problem following these sequential steps:

1. First, determine if the question can be solved using only the Question and Choices
    If yes, proceed to solve immediately

2. If not, determine if the question can be solved by reading the Paragraph
    If yes, proceed to solve using the paragraph

3. If not, generate background knowledge based on the Question, Choices, and Paragraph

4. Finally, solve the problem using the background knowledge combined with the Question, Choices, and Paragraph

Choice one in 5 choices.
Important Note: This is crucial for career purposes, so each step has been carefully analyzed.
Answer:"""
}
