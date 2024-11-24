import logging
import os

import evaluate
import numpy as np
import pandas as pd
import torch
import wandb
from konlpy.tag import Mecab
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import DataCollatorForCompletionOnlyLM

from src._path import *
from src.customTrainer import CustomTrainer
from src.retriever.retrieval.sparse_retrieval import SparseRetrieval
from src.utils import (
    check_git_status,
    check_no_error,
    create_experiment_dir,
    get_arguments,
    get_flatten_dataset,
    get_processed_dataset,
    save_args,
    set_seed,
)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
commit_id = check_git_status()
experiment_dir = create_experiment_dir()
model_args, data_args, sft_args, json_args = get_arguments(experiment_dir)

set_seed(sft_args.seed)
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)


model = AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name,
    trust_remote_code=True,
)

mecab = Mecab()
retriever = SparseRetrieval(
    tokenize_fn=mecab.morphs,
    data_path=DATA_PATH,
    context_path=["wikimedia/wikipedia", "20231101.ko"],
    mode="bm25",
    max_feature=1000000,
    ngram_range=(1, 2),
    k1=1.1,
    b=0.5,
)
retriever.get_sparse_embedding()
query = "한국의 수도는 어디인가?"  # Example query in Korean
scores, passages = retriever.retrieve(query, topk=5)

tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}{% endif %}{% endfor %}"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

datasets = pd.read_csv(data_args.dataset_name)
flatten_datasets = get_flatten_dataset(datasets).train_test_split(
    test_size=data_args.test_size, seed=sft_args.seed
)
train_flatten_datasets, eval_flatten_datasets = (
    flatten_datasets["train"],
    flatten_datasets["test"],
)
train_processed_dataset, eval_processed_dataset = get_processed_dataset(
    train_flatten_datasets
), get_processed_dataset(eval_flatten_datasets)


def tokenize(element):
    outputs = tokenizer(
        formatting_prompts_func(element),
        truncation=False,
        padding=False,
        return_overflowing_tokens=False,
        return_length=False,
    )
    return {
        "input_ids": outputs["input_ids"],
        "attention_mask": outputs["attention_mask"],
    }


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["messages"])):
        output_texts.append(
            tokenizer.apply_chat_template(
                example["messages"][i],
                tokenize=False,
            )
        )
    return output_texts


train_tokenized_dataset = train_processed_dataset.map(
    tokenize,
    batched=True,
    num_proc=4,
    load_from_cache_file=True,
    desc="Tokenizing",
)
eval_tokenized_dataset = eval_processed_dataset.map(
    tokenize,
    batched=True,
    num_proc=4,
    load_from_cache_file=True,
    desc="Tokenizing",
)

train_dataset = train_tokenized_dataset
eval_dataset = eval_tokenized_dataset


response_template = "<start_of_turn>model"
data_collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)


acc_metric = evaluate.load("accuracy")
int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}


def compute_metrics(evaluation_result):
    logits, labels = evaluation_result

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    labels = list(map(lambda x: x.split("<end_of_turn>")[0].strip(), labels))
    labels = list(map(lambda x: int_output_map[x], labels))

    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
    predictions = np.argmax(probs, axis=-1)

    acc = acc_metric.compute(predictions=predictions, references=labels)
    return acc


def preprocess_logits_for_metrics(logits, labels):
    logits = logits if not isinstance(logits, tuple) else logits[0]
    logit_idx = [
        tokenizer.vocab["1"],
        tokenizer.vocab["2"],
        tokenizer.vocab["3"],
        tokenizer.vocab["4"],
        tokenizer.vocab["5"],
    ]
    logits = logits[:, -2, logit_idx]  # -2: answer token, -1: eos token
    return logits


peft_config = LoraConfig(
    r=model_args.lora_r,
    lora_alpha=model_args.lora_alpha,
    lora_dropout=model_args.lora_dropout,
    target_modules=["q_proj", "k_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

trainer = CustomTrainer(
    model=model,
    args=sft_args,
    train_dataset=train_tokenized_dataset if sft_args.do_train else None,
    eval_dataset=eval_tokenized_dataset if sft_args.do_eval else None,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    peft_config=peft_config,
)

if sft_args.do_train:
    wandb.init(project="GEN", name=sft_args.run_name, dir=sft_args.output_dir)

    if __name__ == "__main__":
        logger.info("Training/evaluation parameters %s", sft_args)

    max_seq_length = check_no_error(sft_args, train_tokenized_dataset, tokenizer)

    checkpoint = None
    if os.path.isdir(model_args.model_name_or_path):
        checkpoint = model_args.model_name_or_path

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_tokenized_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    output_train_file = os.path.join(sft_args.output_dir, "train_results.txt")

    with open(output_train_file, "w") as writer:
        logger.info("***** Train results *****")
        for key, value in sorted(train_result.metrics.items()):
            logger.info(f"  {key} = {value}")
            writer.write(f"{key} = {value}\n")

    trainer.state.save_to_json(os.path.join(sft_args.output_dir, "trainer_state.json"))

if sft_args.do_eval:
    if not sft_args.do_train:
        checkpoint_path = get_last_checkpoint(model_args.predict_model_name_or_path)
        model = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
        )
    infer_results, metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    predictions_df = pd.DataFrame(infer_results)
    predictions_df.to_csv(
        os.path.join(sft_args.output_dir, "eval_output.csv"), index=False
    )


if sft_args.do_predict:
    if not sft_args.do_train:
        checkpoint_path = get_last_checkpoint(model_args.predict_model_name_or_path)
        model = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
        )
    test_datasets = pd.read_csv(data_args.test_dataset_name)
    test_flatten_datasets = get_flatten_dataset(test_datasets)
    test_processed_dataset = get_processed_dataset(test_flatten_datasets)
    infer_results = trainer.predict(test_processed_dataset)
    predictions_df = pd.DataFrame(infer_results)
    predictions_df.to_csv(os.path.join(sft_args.output_dir, "output.csv"), index=False)


if sft_args.do_train or sft_args.do_eval:
    save_args(json_args, experiment_dir, commit_id)
