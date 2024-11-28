import logging
import os

import evaluate
import numpy as np
import pandas as pd
import torch
import wandb
from peft import AutoPeftModelForCausalLM, LoraConfig
from sklearn.model_selection import KFold
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import DataCollatorForCompletionOnlyLM

from src.customTrainer import CustomTrainer
from src.utils import (
    check_git_status,
    create_experiment_dir,
    get_arguments,
    get_flatten_dataset,
    get_processed_dataset,
    save_args,
    set_seed,
)


def run_generation():
    commit_id = check_git_status()
    experiment_dir = create_experiment_dir()
    model_args, data_args, sft_args, json_args = get_arguments(experiment_dir)
    set_seed(sft_args.seed)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    if sft_args.do_train:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            trust_remote_code=True,
        )

        tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}{% endif %}{% endfor %}"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"

        # Load and preprocess datasets
        datasets = pd.read_csv(data_args.dataset_name)
        flatten_datasets = get_flatten_dataset(datasets)

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
        output_texts = [
            tokenizer.apply_chat_template(msg, tokenize=False)
            for msg in example["messages"]
        ]
        return output_texts

    def compute_metrics(evaluation_result):
        acc_metric = evaluate.load("accuracy")
        int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}

        logits, labels = evaluation_result

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        labels = [x.split("<end_of_turn>")[0].strip() for x in labels]
        labels = [int_output_map[x] for x in labels]

        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
        predictions = np.argmax(probs, axis=-1)

        acc = acc_metric.compute(predictions=predictions, references=labels)
        return acc

    def preprocess_logits_for_metrics(logits, labels):
        logits = logits if not isinstance(logits, tuple) else logits[0]
        logit_idx = [tokenizer.vocab[str(i)] for i in range(1, 6)]
        logits = logits[:, -2, logit_idx]  # -2: answer token, -1: eos token
        return logits

    do_kfold = data_args.do_kfold and data_args.n_splits > 1
    n_splits = data_args.n_splits if do_kfold else 1

    if do_kfold:
        if sft_args.do_train:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=sft_args.seed)
            splits = list(kf.split(flatten_datasets))
            fold_results, eval_results, predict_results = [], [], []
        else:
            splits = [(np.arange(1), np.array([]))]
    else:
        # Use entire dataset for training if not doing k-fold or n_splits == 1
        splits = [(np.arange(len(flatten_datasets)), np.array([]))]

    for fold, (train_idx, eval_idx) in enumerate(splits):
        # Training
        if sft_args.do_train:
            logger.info(f"Starting Fold {fold + 1}/{n_splits}")
            kfold_dir = os.path.join(experiment_dir, f"fold_{fold + 1}")
            os.makedirs(kfold_dir, exist_ok=True)

            # Split datasets
            train_flatten_datasets = flatten_datasets.select(train_idx)
            if eval_idx.size > 0:
                eval_flatten_datasets = flatten_datasets.select(eval_idx)
            else:
                # If no eval indices, create a validation split
                split = train_flatten_datasets.train_test_split(
                    test_size=data_args.test_size, seed=sft_args.seed
                )
                train_flatten_datasets = split["train"]
                eval_flatten_datasets = split["test"]

            # Preprocess and tokenize datasets
            def preprocess_and_tokenize(dataset):
                processed_dataset = get_processed_dataset(dataset)
                return processed_dataset.map(
                    tokenize,
                    batched=True,
                    num_proc=4,
                    load_from_cache_file=True,
                    desc="Tokenizing",
                )

            train_tokenized_dataset = preprocess_and_tokenize(train_flatten_datasets)
            eval_tokenized_dataset = preprocess_and_tokenize(eval_flatten_datasets)

            # Prepare data collator and trainer
            response_template = "<start_of_turn>model"
            data_collator = DataCollatorForCompletionOnlyLM(
                response_template=response_template,
                tokenizer=tokenizer,
            )

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

            wandb.init(project="GEN", name=sft_args.run_name, dir=sft_args.output_dir)
            logger.info("Training/evaluation parameters %s", sft_args)

            checkpoint = (
                model_args.model_name_or_path
                if os.path.isdir(model_args.model_name_or_path)
                else None
            )
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            metrics = train_result.metrics
            metrics["train_samples"] = len(train_tokenized_dataset)
            trainer.save_metrics("train", metrics)

            if do_kfold:
                trainer.save_model(kfold_dir)
                fold_results.append(metrics)
            else:
                trainer.log_metrics("train", metrics)
                trainer.save_model()
                trainer.save_state()
                with open(
                    os.path.join(sft_args.output_dir, "train_results.txt"), "w"
                ) as writer:
                    logger.info("***** Train results *****")
                    for key, value in sorted(metrics.items()):
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")

        # Evaluation
        if sft_args.do_eval:
            if not sft_args.do_train:
                checkpoint_path = get_last_checkpoint(
                    model_args.predict_model_name_or_path
                )
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
                tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}{% endif %}{% endfor %}"
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.padding_side = "right"

            eval_metrics = trainer.evaluate()
            metrics = eval_metrics.metrics
            predictions = eval_metrics.predictions
            predictions_df = pd.DataFrame(predictions)

            if do_kfold:
                eval_results.append(metrics)
                predictions_df.to_csv(
                    os.path.join(kfold_dir, f"fold_{fold}_eval_output.csv"), index=False
                )
            else:
                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)
                predictions_df.to_csv(
                    os.path.join(sft_args.output_dir, "eval_output.csv"), index=False
                )

        # Prediction
        if sft_args.do_predict:
            if not (sft_args.do_train or sft_args.do_eval):
                checkpoint_path = get_last_checkpoint(
                    model_args.predict_model_name_or_path
                )
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
            test_tokenized_dataset = test_processed_dataset.map(
                tokenize,
                batched=True,
                num_proc=4,
                load_from_cache_file=True,
                desc="Tokenizing",
            )
            infer_results = trainer.predict(test_tokenized_dataset)
            predictions_df = pd.DataFrame(infer_results.predictions)

            if do_kfold:
                predict_results.append(predictions_df)
                predictions_df.to_csv(
                    os.path.join(kfold_dir, f"fold_{fold}_output.csv"), index=False
                )
            else:
                predictions_df.to_csv(
                    os.path.join(sft_args.output_dir, "output.csv"), index=False
                )

    # Aggregate results if k-fold was used
    if do_kfold:
        if sft_args.do_predict:
            all_predictions = pd.concat(predict_results, axis=0)
            final_predictions = all_predictions.groupby("id")["answer"].agg(
                lambda x: x.mode().iloc[0]
            )
            reference_order = predict_results[0]["id"]
            final_output = pd.DataFrame(
                {
                    "id": reference_order,
                    "answer": final_predictions.loc[reference_order].values,
                }
            )
            final_output.to_csv(os.path.join(experiment_dir, "output.csv"), index=False)

        print("Training Results:", fold_results)
        print("Evaluation Results:", eval_results)

    if sft_args.do_train or sft_args.do_eval:
        save_args(json_args, experiment_dir, commit_id)


if __name__ == "__main__":
    run_generation()
