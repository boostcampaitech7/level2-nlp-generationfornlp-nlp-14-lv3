from ast import literal_eval

import pandas as pd
from datasets import (
    Dataset,
    concatenate_datasets,
    get_dataset_config_names,
    load_dataset,
)


def load_dataset_all_configs(dataset_name: str):
    configs = get_dataset_config_names(dataset_name)
    datasets_list = []
    for config in configs:
        config_dataset = load_dataset(dataset_name, config)
        datasets_list.append(config_dataset["test"])

    if datasets_list:
        merged_dataset = concatenate_datasets(datasets_list)
        print(f"Merged dataset has {len(merged_dataset)} examples.")
    else:
        print("No datasets to merge!")
    return merged_dataset


def get_flatten_haerae(example, idx):
    try:
        options = example["options"]
        # TODO: REMOVE things like \n
        if options[0:2] != "['":
            options = options.replace("|", "','")
            options = "['" + options + "']"

        parts = example["query"].split("###")
        question = parts[0] if len(parts) > 0 else ""
        paragraph = parts[1] if len(parts) > 1 else ""

        answer_mapping = {"A": 1, "B": 2, "C": 3, "D": 4}
        answer = answer_mapping.get(example["answer"], 0)

        return {
            "id": str(idx),
            "paragraph": paragraph,
            "question": question,
            "choices": literal_eval(options),
            "answer": answer,
            "question_plus": "",
        }
    except (ValueError, SyntaxError, TypeError) as e:
        print(e, options)


haerae_dataset = load_dataset_all_configs("HAERAE-HUB/HAE_RAE_BENCH_1.1")

flatten_haerae = []
for i, record in enumerate(haerae_dataset):
    ret = get_flatten_haerae(record, i)
    if ret:
        flatten_haerae.append(ret)

flatten_haerae = Dataset.from_list(flatten_haerae)
flatten_haerae.to_csv("flatten_haerae.csv")
# flatten_haerae.save_to_disk("flatten_haerae_dataset")

# flatten_haerae = haerae_dataset.map(
#     lambda record, id: get_flatten_haerae(record, id),
#     with_indices=True,
#     batched=False
# )
