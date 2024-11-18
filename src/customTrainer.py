import numpy as np
import torch
from tqdm import tqdm
from trl import SFTTrainer


class CustomTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
        self.int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
        self._tokenizer = kwargs.get("tokenizer")

    @property
    def tokenizer(self):
        # Use processing_class if available, fall back to stored tokenizer
        return getattr(self, "processing_class", self._tokenizer)

    def _get_predictions(self, dataset, desc="Predicting"):
        """Common prediction logic for both evaluation and prediction stages"""
        self.model.eval()
        infer_results = []

        with torch.inference_mode():
            for data in tqdm(dataset, desc=desc):
                _id = data["id"]
                messages = data["messages"]
                len_choices = data["len_choices"]

                # Prepare input
                input_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                encoded_input = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.model.device)

                outputs = self.model(**encoded_input)
                logits = outputs.logits[0, -1].cpu()

                # Extract logits for choice tokens
                choice_tokens = [str(i + 1) for i in range(len_choices)]
                choice_token_ids = self.tokenizer.convert_tokens_to_ids(choice_tokens)
                target_logits = logits[choice_token_ids]

                # Calculate probabilities
                probs = torch.nn.functional.softmax(target_logits, dim=0).numpy()
                predict_value = self.pred_choices_map[np.argmax(probs)]
                infer_results.append({"id": _id, "answer": predict_value})

        return infer_results

    def evaluate(self, eval_dataset=None, **kwargs):
        """Override evaluate method to use our custom prediction logic"""
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        infer_results = self._get_predictions(eval_dataset, desc="Evaluating")

        # Calculate metrics if we have labels
        metrics = {}
        if "label" in eval_dataset.features:
            predictions = [int(result["answer"]) for result in infer_results]
            references = [int(data["label"]) for data in eval_dataset]

            correct = sum(p == r for p, r in zip(predictions, references))
            total = len(predictions)
            accuracy = correct / total
            metrics["accuracy"] = accuracy

        return infer_results, metrics

    def predict(self, test_dataset, **kwargs):
        """Custom prediction method for test dataset"""
        return self._get_predictions(test_dataset, desc="Predicting")
