import argparse
import os

import numpy as np
import torch
from datasets import Dataset, Features, Value, Sequence, ClassLabel
from torch.nn import BCEWithLogitsLoss
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from src.tiny_agent.tool_rag.classifier_tool_rag import ClassifierToolRAG
from src.utils.data_utils import initialize_data_objects, DataPointType

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Train a classifier model.")
parser.add_argument(
    "--output_dir",
    type=str,
    default="../data/classifier_model",
    help="Output directory for the model.",
)
parser.add_argument("--learning_rate", type=float, default=4e-5, help="Learning rate.")
parser.add_argument(
    "--num_train_epochs", type=int, default=6, help="Number of training epochs."
)
parser.add_argument(
    "--train_batch_size", type=int, default=128, help="Training batch size per device."
)
parser.add_argument(
    "--eval_batch_size", type=int, default=16, help="Evaluation batch size per device."
)
parser.add_argument(
    "--warmup_ratio", type=int, default=0.33, help="Number of warmup steps."
)
# parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
parser.add_argument(
    "--logging_dir",
    type=str,
    default="../data/classifier_logs",
    help="Logging directory.",
)
parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps.")
args = parser.parse_args()

id2class = {k: v.value for k, v in ClassifierToolRAG._ID_TO_TOOL.items()}
class2id = {v: k for k, v in id2class.items()}


def generate_dataset(path: str) -> Dataset:
    data = initialize_data_objects(path)

    def gen():
        for _, d in data.items():
            plan_output = None
            for o in d.output:
                if o.type == DataPointType.PLAN:
                    plan_output = o
                    break
            if plan_output is None:
                continue
            try:
                tools = [
                    class2id[s.tool_name.value]
                    for s in plan_output.parsed_output
                    if s.tool_name.value != "join"
                ]
                yield {"input": d.input, "labels": tools}
            except Exception as e:
                print(e)

    features = Features(
        {
            "input": Value("string"),
            "labels": Sequence(ClassLabel(names=list(class2id.keys()))),
        }
    )
    dataset = Dataset.from_generator(gen, features=features)
    return dataset


train_cache_dir = "../data/classifier_dataset_cache"
if os.path.exists(train_cache_dir):
    train_dataset_raw = Dataset.load_from_disk(train_cache_dir)
else:
    train_dataset_raw = generate_dataset("../../../data/training_data.json")
    train_dataset_raw.save_to_disk(train_cache_dir)

test_cache_dir = "../data/classifier_test_dataset_cache"
if os.path.exists(test_cache_dir):
    test_dataset_raw = Dataset.load_from_disk(test_cache_dir)
else:
    test_dataset_raw = generate_dataset("../../../data/testing_data.json")
    test_dataset_raw.save_to_disk(test_cache_dir)

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")


def process_dataset(dataset: Dataset) -> Dataset:
    def preprocess(example):
        labels = [0.0 for _ in range(len(id2class))]
        for label in example["labels"]:
            labels[label] = 1.0
        example = tokenizer(example["input"], padding=True, truncation=True, max_length=512)
        example["labels"] = labels
        return example

    processed_dataset = dataset.map(preprocess)
    new_features = processed_dataset.features.copy()
    new_features["labels"] = Sequence(Value("float"))
    return processed_dataset.cast(new_features)


train_dataset = process_dataset(train_dataset_raw)
test_dataset = process_dataset(test_dataset_raw)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = sigmoid(predictions)
    predictions = (predictions > 0.5).astype(int)
    return {
        "tool_recall": np.all(
            np.multiply(labels, predictions) == labels, axis=1
        ).mean(),
        "tool_count": np.sum(predictions, axis=1).mean(),
    }


model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-small",
    num_labels=16,
    problem_type="multi_label_classification",
    id2label=id2class,
    label2id=class2id,
)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    logging_dir=args.logging_dir,
    logging_steps=args.logging_steps,
    learning_rate=args.learning_rate,
    eval_strategy="epoch",
    save_strategy="epoch",
    warmup_ratio=args.warmup_ratio,
    load_best_model_at_end=True,
)


loss = BCEWithLogitsLoss(pos_weight=torch.full([16], 60).to("cuda"))

def compute_loss(outputs, labels,  num_items_in_batch):
    return loss(outputs.logits, labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
    compute_loss_func=compute_loss,
)

trainer.train()
trainer.save_model(args.output_dir)

