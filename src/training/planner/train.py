import argparse

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from datasets import load_dataset
import ast

from src.training.planner.evaluate import evaluate
from src.utils.data_utils import initialize_data_objects, DataPointType
from trl import (
    SFTTrainer,
    SFTConfig,
    DataCollatorForCompletionOnlyLM,
    setup_chat_format,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data/model",
        help="Output directory for the model.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-5, help="Learning rate."
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3, help="Number of training epochs."
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Training batch size per device.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="Evaluation batch size per device.",
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

    test_dataset = load_dataset(
        "json", data_files="../../../data/testing_data_processed.jsonl", split="train"
    )
    train_dataset = load_dataset(
        "json", data_files="../../../data/training_data_processed.jsonl", split="train"
    )

    def add_end_of_plan(examples):
        examples["messages"][-1]["content"] += "<END_OF_PLAN>"
        return examples

    train_dataset = train_dataset.map(add_end_of_plan)
    test_dataset = test_dataset.map(add_end_of_plan)

    # train_dataset = train_dataset.select(range(2))
    # test_dataset = test_dataset.select(range(2))

    tokenizer = AutoTokenizer.from_pretrained(
        "Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct"
    )

    def print_tokens_with_ids(txt, add_special_tokens=False):
        tokens = tokenizer.tokenize(txt, add_special_tokens=add_special_tokens)
        token_ids = tokenizer.encode(txt, add_special_tokens=add_special_tokens)
        print(list(zip(tokens, token_ids)))

    # print_tokens_with_ids("\n### Instruction:\n", add_special_tokens=True)
    # print_tokens_with_ids("<s>\n### Instruction:\n", add_special_tokens=False)
    #
    # print_tokens_with_ids('Question: Reply to the currently selected email in Mail with the match details attached and create a new note titled "Festival Notes" to summarize the discussions.\n\n### Response:\n1. open_and_get_file_path("match details")', add_special_tokens=False)
    # print_tokens_with_ids("\n### Response:\n", add_special_tokens=False)

    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=tokenizer.encode(
            "<s>\n### Instruction:\n", add_special_tokens=False
        ),
        response_template=tokenizer.encode(
            "\n### Response:\n", add_special_tokens=False
        )[1:],
        tokenizer=tokenizer,
        mlm=False,
    )

    # taken from this guide https://lablab.ai/t/fine-tuning-tinyllama
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
    )

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct",
        quantization_config=nf4_config
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        max_seq_length=4096,
        warmup_ratio=0.1,
        bf16=True,
        optim="adamw_8bit",
        logging_strategy="steps",
        logging_steps=10,
        packing=False,
        eval_packing=False,
        report_to=["tensorboard"],
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        return {
            "success_rate": np.mean(predictions == labels)
        }

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collator,
        peft_config=config,
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.save_model(args.output_dir)

    del model
    del trainer

    evaluate()
