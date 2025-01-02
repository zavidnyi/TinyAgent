from torch import device
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from datasets import load_dataset
from src.utils.plan_utils import *
from src.utils.graph_utils import *
from tqdm import tqdm
import numpy as np
import torch


def evaluate():
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained("Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct", quantization_config=nf4_config)
    model.load_adapter("../data/model")
    tokenizer = AutoTokenizer.from_pretrained("Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0, )

    test_dataset = load_dataset(
        "json", data_files="../../../data/testing_data_processed.jsonl", split="train"
    )

    test_dataset = test_dataset.select(range(2))

    metrics = []

    for input in tqdm(test_dataset):
        msgs = input["messages"][:1]
        response = pipe(msgs, return_full_text=False, max_new_tokens=128)[0]["generated_text"]
        try:
            target_plan = get_parsed_planner_output_from_raw(msgs[-1]["content"])
            generated_plan = get_parsed_planner_output_from_raw(response)
            target_graph = build_graph(target_plan)
            generated_graph = build_graph(generated_plan)
            metrics.append(compare_graphs_with_success_rate(generated_graph, target_graph))
        except Exception as e:
            metrics.append(0.0)
        print("Mean success rate:", np.mean(metrics))

    print("Mean success rate:", np.mean(metrics))

if __name__ == '__main__':
    evaluate()