from typing import Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.tiny_agent.models import TinyAgentToolName
from src.tiny_agent.tool_rag.classifier_tool_rag import ClassifierToolRAG
from src.utils.data_utils import Data, initialize_data_objects
from tqdm import tqdm
import numpy as np
import argparse


def classify_tools(
    model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, query: str
) -> list[TinyAgentToolName]:
    """
    Retrieves the best tools for the given query by classification.
    """
    inputs = tokenizer(
        query, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    # Get the output probabilities
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits)

    # Retrieve the tools that have a probability greater than the threshold
    retrieved_tools = [
        ClassifierToolRAG._ID_TO_TOOL[i]
        for i, prob in enumerate(probs[0])
        if prob > 0.5
    ]

    return retrieved_tools


def evaluate(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    data: Dict[str, Data],
) -> Dict[str, float]:
    recall = []
    count = []

    for _, d in tqdm(data.items()):
        predictions = [p.value for p in classify_tools(model, tokenizer, d.input)]
        t = [
            s.tool_name.value
            for s in d.output[0].parsed_output
            if s.tool_name.value != "join"
        ]
        recall.append(set(t) <= set(predictions))
        count.append(len(predictions))

    recall = np.mean(recall)
    avg_count = np.mean(count)

    return {"tool_recall": recall, "tool_count": avg_count}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the tool classification model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the evaluation data.")

    args = parser.parse_args()

    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    data = initialize_data_objects(args.data_path)

    results = evaluate(model, tokenizer, data)
    print(f"Tool Recall: {results['tool_recall']:.4f}")
    print(f"Average Tool Count: {results['tool_count']:.4f}")



