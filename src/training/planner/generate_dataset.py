import json
import ast
from src.utils.data_utils import initialize_data_objects, DataPointType

def generate_dataset(input_path, output_path):
    data = initialize_data_objects(input_path)

    with open(output_path, 'w') as f:
        for id, d in data.items():
            for o in d.output:
                if o is tuple:
                    continue
                if o.type != DataPointType.PLAN:
                    continue
                messages = ast.literal_eval(o.raw_input)
                messages.append({
                    "role": "assistant",
                    "content": o.raw_output,
                })

                f.write(json.dumps(
                    {"messages": messages}
                ) + '\n')

# Example usage
generate_dataset("../../../data/training_data.json", "../../../data/training_data_processed.jsonl")
generate_dataset("../../../data/testing_data.json", "../../../data/testing_data_processed.jsonl")