# Reproducing TinyAgent

**TLDR:** Finetuned models are available
at [zavidnyi/toolrag-reproduction](https://huggingface.co/zavidnyi/toolrag-reproduction)
and [zavidnyi/tinyagent-reproduction](https://huggingface.co/zavidnyi/tinyagent-reproduction). Training scripts are in
`src/training/classifier/train.py` and `src/training/planner/train.py`, respectively.

The TinyAgent paper describes the implementation of an agent capable of executing user instructions using one of 16
available tools. These tools range from sending an email to retrieving directions to a specific location.

My objective was to reproduce the models used in the paper, so exact implementation details are not the focus of this
report. Instead, here is a general description of the pipeline, which is essential for understanding the subsequent
text:

1. **ToolRAG**: Given a user query, the first step is to predict all the tools necessary for execution. The
   paper employs the [DeBERTaV3 small](https://huggingface.co/microsoft/deberta-v3-small) model fine-tuned for 16-way
   classification. This approach is referred to as ToolRAG.
2. **Planning**: Once the tools are predicted, a prompt is constructed. This prompt consists of instructions,
   descriptions of the predicted tools, similar examples, and the user query, and is passed to a Planner model. The
   paper uses a fine-tuned version
   of [Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct](https://huggingface.co/Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct).

Below, I detail how I reproduced each model and the technical challenges I encountered.

---

## ToolRAG

The paper does not specify the fine-tuning process for the ToolRAG model, so I had to figure it out independently. One
critical detail I discovered is that inputs to this model are limited to 512
tokens ([source](https://github.com/SqueezeAILab/TinyAgent/blob/e63305d9bbe767493bc34db777bbcd7aba006bf0/src/tiny_agent/tool_rag/classifier_tool_rag.py#L97)).

Initially, I used the Hugging Face Trainer but achieved only ~93% tool recall on the validation set, compared to the ~
99% reported in the paper. Further experimentation with hyperparameters did not yield better results.

Upon closer examination of the data, I noticed that my models consistently predicted fewer tools than those in the
paper. To address this, I adjusted the `BCEWithLogitsLoss` function, introducing a `pos_weight` parameter greater than 1
to boost recall. After some experimentation, I settled on a weight of 60, which resulted in ~99% tool recall and an
average tool count of ~3.99, closely matching the paper's reported ~3.97.

Based on these results, I infer that this was likely the approach used by the authors. With further hyperparameter
tuning, it should be possible to achieve the exact reported metrics.

As a side note, I believe that using a Transformer model for this task may be overkill. A simpler classifier could
likely
achieve similar results with the right feature engineering. For instance, if the user query contains "Send email," the
`send_email` tool would likely be necessary.

The trained model is available at [zavidnyi/toolrag-reproduction](https://huggingface.co/zavidnyi/toolrag-reproduction).
The training script can be found in `src/training/classifier/train.py`.

You can evaluate the model using the following command:

```bash
python src/evaluation/classifier/evaluate.py \
    --model_path zavidnyi/toolrag-reproduction or squeeze-ai-lab/TinyAgent-ToolRAG \
    --data_path <path_to_your_data_json_file>
```

## Planner

The TinyAgent paper states that TinyLlama 1.1B was fine-tuned for three epochs with a learning rate of 7e-5 using LoRA.
Based on the config.json file, I inferred that the batch size was likely 8.

Initially, my reproduced model’s performance significantly lagged behind the paper’s reported metrics (~59% correct
plans vs. ~78%). One recurring issue was that the model frequently generated empty outputs, which I couldn’t explain
initially.

Upon investigation, I realized the fine-tuning process likely used the same prompt format as the base
model [Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct](https://huggingface.co/Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct):

```md
### Instruction:

{system prompt}

### Input:

{user message}

### Response:

{model response}
```

Weirdly enough, the code which uses this model doesn't seem to use this format.
Instead, messages are just concatenated as is.
Anyway, after fixing the message format, I've managed to get reference model metrics to ~76%.

Finetuning my own model didn't go without hiccups either.
Took quite some trial and error to figure out how to setup the training pipeline correctly.
For example, issues with `max_seq_len` and issues with prompting from the point above.
In the end, I've managed to configure everything correctly and finetuned a model for ~15 hours on A100 GPU.
Evaluating the model I've trained I've got ~75% correct plans, which is quite close to the reference model.
Testing using a McNemar's test, I've got a p-value of ~`0.47` suggesting insignificant difference between the models.
That makes me believe that after some more hyperparameter tuning, I'd be able to get the exact same metrics as in the
paper.

You can download the trained model
from [zavidnyi/tinyagent-reproduction](https://huggingface.co/zavidnyi/tinyagent-reproduction).
Training script is in `src/training/planner/train.py`.

### Concerns on the data

One thing which bothers me is some ambiguity around what exactly the model was trained and evaluated on by the authors.
Specifically, the dataset has 40k examples (also mentioned on dataset hf page), while says it's 80k samples.
The reason for it, is that each of the 40k samples consists of 2 different message sequences:

1. Where model given a user query constructs a sequential plan of tool execution to fulfill the query. That's what I
   used for finetuning, and that's what authors give evaluation metric for.
2. Given results of execution, say whether the query was fulfilled or if a model should replan. This is not directly
   mentioned
   in the paper. I assume this wasn't used for evaluation as well.

Potentially adding this data could improve performance of the model I trained.
However, I thought about this issue pretty late in the process, and rerunning the experiments from scratch would be a
bit too out of scope of this reproduction.
