# Reproducing TinyAgent

TinyAgent paper describes an implementation of an agent capable of executing user instructions using one of the 16
provided tools.
Tools range from sending an email to getting direction to a specific location.
My purpose was to reproduce the models used in the paper, so exact implementation details are not important in this
report.
However, here is the general description of the pipeline, that is important for understanding of the following text:

1. Given a user query, we first predict all of the tools, that could be needed for the execution.
   For this, paper uses [DeBERTaV3 small](https://huggingface.co/microsoft/deberta-v3-small) model, fine-tuned for
   16-way classification. They call this approach ToolRAG.
2. Once that's done, the prompt consisting of instructions, descriptions of the predicted tools from previous step, some
   similar examples and the user query passed to a Planner model. For this, paper uses
   fine-tuned [Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct](https://huggingface.co/Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct).

Now, I'll describe how I reproduced each of the models and technical issues I've encountered.

## ToolRAG

The paper didn't describe how exactly the ToolRAG model was fine-tuned, so I had to figure it out myself.
One of the important details, I've found is that inputs to this model are limited to `512`
tokens ([see](https://github.com/SqueezeAILab/TinyAgent/blob/e63305d9bbe767493bc34db777bbcd7aba006bf0/src/tiny_agent/tool_rag/classifier_tool_rag.py#L97)).
First approach using HF Trainer failed.
I got ~93% tool recall on the validation set, comparing to ~99% in the paper.
More experimentation with different parameters didn't yield any better results :(

However, looking more closely at the data I've noticed that my models on average predicted much less tools than the
model from the paper.
To mitigate this problem, I've tried parametrizing a `BCEWithLogitsLoss` loss function HF uses under the hood for
multi-label classification with a >1 `pos_weight` parameter, which helps to boost recall.
After, some experimentation, I've settled for the weight `60`, with which I got `~99%` tool recall with `~3.99` tool
count vs. `~3.97` in the paper.
Since numbers are quite similar I assume this is the approach that the authors used, so after fiddling a bit more with
hyperparameters, it would'd be possible to arrive to the exact same metrics.

I also would like to mention that using Transformer model for this task is probably an overkill.
I think that a simpler classifier would work just fine here, provided right features are used.
For example, if we have "Send email" text in the user query, we'd most likely need a tool `send_email`.

You can download the trained model
from [zavidnyi/toolrag-reproduction](https://huggingface.co/zavidnyi/toolrag-reproduction).
Training script is in `src/training/classifier/train.py`.

## Planner

The paper tells us that a TinyLlama 1.1B was fine tuned for 3 epochs with learning rate `7e-5` using LoRA.
Additionally, from the `config.json` of the model, I've infered that batch size was probably `8`, since it was marked
in the `model_path` parameter.

First difficulty I've encountered was that I could't get similar metrics as the ones reported in the paper.
For comparison, paper reports that model generates ~78% of correct plans, while I got only ~59%.
Among other issues, it was a problem that for some reason model generated a lot of empty outputs, which I couldn't
really understand.
Investigating more I've realised that a model was most likely fine-tuned using the same prompt format as the model it
was fine-tuned
from [Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct](https://huggingface.co/Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct):
```md
### Instruction:
{system prompt}

### Input:
{user message}

### Response:
{model response}
```

Weirdly enough, the code which uses this model doesn't seem to use this format.
Instead, messages are just concatanated as is.
Anyway, after fixing the message format I've managed to get reference model metrics to ~76%.

