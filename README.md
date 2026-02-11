### PROJECT DESCRIPTION

This project involves implementing a ```GPT-2–style autoregressive language model``` from scratch in Python. The model is a decoder-only ```Transformer``` trained using next-token prediction, featuring ***token and positional embeddings***, ***multi-head causal self-attention***, ***feed-forward networks***, ***shortcut residual connections***, and ***layer normalization***. 

The goal is to understand the core architecture and training dynamics of large language models by building and training a small-scale GPT-2–like model without relying on high-level libraries.

### ENVIRONMENT SETUP

```bash
mkdir llm-from-scratch

cd llm-from-scratch

uv init

uv sync

source .venv/bin/activate

uv add -r requirements.txt
```

[Notebook-1](notebook1-BuildLLM) - this section describes the essence of ```tokenizing texts```, ```attention mechanism``` and the broad overview of the ```llm architecture```.

[Notebook-2](notebook2-TrainLLM) - this section describes calculates the calculating ```cross-entropy-loss``` between ```logits and target outputs```, decoding strategies such as ```temperature and top-k``` to get the best next token and loading the released OpenAI GPT-2 weights into the built LLM Architecture.

[Notebook-3](notebook3-FinetuneLLM) - this section describes finetuning the pretrained GPT model on a ```classification dataset```, and also enabling the model to follow instructions in ```Alpaca format```.

[Notebook-4](notebook4-Quantization) - run this notebook to get the quantized model of the pretrained model.