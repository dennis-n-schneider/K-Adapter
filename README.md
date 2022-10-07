# K-Adapter Reimplementation

```
Disclaimer: This implementation is not related to the original paper and has simply been
rewritten since I needed to use it and the original codebase was ... messy.
```

This is a clean reimplementation of the K-Adapter paper (["K-Adapter: Infusing Knowledge into Pre-Trained Models with Adapters"](https://arxiv.org/abs/2002.01808), R. Wang et al.).
Click to get to the original [repo](https://github.com/microsoft/K-Adapter).

## Requirements

Python 3.8

For used packages, see requirements.txt \
And yes, the code would have looked a little better if I had used Python 3.9 or 3.10, but this way it is compatible to more setups.

## Usage

See Usage.ipynb for general usage information.

## Configuration

The model is configured in config.py \
Each used adapter can be modified as depicted in the provided example.
In the end, the head (either ConcatHead or SumHead) combines the outputs of the $n$ provided adapters. \
The rest of the config-file described configuration of the BertEncoder-Layer used within the adapter-layers. \
For more information on the meaning of these configuration options, see the paper.
