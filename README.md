# K-Adapter Reimplementation

```
Disclaimer: This implementation is not related to the original paper and has simply been
rewritten since I needed to use it and the original codebase was ... messy.
```

This is a clean reimplementation (WIP) of the K-Adapter paper (["K-Adapter: Infusing Knowledge into Pre-Trained Models with Adapters"](https://arxiv.org/abs/2002.01808), R. Wang et al.).
Click to get to the original [repo](https://github.com/microsoft/K-Adapter).
This Repository is Huggingface-compatible and trained models can thus be saved, loaded and used interchangeably with the original basemodels (BERT, RoBERTa, ...).

## Requirements

Python 3.8

For used packages, see requirements.txt \
And yes, the code would have looked a little better if I had used Python 3.9 or 3.10, but this way it is compatible to more setups.

## Usage

See Usage.ipynb and standard Huggingface-resources for general usage information and examples.
The ```KAdapterModel``` is a wrapper around an arbitrary amount of ```Adapter```s and a ```Head``` which combines the adapter outputs.
The adapters can be inserted one-by-one and trained independently of each other. \
In the end, the list of trained adapters can be inserted into the kadapter model with a head, be trained and saved to a directory. \
From this point on, the KAdapterModel can be loaded and used exactly as the basemodel would be used.

## Configuration

The models are configured according to configuration classes defined in ```kadapter/configurations.py```.
The basemodel's configuration is saved alongside the KAdapter configurations as the adapters depend on the basemodels configuration and weights.
For more information on the meaning of the configuration options, see the original paper.
