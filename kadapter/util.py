import torch
from torch import nn


class FeatureExtractor(nn.Module):
    """
    Extracts hidden features of a certain model at given layers.
    """
    
    def __init__(self, model, layer_names):
        super().__init__()
        self.model = model
        
        self._features = [torch.empty(0) for _ in layer_names]

        for i, name in enumerate(layer_names):
            get_module(self.model, name).register_forward_hook(
                self.save_features_hook(i)
            )

    def save_features_hook(self, layer_id: str):
        def fn(_, __, output):
            if isinstance(output, tuple):
                assert len(output) == 1
                output = output[0]
            self._features[layer_id] = output
        return fn

    def forward(self, x: torch.Tensor):
        return self.model(x)
    
    def features(self):
        return self._features


def get_module(model, module):
    if isinstance(module, nn.Module):
        return module

    assert isinstance(module, str)
    if module == '':
        return model

    for name, curr_module in model.named_modules():
        if name == module:
            return curr_module

    return None
