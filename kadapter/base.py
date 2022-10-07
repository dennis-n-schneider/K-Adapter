from . import util

from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
import torch
from torch import nn, Tensor


class AdapterLayer(nn.Module):
    def __init__(self, basemodel_hidden_dim: int, hidden_dimension: int, initializer_range: float, **kwargs):
        super().__init__()
        bert_config = BertConfig(**{**kwargs, **{'basemodel_hidden_dim': basemodel_hidden_dim}})
        self.initializer_range = initializer_range
        self.down_project = nn.Linear(basemodel_hidden_dim, hidden_dimension)
        self.encoder = BertEncoder(bert_config)
        self.up_project = nn.Linear(hidden_dimension, basemodel_hidden_dim)

    def get_masks(self, input_shape, input_device):
        attention_mask = torch.zeros(input_shape, device=input_device, dtype=next(self.parameters()).dtype)
        extended_attention_mask = attention_mask.unsqueeze(1)
        if attention_mask.dim() == 2:
            extended_attention_mask = extended_attention_mask.unsqueeze(2)
        # head_mask = [None] * self.n_hidden_layers # needed if only None?
        return extended_attention_mask# , head_mask # needed if all zero?

    def forward(self, input_features: Tensor) -> Tensor:
        down_projected = self.down_project(input_features)
        attention_mask = self.get_masks(down_projected.size()[:-1], input_features.device)
        encoder_outputs = self.encoder(down_projected,
                                       attention_mask=attention_mask)
        up_projected = self.up_project(encoder_outputs[0])
        # skip connection
        return input_features + up_projected

    def init_weights(self):
        # original
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-self.initializer_range, self.initializer_range)
                if not m.bias is None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data.uniform_(-self.initializer_range, self.initializer_range)


class Adapter(nn.Module):

    def __init__(self, model, injection_layers, skip_layers=3, **kwargs):
        super().__init__()
        self.base_model = util.FeatureExtractor(model, injection_layers)
        self.skip_layers = skip_layers
        self.adapter_layers = [AdapterLayer(model.config.hidden_size, **kwargs) for _ in injection_layers]

    def forward(self, inputs):
        base_output = self.base_model(inputs)
        base_hidden_injections = self.base_model.features()
        adapter_outputs = []
        for adapter_module, base_hidden_features in zip(self.adapter_layers, base_hidden_injections):
            prev_adapter_output = adapter_outputs[-1] if adapter_outputs else torch.zeros(base_output.last_hidden_state.shape)
            fusion_input = base_hidden_features + prev_adapter_output
            adapter_outputs.append(adapter_module(fusion_input))

            if (self.skip_layers > 0) and (len(adapter_outputs) % self.skip_layers == 0):
                adapter_outputs[-1] += adapter_outputs[int(len(adapter_outputs) // self.adapter_skip_layers)]

        return adapter_outputs[-1], base_output.last_hidden_state


class AdapterFactory:
    
    def __init__(self, model):
        self.model = model
    
    def __call__(self, **kwargs):
        return Adapter(self.model, **kwargs)
