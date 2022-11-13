from . import util

from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
import torch
from torch import nn, Tensor

from transformers import PreTrainedModel


class AdapterLayer(PreTrainedModel):
    """
    A single adapter layer injected at a certain hidden layer of the base-model.
    """
    
    def __init__(self, config):
        super().__init__(config)
        basemodel_hidden_dim = config.hidden_size
        self.down_project = nn.Linear(basemodel_hidden_dim, config.hidden_dimension)
        
        config_dict = config.to_dict()
        config_dict['hidden_size'] = config.hidden_dimension
        self.encoder = BertEncoder(BertConfig(**config_dict))
        
        self.up_project = nn.Linear(config.hidden_dimension, basemodel_hidden_dim)

    def forward(self, input_features: Tensor) -> Tensor:
        down_projected = self.down_project(input_features)
        encoder_outputs = self.encoder(down_projected)
        up_projected = self.up_project(encoder_outputs[0])
        # skip connection
        return input_features + up_projected

    def init_weights(self):
        # TODO never used!
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-self.config.initializer_range, self.config.initializer_range)
                if not m.bias is None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data.uniform_(-self.config.initializer_range, self.config.initializer_range)


class Adapter(PreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.adapter_layers = nn.ModuleList([AdapterLayer(config) for _ in config.injection_layers])
        if config.freeze:
            util.freeze_model(self)
        
    def forward(self, base_output, base_hidden_states):
        base_hidden_injections = util.extract_features(base_hidden_states, self.config.injection_layers)
        adapter_outputs = []
        for adapter_module, base_hidden_features in zip(self.adapter_layers,
                                                       base_hidden_injections):
            prev_adapter_output = adapter_outputs[-1] if adapter_outputs else torch.zeros(base_output.shape)
            fusion_input = base_hidden_features + prev_adapter_output
            adapter_outputs.append(adapter_module(fusion_input))

            if (self.config.skip_layers > 0) and (len(adapter_outputs) % self.config.skip_layers == 0):
                adapter_outputs[-1] += adapter_outputs[int(len(adapter_outputs) // self.adapter_skip_layers)]

        return adapter_outputs[-1], base_output
