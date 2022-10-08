from .base import AdapterFactory
from .head import KAdapterHead, HeadFactory

from torch import nn


class KAdapter(nn.Module):
    """
    The entire K-Adapter Architecture.
    Encapsules a number of Adapters.
    Combines Adapter-Ouputs in a KAdapterHead.
    """
    
    def __init__(self, basemodel: nn.Module, config, 
                 adapters: list = None, head: KAdapterHead = None):
        super().__init__()
        self.basemodel = basemodel
        KAdapter.freeze_model(basemodel)
        self.config = config
        if adapters is None:
            adapters = [AdapterFactory(self.basemodel)(**adapter_config) for adapter_config in config.adapters]
        self.adapters = nn.ModuleList(adapters)
        self.head = HeadFactory()(config.head['combine'], **vars(config)) if head is None else head

    def forward(self, input_values):
        entire_adapter_outputs = [adapter(input_values) for adapter in self.adapters]
        basemodel_output = entire_adapter_outputs[0][1]
        adapter_outputs = zip(*entire_adapter_outputs)[0]
        return self.head([basemodel_output] + adapter_outputs)
    
    @staticmethod
    def freeze_model(model):
        for param in model.parameters():
            param.requires_grad = False
