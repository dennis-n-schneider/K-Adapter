import torch
from torch import nn
import sys


class KAdapterHead(nn.Module):
    
    def __init__(self, num_labels, hidden_size, p_dropout=0.1, **kwargs):
        super().__init__()
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.dropout = torch.nn.Dropout()
        self.linear = torch.nn.Linear(hidden_size, num_labels)
        
    def forward(self, adapter_outputs):
        output = self.combine(torch.stack(adapter_outputs))
        output = self.dropout(output)
        output = self.linear(output[:, 0, :].squeeze(1))
        return output
        
    def combine(self, adapter_outputs):
        pass


class SumHead(KAdapterHead):
    
    def combine(self, adapter_outputs: torch.Tensor):
        return adapter_outputs.sum(0)


class ConcatHead(KAdapterHead):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_adapters = len(kwargs['adapters'])
        self.linear_out = torch.nn.Linear(2*self.n_adapters*self.hidden_size, self.hidden_size)
    
    def combine(self, adapter_outputs: torch.Tensor):
        hidden = torch.concat([adapter_outputs[1:], adapter_outputs[0].repeat(self.n_adapters,1,1,1)], -1)
        assert adapter_outputs.dim() == 4
        output = torch.cat(torch.unbind(hidden), -1)
        return self.linear_out(output)


class HeadFactory:
    
    def __call__(self, combine: str, head_config: dict) -> KAdapterHead:
        cls = getattr(sys.modules[__name__], combine)
        return cls(**head_config)
