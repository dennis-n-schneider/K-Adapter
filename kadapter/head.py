from .configurations import KAdapterHeadConfig

from transformers import PreTrainedModel, AutoConfig, AutoModel
import torch


class KAdapterHead:
    
    def __init__(self, 
                 config: KAdapterHeadConfig):
        super().__init__(config)
        self.dropout = torch.nn.Dropout()
        
    def forward(self, adapter_outputs: torch.Tensor):
        output = self.combine(torch.stack(adapter_outputs))
        assert output.shape == adapter_outputs.shape
        output = self.dropout(output)
        # TODO this transforms from the entire output to num_labels!
        # output = self.linear(output[:, 0, :].squeeze(1))
        return output
    
    def combine(self, adapter_outputs):
        pass
    
    def get_output_embeddings(self):
        raise NotImplementedError()


class SumHead(KAdapterHead, PreTrainedModel):
    """
    Sums up the outputs of Adapters.
    """
    config_class = KAdapterHeadConfig
    
    def combine(self, adapter_outputs: torch.Tensor):
        return adapter_outputs.sum(0)


class ConcatHead(KAdapterHead, PreTrainedModel):
    """
    Concatenates the outputs of Adapters
    """
    config_class = KAdapterHeadConfig
    
    
    def __init__(self, 
                 config: KAdapterHeadConfig):
        super().__init__(config)
        print(config)
        self.n_adapters = config.n_adapters
        self.hidden_size = config.hidden_size
        self.linear_out = torch.nn.Linear(2*self.n_adapters*self.hidden_size, self.hidden_size)
        
    def combine(self, adapter_outputs: torch.Tensor):
        hidden = torch.concat([adapter_outputs[1:], adapter_outputs[0].repeat(self.n_adapters,1,1,1)], -1)
        assert adapter_outputs.dim() == 4
        output = torch.cat(torch.unbind(hidden), -1)
        return self.linear_out(output)

    
from .configurations import KAdapterHeadConfig

AutoConfig.register("kadapter-head-sum", KAdapterHeadConfig)
AutoModel.register(KAdapterHeadConfig, SumHead)

#AutoConfig.register("kadapter-head-concat", KAdapterHeadConfig)
#AutoModel.register(KAdapterHeadConfig, ConcatHead)