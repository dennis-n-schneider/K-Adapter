from .base import Adapter
from .configurations import KAdapterConfig
from .head import KAdapterHead

from transformers import PreTrainedModel, AutoModel
from typing import Optional
from torch import nn


class KAdapterModel(PreTrainedModel):
    config_class = KAdapterConfig
    base_model_prefix = 'kadapter'
    
    # This is correct. Models and configs
    def __init__(self, 
                 basemodel: PreTrainedModel = None,
                 config: Optional[KAdapterConfig]=None,
                 adapters: list = None,
                 head: Optional[KAdapterHead] = None):
        if config is None and (basemodel is None or adapters is None or head is None):
            raise ValueError("Either a configuration or models have to be provided.")
        if config is None:
            config = KAdapterConfig.from_adapter_configs(basemodel.config, [adapter.config for adapter in adapters],
                                                        head.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        # initialize with config
        super().__init__(config)
        
        if basemodel is None:
            basemodel = AutoModel(config.basemodel)
        if adapters is None:
            adapters = nn.ModuleList([Adapter(config) for config in config.adapters])
        if head is None:
            head = KAdapterHead(config.head)
        
        assert basemodel.config.return_dict
        assert basemodel.config.output_hidden_states
        self.basemodel = basemodel
        self.adapters = adapters
        self.head = head
        
        if self.basemodel.config.to_dict() != self.config.basemodel.to_dict():
            logger.warning(
                f"Config of the Basemodel is overwritten by shared Basemodel config"
            )
        if [adapter.config.to_dict() for adapter in adapters] != \
            [config.to_dict() for config in self.config.adapters]:
            logger.warning(
                f"Configs of the Adapters are overwritten by shared Adapter config"
            )
        if self.head.config.to_dict() != self.config.head.to_dict():
            logger.warning(
                f"Config of the K-Adapter Head is overwritten by shared Head config"
            )
        
        self.basemodel.config = self.config.basemodel
        self.head.config = self.config.head
        # probably does not work, should match by name.
        for adapter, shared_adapter_config in zip(self.adapters, config.adapters):
            adapter.config = shared_adapter_config

    def forward(self, inputs):
        basemodel_out = self.basemodel(inputs)
        entire_adapter_outputs = [adapter(basemodel_out.last_hidden_state, basemodel_out.hidden_states) for adapter in self.adapters]
        adapter_outputs, basemodel_outputs = zip(*entire_adapter_outputs)
        basemodel_output = basemodel_outputs[0]
        return basemodel_output #self.head([basemodel_output] + adapter_outputs)
    
    @staticmethod
    def freeze_model(model):
        for param in model.parameters():
            param.requires_grad = False
    
    @classmethod
    def from_adapters_pretrained(
        cls,
        basemodel_pretrained_model_name_or_path : str = None,
        adapters_pretrained_model_name_or_path : list = None,
        head_pretrained_model_name_or_path : str = None,
        *model_args,
        **kwargs
    ):
        # TODO: Case where extra config is supplied to a certain adapter!
        config = None
        # TODO: should I supply the basemodel here?
        adapter_models = [Adapter.from_pretrained(name_or_path)
                          for name_or_path in adapters_pretrained_model_name_or_path]
        basemodel = AutoModel.from_pretrained(basemodel_pretrained_name_or_path)
        # TODO: Load pretrained Head
        # kadapter_head = KAdapterHead.from_pretrained(head_pretrained_model_name_or_path)
        return cls(basemodel=basemodel, adapters=adapter_models, head=None, config=config)
    
    def get_output_embeddings(self):
        return self.head.get_output_embeddings()
