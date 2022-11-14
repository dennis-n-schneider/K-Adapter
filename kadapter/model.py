import logging
from typing import Optional

from torch import nn
from transformers import PreTrainedModel, AutoModel

from . import util
from .base import Adapter
from .configurations import KAdapterConfig
from .head import KAdapterHead


class KAdapterModel(PreTrainedModel):
    config_class = KAdapterConfig
    base_model_prefix = 'basemodel'

    def __init__(self,
                 config: Optional[KAdapterConfig] = None,
                 basemodel: PreTrainedModel = None,
                 adapters: list = None,
                 head: Optional[KAdapterHead] = None):
        if config is None:
            config = KAdapterConfig.from_adapter_configs(basemodel.config if basemodel else None,
                                                         [adapter.config for adapter in adapters] if adapters else None,
                                                         head.config if head else None)
        elif not isinstance(config, self.config_class):
            raise ValueError(f"Config: {config} has to be of type {self.config_class}")
        super().__init__(config)
        # Either load from config or from a pretrained model
        assert (self.config.basemodel or basemodel) and \
               (self.config.adapters or adapters) and \
               (self.config.head or head)
        assert basemodel.config.return_dict
        assert basemodel.config.output_hidden_states
        self.basemodel = KAdapterModel.__get_object_from_config(basemodel, self.config, 'basemodel')
        self.adapters = KAdapterModel.__get_object_from_config(adapters, self.config, 'adapters')
        self.head = KAdapterModel.__get_object_from_config(head, self.config, 'head')
        if self.config.freeze_basemodel:
            util.freeze_model(self.basemodel)

    def forward(self, inputs):
        basemodel_out = self.basemodel(inputs)
        entire_adapter_outputs = [adapter(basemodel_out.last_hidden_state, basemodel_out.hidden_states) for adapter in
                                  self.adapters]
        adapter_outputs, basemodel_outputs = zip(*entire_adapter_outputs)
        basemodel_output = basemodel_outputs[0]

        return self.head([basemodel_output] + adapter_outputs)

    @classmethod
    def from_adapters_pretrained(
            cls,
            basemodel_pretrained_model_name_or_path: str = None,
            adapters_pretrained_model_name_or_path: list = None,
            head_pretrained_model_name_or_path: str = None,
            *model_args,
            **kwargs
    ):
        # TODO: Case where extra config is supplied to a certain adapter!
        if 'config' in kwargs:
            config = kwargs.pop('config')
        else:
            config = None
        basemodel = AutoModel.from_pretrained(basemodel_pretrained_model_name_or_path, **kwargs) \
            if basemodel_pretrained_model_name_or_path else None
        adapter_models = [Adapter.from_pretrained(name_or_path)
                          for name_or_path in adapters_pretrained_model_name_or_path] \
            if adapters_pretrained_model_name_or_path else None
        kadapter_head = AutoModel.from_pretrained(head_pretrained_model_name_or_path) \
            if head_pretrained_model_name_or_path else None
        return cls(basemodel=basemodel, adapters=adapter_models, head=kadapter_head, config=config)

    def get_output_embeddings(self):
        return self.head.get_output_embeddings()

    @staticmethod
    def __get_object_from_config(pretrained, explicit_config, attribute_name: str):
        if attribute_name == 'adapters':
            return KAdapterModel.__get_adapters_from_config(pretrained, explicit_config)
        configured_object = getattr(explicit_config, attribute_name)
        if not pretrained:
            pretrained = AutoModel.from_config(configured_object)
        if not configured_object:
            setattr(explicit_config, attribute_name, pretrained.config)
        elif pretrained.config.to_dict() != configured_object.to_dict():
            logging.warning(
                f"Config of pretrained {attribute_name} is overwritten by explicit {attribute_name} config"
            )
            pretrained.config = configured_object
        return pretrained

    @staticmethod
    def __get_adapters_from_config(pretrained, explicit_config):
        if not pretrained:
            pretrained = nn.ModuleList([Adapter(config) for config in explicit_config.adapters])
        if not explicit_config.adapters:
            explicit_config.adapters = [adapter.config for adapter in pretrained]
        elif [adapter.config.to_dict() for adapter in pretrained] != \
                [config.to_dict() for config in explicit_config.adapters]:
            logging.warning(
                f"Configs of pretrained Adapters are overwritten by explicit Adapter configs"
            )
            for pretrained_adapter, configured_adapter in zip(pretrained, explicit_config.adapters):
                pretrained_adapter.config = configured_adapter
        return pretrained
