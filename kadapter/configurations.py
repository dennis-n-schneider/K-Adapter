import copy

from transformers import BertConfig, PretrainedConfig, AutoConfig


class AdapterConfig(BertConfig):

    def __init__(self,
                 model_name='adapter',
                 hidden_dimension=120,
                 injection_layers=(0, 11),
                 skip_layers=3,
                 initializer_range=0.0002,
                 num_hidden_layers=2,
                 freeze=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.hidden_dimension = hidden_dimension
        self.injection_layers = injection_layers
        self.skip_layers = skip_layers
        self.initializer_range = initializer_range
        self.num_hidden_layers = 2
        self.freeze = freeze


"""
class KAdapterHeadConfig(PretrainedConfig):
    model_type = 'kadapter-head-sum'

    def __init__(self,
                 head_type: str = 'kadapter-head-sum',
                 **kwargs):
        self.head_type = head_type
        super().__init__(**kwargs)
"""


class KAdapterHeadConfig:

    def __init__(self,
                 p_dropout=0.5,
                 **kwargs,
                 ):
        super().__init__(**kwargs)
        self.p_dropout = p_dropout


class KAdapterSumHeadConfig(KAdapterHeadConfig, PretrainedConfig):
    model_type = 'kadapter-head-sum'

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)


class KAdapterConcatHeadConfig(KAdapterHeadConfig, PretrainedConfig):
    model_type = 'kadapter-head-concat'

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)


class KAdapterConfig(PretrainedConfig):
    model_type = 'kadapter'
    is_composition = True

    def __init__(self, **kwargs):
        self.freeze_basemodel = False
        basemodel_config = kwargs.pop('basemodel')
        self.basemodel = AutoConfig.for_model(basemodel_config.pop("model_type"), **basemodel_config)
        self.adapters = [AdapterConfig(**params) for params in kwargs.pop('adapters')]
        head_config = kwargs.pop('head')
        head_config['n_adapters'] = len(self.adapters)
        head_config['hidden_size'] = self.basemodel.hidden_size
        self.head = AutoConfig.for_model(head_config.pop('model_type'), **head_config)
        super().__init__(**kwargs)

    @classmethod
    def from_adapter_configs(cls, basemodel: PretrainedConfig, adapters: list, head: PretrainedConfig):
        adapter_config_dicts = [config.to_dict() for config in adapters]
        return cls(basemodel=basemodel.to_dict(), adapters=adapter_config_dicts, head=head.to_dict())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output['basemodel'] = self.basemodel.to_dict()
        output['adapters'] = [adapter.to_dict() for adapter in self.adapters]
        output['head'] = self.head.to_dict()
        output['model_type'] = self.__class__.model_type
        return output
