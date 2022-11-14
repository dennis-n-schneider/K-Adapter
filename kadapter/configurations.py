import copy

from transformers import BertConfig, PretrainedConfig, AutoConfig


class AdapterConfig(BertConfig):
    model_type = 'kadapter-adapter'

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
        self.num_hidden_layers = num_hidden_layers
        self.freeze = freeze


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

    def __init__(self,
                 freeze_basemodel=False,
                 **kwargs):
        self.freeze_basemodel = freeze_basemodel
        basemodel_config = kwargs.pop('basemodel')
        if basemodel_config:
            self.basemodel = AutoConfig.for_model(basemodel_config.pop("model_type"), **basemodel_config)
        else:
            self.basemodel = None
        adapter_configs = kwargs.pop('adapters')
        if adapter_configs:
            self.adapters = [AdapterConfig(**params) for params in adapter_configs]
        else:
            self.adapters = None
        head_config = kwargs.pop('head')
        if head_config:
            if 'n_adapters' not in head_config:
                head_config['n_adapters'] = len(self.adapters)
            assert 'hidden_size' in head_config or self.basemodel, \
                'Either supply a basemodel or explicitly state the hidden_size in the HeadConfig.'
            if 'hidden_size' not in head_config:
                head_config['hidden_size'] = basemodel_config['hidden_size']
            self.head = AutoConfig.for_model(head_config.pop('model_type'), **head_config)
        else:
            self.head = None
        super().__init__(**kwargs)

    @classmethod
    def from_adapter_configs(cls,
                             basemodel: PretrainedConfig = None,
                             adapters: list = None,
                             head: PretrainedConfig = None,
                             **kwargs):
        adapter_config_dicts = [config.to_dict() for config in adapters] if adapters else None
        basemodel_dict = basemodel.to_dict() if basemodel else None
        head_dict = head.to_dict() if head else None
        return cls(basemodel=basemodel_dict, adapters=adapter_config_dicts, head=head_dict, **kwargs)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output['basemodel'] = self.basemodel.to_dict()
        output['adapters'] = [adapter.to_dict() for adapter in self.adapters]
        output['head'] = self.head.to_dict()
        output['model_type'] = self.__class__.model_type
        return output
