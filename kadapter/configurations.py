from transformers import BertConfig, PretrainedConfig, AutoConfig
import copy


class AdapterConfig(BertConfig):
    
    def __init__(self,
                 model_name='adapter',
                 hidden_dimension=120,
                 injection_layers = (0,11),
                 skip_layers = 3,
                 initializer_range = 0.0002,
                 num_hidden_layers = 2,
                 **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.hidden_dimension = hidden_dimension
        self.injection_layers = injection_layers
        self.skip_layers = skip_layers
        self.initializer_range = initializer_range
        self.num_hidden_layers = 2


class KAdapterHeadConfig(PretrainedConfig):
    
    def __init__(self, 
                 head_type: str = 'concat',
                 **kwargs):
        self.head_type = head_type
        super().__init__(**kwargs)


class KAdapterConfig(PretrainedConfig):
    model_type = 'kadapter'
    is_composition=True
    
    def __init__(self, **kwargs):
        basemodel_config = kwargs.pop('basemodel_config')
        self.basemodel = AutoConfig.for_model(basemodel_config.pop("model_type"), **basemodel_config)
        self.adapters = [AdapterConfig(**params) for params in kwargs.pop('adapter_configs')]
        self.head = KAdapterHeadConfig(**kwargs.pop('head_config'))
        super().__init__(**kwargs)

    @classmethod
    def from_adapter_configs(cls, basemodel_config: PretrainedConfig, adapter_configs : list, head_config : KAdapterHeadConfig):
        adapter_config_dicts = [config.to_dict() for config in adapter_configs]
        return cls(basemodel_config=basemodel_config.to_dict(), adapter_configs=adapter_config_dicts, head_config=head_config.to_dict())
    
    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output['basemodel'] = self.basemodel.to_dict()
        output['adapters'] = [adapter.to_dict() for adapter in self.adapters]
        output['head'] = self.head.to_dict()
        output['model_type'] = self.__class__.model_type
        return output
    
