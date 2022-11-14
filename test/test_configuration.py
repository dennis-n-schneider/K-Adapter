import unittest
from kadapter.configurations import AdapterConfig


class TestAdapterConfig(unittest.TestCase):
    
    def test_save_load(self):
        config = AdapterConfig(injection_layers = [3,7])
        config.save_pretrained('test/save/adapter_config_test')
        
        loaded_config = AdapterConfig.from_pretrained('test/save/adapter_config_test')
        self.assertDictEqual(config.to_dict(), loaded_config.to_dict())


class TestKAdapterHeadConfig(unittest.TestCase):
    """
    - auto-load
    - save
    """
    pass


class TestKAdapterConfig(unittest.TestCase):
    
    # TODO
    def test_save_load_all(self):
        config = AdapterConfig(injection_layers = [3,7])
        config.save_pretrained('test/save/adapter_config_test')
        
        loaded_config = AdapterConfig.from_pretrained('test/save/adapter_config_test')
        self.assertDictEqual(config.to_dict(), loaded_config.to_dict())

    def test_save_load_individual(self):
        config = AdapterConfig(injection_layers = [3,7])
        config.save_pretrained('test/save/adapter_config_test')
        
        loaded_config = AdapterConfig.from_pretrained('test/save/adapter_config_test')
        self.assertDictEqual(config.to_dict(), loaded_config.to_dict())

