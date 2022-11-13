from transformers import PreTrainedModel


class ConcatHead():
    
    def __init__(self):
        print("Hi, I'm a KAdapterHead Concat!")


class KAdapterHead(PreTrainedModel):
    
    def __call__(self, head_type:str):
        if head_type == 'concat':
            return ConcatHead()

        