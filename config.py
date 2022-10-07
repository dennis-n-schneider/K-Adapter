class AdapterConfig:
    # Configuration per Adapter
    adapters = [
        dict(
            injection_layers = ['embeddings.dropout', 'encoder.layer.10'],
            skip_layers = 3,
            hidden_dimension = 768, # size of the downscaled adapter
            initializer_range = 0.0002,
        )
    ]
    head = dict(
        """
        Combination method of the Adapter-Outputs
        """
        combine = 'ConcatHead'
    )
    num_labels: int = 3
    initializer_range: float = 0.02 # ? Is this used in the original paper?
    
    # Configuration of the adapter-internal BertEncoder-Layer
    vocab_size = 50265
    hidden_size = 768
    num_hidden_layers = 2
    num_attention_heads = 12
    intermediate_size = 3072
    hidden_act = 'gelu'
    hidden_dropout_prob = 0.1
    attention_probs_dropout_prob = 0.1
    max_position_embeddings = 514
    type_vocab_size = 1
    initializer_range = 0.02
    layer_norm_eps = 1e-05
    pad_token_id = 0
    position_embedding_type = 'absolute'
    use_cache = True
    classifier_dropout = None
    is_decoder: bool = False
    torchscript: bool = False
    chunk_size_feed_forward: int = 0
    add_cross_attention: bool = False