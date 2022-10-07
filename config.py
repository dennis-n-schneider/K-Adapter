class AdapterConfig:
    # Configuration per Adapter
    adapters = [
        dict(
            # This is equal to the papers usage of layers [0,11];
            # In general, layer 0 is the embedding layer 'embeddings.dropout'
            #             layer i is 'encoder.layer.(i-1)'
            # Names may very from model to model,
            # but can be retrieved by running
            # print([name for name, mod in basemodel.named_modules()])
            injection_layers = ['embeddings.dropout', 'encoder.layer.10'],
            # Add a skip-connection spanning n adapter-layers.
            skip_layers = 3,
            # Size of the downscaled adapter hidden-dimension
            hidden_dimension = 768,
            # Range of initialization of adapter-weights (uniformly in [-n,n])
            initializer_range = 0.0002,
        )
    ]
    
    # Combination method of the Adapter-Outputs
    head = dict(
        # One of ConcatHead, SumHead
        combine = 'ConcatHead'
    )
    # number of labels in the output
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