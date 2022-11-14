import torch


def extract_features(hidden_states: torch.Tensor, layer_ids: list):
    outputs = tuple(feature_tensor for i, feature_tensor in enumerate(hidden_states) if i in layer_ids)
    return outputs


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
