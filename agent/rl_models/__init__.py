import torch

from .value import *
from .heightmap_value import HeightmapFeasibilityModel, HeightmapValueModel


def load_model(model_cls, model_path, device):
    object = torch.load(model_path, device, weights_only=False)
    object["cfg"].device = device
    model = model_cls(object["cfg"])
    model.set_device(device)
    model.load_state_dict(object["model_state"])
    print(f"Load model from {model_path}")
    return model, object["cfg"]
