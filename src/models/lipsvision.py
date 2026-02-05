from .lipsalexnet import LipsAlexNetModule
from .lipsresnet import LipsResNetModule


def get_module(config: dict):
    model_name = config["model_name"]
    if model_name == "lipsalexnet":
        return LipsAlexNetModule
    if model_name == "lipsresnet":
        return LipsResNetModule
    else:
        raise ValueError(f"Module name {model_name} not supported")
