from .lipsalexnet import LipsAlexNetModule


def get_module(config: dict):
    model_name = config["model_name"]
    if model_name == "lipsalexnet":
        return LipsAlexNetModule
    else:
        raise ValueError(f"Module name {model_name} not supported")
