import torch
from torch import nn

from .layers.LipsLayers import LipsConv2d, LipsLinear


class LipsModel(nn.Module):
    """
    Base class for Lipschitz-constrained models.

    Provides shared functionality for weight projection and Lipschitz bound
    computation by walking ``named_modules()`` for ``LipsConv2d`` and
    ``LipsLinear`` layers. Concrete subclasses only need to define
    ``__init__`` (architecture) and ``forward``.
    """

    def get_lips_layers(self):
        """Yield ``(name, module)`` for every Lipschitz-constrained layer."""
        for name, module in self.named_modules():
            if isinstance(module, (LipsConv2d, LipsLinear)):
                yield name, module

    @torch.no_grad()
    def project_weights(self):
        """
        Project all Lipschitz layers to enforce their constraints.

        Returns:
            dict[str, float]: Mapping of layer name to the norm-change ratio
            returned by each layer's ``project_()`` method.
        """
        ratios = {}
        for name, module in self.get_lips_layers():
            ratio = module.project_()
            ratios[name] = float(ratio)
        return ratios

    def get_lips_bound(self):
        """
        Compute the overall Lipschitz bound of the model as the product of
        per-layer spectral norms.

        Returns:
            float: The composite Lipschitz bound.
        """
        bound = 1.0
        for _name, module in self.get_lips_layers():
            bound *= module.get_lips_bound()
        return bound

    # ------------------------------------------------------------------
    # Metamer-generation interface
    # ------------------------------------------------------------------

    def __init__(self):
        self.metamer_layers: list[str] = []


    def forward_with_representations(self, x, fake_relu=False):
        """Return ``(logits, dict[str, Tensor])`` with all intermediate
        activations.  Subclasses must implement this.
        """
        raise NotImplementedError

    @classmethod
    def list_representation_layers(cls):
        """Return the list of recommended metamer-generation layers."""
        return list(cls.metamer_layers)
