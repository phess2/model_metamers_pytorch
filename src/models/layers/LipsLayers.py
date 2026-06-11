import torch
import torch.nn as nn
import torch.nn.functional as F

from .bound_methods import BOUND_METHODS, get_projection_fn, resolve_bound_method
from .norm_ops import orthogonalize as _orthogonalize
from .norm_ops import power_iterate as _power_iterate
from .norm_ops import spectral_normalize as _spectral_normalize


# def batch_project(M, project_fn):
#     """
#     Batch project tensors of shape [..., d_out, d_in]
#     Adapted from lipschitz transformers code
#     """
#     m_shape = M.shape[-2:]
#     M_flattened = M.reshape((-1,) + m_shape)
#     M_projected = project_fn(M_flattened)
#     return M_projected.reshape(M.shape) / len(M_flattened)


def _soft_cap(matrix, alpha):
    """
    Apply min(1, x) approximately to the singular values of a single matrix
    Adapted from lipschitz transformers code
    """
    coeffs = [(1, -alpha), (1, alpha)]
    transpose = matrix.shape[1] > matrix.shape[0]
    if transpose:
        matrix = matrix.T
    for a, b in coeffs:
        A = matrix.T @ matrix
        identity_matrix = torch.eye(A.shape[0], dtype=A.dtype)
        matrix = matrix @ (a * identity_matrix + b * A)
    if transpose:
        matrix = matrix.T
    return matrix


def soft_cap_coupling(w_max, wd, max_update_norm):
    """
    Calculates the strength for soft cap that bounds singular values at w_max.
    Adapted from lipschitz transformers code
    """
    k = w_max * (1 - wd) + max_update_norm
    coeffs = torch.tensor([-(k**9), 3 * k**7, -3 * k**5, 0, k - w_max])
    roots = torch.roots(coeffs, strip_zeros=False)
    is_real = torch.abs(roots.imag) < 1e-6
    is_nonnegative = roots.real >= 0
    padded_reals = torch.where(
        is_real & is_nonnegative, roots.real, torch.ones_like(roots.real)
    )
    return torch.min(padded_reals)


# def soft_cap(M, alpha, **kwargs):
#     """
#     Apply soft cap to the singular values of a single matrix
#     Adapted from lipschitz transformers code
#     """
#     return batch_project(M, lambda x: _soft_cap(x, alpha=alpha, **kwargs))


# def spectral_normalize(M, **kwargs):
#     """
#     Normalize the singular values of M to 1
#     Adapted from lipschitz transformers code
#     """
#     return batch_project(M, lambda x: _spectral_normalize(x, **kwargs))


# def orthogonalize(M, **kwargs):
#     """
#     Orthogonalize a single matrix, always bfloat16. Credit for coefficients to @YouJiacheng and @leloykun.
#     """
#     return batch_project(M, lambda x: _orthogonalize(x, **kwargs))


def _no_projection(projection, w_max):
    """True when projection is disabled: either projection is None or w_max is 0."""
    return projection is None or w_max == 0


class LipsLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        w_max=1.0,
        projection=None,
        bound_method=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_max = w_max
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.projection = (
            projection
            # None | 'orthogonalize' | 'spectral_normalize' | 'modular_linf_cap' | callable
        )
        self.bound_method_name = resolve_bound_method(bound_method, projection)
        self._bound_method = BOUND_METHODS[self.bound_method_name]
        if _no_projection(projection, w_max):
            # Keep sqrt(out/in) for RMS -> RMS norm consistency; no w_max scaling
            self.scale = torch.sqrt(torch.tensor(self.out_features / self.in_features))
            self.lips_weight_scale = self.scale
        else:
            self.scale = (
                torch.sqrt(torch.tensor(self.out_features / self.in_features))
                * self.w_max
            )
            self.lips_weight_scale = (
                self.scale / self.w_max
            )  # the scale of the weight in the lipschitz bound doesn't change with w_max
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        self.weight.data = self.weight.data * self.scale

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    @torch.no_grad()
    def project_(self):
        """
        Project weights to enforce Lipschitz constraint.

        Returns:
            float: The norm-change ratio (||ΔW||_F / lips_weight_scale) / w_max,
                   which measures how much the projection changed the weights
                   relative to the allowed Lipschitz bound. Returns 0.0 if
                   projection is None or not applicable.
        """
        if _no_projection(self.projection, self.w_max):
            return 0.0

        # Cache weight before projection
        W_before = self.weight.data.clone()

        if isinstance(self.projection, str):
            if self.projection in ("orthogonalize", "spectral_normalize"):
                unscaled_weight = self.weight.data / self.scale
                if self.projection == "orthogonalize":
                    W = _orthogonalize(unscaled_weight)
                else:
                    W = _spectral_normalize(unscaled_weight)
                W = W * self.scale
                self.weight.copy_(W)
            elif self.projection in ("modular_linf_cap", "linf_cap"):
                project_fn = get_projection_fn(self.projection)
                self.weight.copy_(
                    project_fn(self.weight.data, self.scale, self.w_max)
                )
            else:
                return 0.0
        else:
            W = self.projection(self.weight.data / self.scale) * self.scale
            self.weight.copy_(W)

        # Compute norm-change ratio: (||ΔW||_F / lips_weight_scale) / w_max
        delta = self.weight.data - W_before
        delta_norm = delta.norm().item()
        effective_delta = delta_norm / self.lips_weight_scale.item()
        ratio = effective_delta / self.w_max
        return ratio

    def get_lips_bound(self):
        """Gets the Lipschitz bound of the linear layer."""
        return self._bound_method.linear_bound(
            self.weight.data, self.lips_weight_scale
        )


# class LipsHannPooling2d(nn.Module):
#     def __init__(self, stride, pool_size, padding=0, normalize=True):
#         super().__init__()
#         if isinstance(pool_size, int):
#             kh = kw = pool_size
#         else:
#             kh, kw = pool_size
#         self.stride = stride if isinstance(stride, tuple) else (stride, stride)
#         self.pool_size = (kh, kw)
#         self.padding = padding if isinstance(padding, tuple) else (padding, padding)
#         self.normalize = normalize
#         kernel_1d_h = torch.hann_window(kh)
#         kernel_1d_w = torch.hann_window(kw)
#         kernel_2d = torch.outer(kernel_1d_h, kernel_1d_w)
#         if normalize:
#             kernel_2d = kernel_2d / kernel_2d.sum()
#         self.register_buffer("kernel", kernel_2d)

#     def forward(self, x):
#         b, c, h, w = x.shape
#         k = self.kernel.to(x.dtype).to(x.device)
#         k = k.view(1, 1, *self.pool_size)
#         k = k.repeat(c, 1, 1, 1)  # depthwise
#         y = F.conv2d(
#             x, k, bias=None, stride=self.stride, padding=self.padding, groups=c
#         )
#         return y

#     @torch.no_grad()
#     def project_(self):
#         return


class LipsConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        w_max=1.0,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        projection=None,
        bound_method=None,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kh, kw)
        self.w_max = w_max
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = (
            dilation if isinstance(dilation, tuple) else (dilation, dilation)
        )
        self.groups = groups
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kh, kw)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.projection = (
            projection
            # None | 'orthogonalize' | 'spectral_normalize' | 'modular_linf_cap' | callable
        )
        self.bound_method_name = resolve_bound_method(bound_method, projection)
        self._bound_method = BOUND_METHODS[self.bound_method_name]
        if _no_projection(projection, w_max):
            # Keep sqrt(out/in) / (kh*kw) for RMS -> RMS norm consistency; no w_max scaling
            self.scale = torch.sqrt(torch.tensor(self.out_channels / self.in_channels))
            self.scale /= self.kernel_size[0] * self.kernel_size[1]
            self.lips_weight_scale = self.scale
        else:
            self.scale = (
                torch.sqrt(torch.tensor(self.out_channels / self.in_channels))
                * self.w_max
            )
            self.scale /= self.kernel_size[0] * self.kernel_size[1]
            self.lips_weight_scale = (
                self.scale / self.w_max
            )  # the scale of the weight in the lipschitz bound doesn't change with w_max
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        # always project the weight to the orthogonalized basis
        self.project_()

    def forward(self, x):
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    @torch.no_grad()
    def project_(self):
        """
        Project weights to enforce Lipschitz constraint.

        Returns:
            float: The norm-change ratio (||ΔW||_F / lips_weight_scale) / w_max,
                   which measures how much the projection changed the weights
                   relative to the allowed Lipschitz bound. Returns 0.0 if
                   projection is None or not applicable.
        """
        if _no_projection(self.projection, self.w_max):
            return 0.0

        # Cache weight before projection
        W_before = self.weight.data.clone()

        if isinstance(self.projection, str):
            W = self.weight
            _, _, kh, kw = W.shape
            if self.projection == "orthogonalize":
                for i in range(kh):
                    for j in range(kw):
                        slice_ij = W[:, :, i, j] / self.scale
                        W[:, :, i, j] = _orthogonalize(slice_ij) * self.scale
            elif self.projection == "spectral_normalize":
                for i in range(kh):
                    for j in range(kw):
                        slice_ij = W[:, :, i, j] / self.scale
                        W[:, :, i, j] = _spectral_normalize(slice_ij) * self.scale
            elif self.projection in ("modular_linf_cap", "linf_cap"):
                project_fn = get_projection_fn(self.projection)
                for i in range(kh):
                    for j in range(kw):
                        W[:, :, i, j] = project_fn(W[:, :, i, j], self.scale, self.w_max)
            else:
                return 0.0
        else:
            self.weight.copy_(self.projection(self.weight))

        # Compute norm-change ratio: (||ΔW||_F / lips_weight_scale) / w_max
        delta = self.weight.data - W_before
        delta_norm = delta.norm().item()
        effective_delta = delta_norm / self.lips_weight_scale.item()
        ratio = effective_delta / self.w_max
        return ratio

    def get_lips_bound(self):
        """Gets the Lipschitz bound of the convolutional layer."""
        return self._bound_method.conv_bound(
            self.weight.data, self.lips_weight_scale
        )
