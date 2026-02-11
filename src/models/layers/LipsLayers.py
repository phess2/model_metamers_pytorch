import torch
import torch.nn as nn
import torch.nn.functional as F


# def batch_project(M, project_fn):
#     """
#     Batch project tensors of shape [..., d_out, d_in]
#     Adapted from lipschitz transformers code
#     """
#     m_shape = M.shape[-2:]
#     M_flattened = M.reshape((-1,) + m_shape)
#     M_projected = project_fn(M_flattened)
#     return M_projected.reshape(M.shape) / len(M_flattened)


def _orthogonalize(M, **kwargs):
    """
    Orthogonalize a single matrix, always bfloat16. Credit for coefficients to @YouJiacheng and @leloykun.
    """
    abc_list = [
        (3955 / 1024, -8306 / 1024, 5008 / 1024),
        (3735 / 1024, -6681 / 1024, 3463 / 1024),
        (3799 / 1024, -6499 / 1024, 3211 / 1024),
        (4019 / 1024, -6385 / 1024, 2906 / 1024),
        (2677 / 1024, -3029 / 1024, 1162 / 1024),
        (2172 / 1024, -1833 / 1024, 682 / 1024),
    ]
    transpose = M.shape[1] > M.shape[0]
    if transpose:
        M = M.T
    M = M / (M.norm() + 1e-12)
    for a, b, c in abc_list:
        A = M.T @ M
        identity_matrix = torch.eye(A.shape[0], dtype=M.dtype)
        M = M @ (a * identity_matrix + b * A + c * A @ A)
    if transpose:
        M = M.T
    return M


def _soft_cap(M, alpha):
    """
    Apply min(1, x) approximately to the singular values of a single matrix
    Adapted from lipschitz transformers code
    """
    coeffs = [(1, -alpha), (1, alpha)]
    transpose = M.shape[1] > M.shape[0]
    if transpose:
        M = M.T
    for a, b in coeffs:
        A = M.T @ M
        identity_matrix = torch.eye(A.shape[0], dtype=A.dtype)
        M = M @ (a * identity_matrix + b * A)
    if transpose:
        M = M.T
    return M


def _power_iterate(M, num_iters=16):
    """
    Power iterate to find the largest singular value and vectors of a matrix
    """
    m, n = M.shape
    device = M.device
    dtype = M.dtype
    if m < n:
        u = torch.randn((m,), device=device, dtype=dtype)
        u = u / (u.norm() + 1e-12)
        for _ in range(num_iters):
            w = M @ (M.T @ u)
            u = w / (w.norm() + 1e-12)
        MTu = M.T @ u
        sigma = MTu.norm()
        v = MTu / (sigma + 1e-12)
    else:
        v = torch.randn((n,), device=device, dtype=dtype)
        v = v / (v.norm() + 1e-12)
        for _ in range(num_iters):
            w = M.T @ (M @ v)
            v = w / (w.norm() + 1e-12)
        Mv = M @ v
        sigma = Mv.norm()
        u = Mv / (sigma + 1e-12)
    return u, sigma, v


def _spectral_normalize(M):
    """
    Normalize the singular values of M to 1
    Adapted from lipschitz transformers code
    """
    _, sigma_max, _ = _power_iterate(M)
    sigma_clamped = torch.clamp(sigma_max, min=1.0)
    return M / sigma_clamped


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
        self, in_features, out_features, bias=False, w_max=1.0, projection=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_max = w_max
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.projection = (
            projection  # None | 'orthogonalize' | 'spectral_normalize' | callable
        )
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

        unscaled_weight = self.weight.data / self.scale
        if isinstance(self.projection, str):
            if self.projection == "orthogonalize":
                W = _orthogonalize(unscaled_weight)
            elif self.projection == "spectral_normalize":
                W = _spectral_normalize(unscaled_weight)
            else:
                return 0.0
        else:
            W = self.projection(unscaled_weight)
        W = W * self.scale
        self.weight.copy_(W)

        # Compute norm-change ratio: (||ΔW||_F / lips_weight_scale) / w_max
        delta = self.weight.data - W_before
        delta_norm = delta.norm().item()
        effective_delta = delta_norm / self.lips_weight_scale.item()
        ratio = effective_delta / self.w_max
        return ratio

    def get_lips_bound(self):
        """
        Gets the Lipschitz bound of the linear layer
        """
        W = self.weight.data
        W = W / self.lips_weight_scale
        _, sigma_max, _ = _power_iterate(W)
        # || W ||*
        return sigma_max


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
            projection  # None | 'orthogonalize' | 'spectral_normalize' | callable
        )
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
            oc, icg, kh, kw = W.shape
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
            else:
                return 0.0
        else:
            W = self.projection(self.weight)
            self.weight.copy_(W)

        # Compute norm-change ratio: (||ΔW||_F / lips_weight_scale) / w_max
        delta = self.weight.data - W_before
        delta_norm = delta.norm().item()
        effective_delta = delta_norm / self.lips_weight_scale.item()
        ratio = effective_delta / self.w_max
        return ratio

    def get_lips_bound(self):
        """
        Gets the Lipschitz bound of the convolutional layer
        """
        # getting the RMS -> RMS operator norm
        W = self.weight.data
        oc, icg, kh, kw = W.shape
        max_val = float("-inf")
        for i in range(kh):
            for j in range(kw):
                slice_ij = W[:, :, i, j]
                slice_ij = slice_ij / self.lips_weight_scale
                _, sigma_max, _ = _power_iterate(slice_ij)
                max_val = max(max_val, sigma_max)
        # max ||C..ij||*
        return max_val
