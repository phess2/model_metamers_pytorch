"""Shared matrix normalization helpers for Lipschitz layers."""

import torch


def orthogonalize(matrix, **kwargs):
    """
    Orthogonalize a single matrix, always bfloat16.
    Credit for coefficients to @YouJiacheng and @leloykun.
    """
    del kwargs
    abc_list = [
        (3955 / 1024, -8306 / 1024, 5008 / 1024),
        (3735 / 1024, -6681 / 1024, 3463 / 1024),
        (3799 / 1024, -6499 / 1024, 3211 / 1024),
        (4019 / 1024, -6385 / 1024, 2906 / 1024),
        (2677 / 1024, -3029 / 1024, 1162 / 1024),
        (2172 / 1024, -1833 / 1024, 682 / 1024),
    ]
    transpose = matrix.shape[1] > matrix.shape[0]
    if transpose:
        matrix = matrix.T
    matrix = matrix / (matrix.norm() + 1e-12)
    for a, b, c in abc_list:
        A = matrix.T @ matrix
        identity_matrix = torch.eye(A.shape[0], dtype=matrix.dtype)
        matrix = matrix @ (a * identity_matrix + b * A + c * A @ A)
    if transpose:
        matrix = matrix.T
    return matrix


def power_iterate(matrix, num_iters=16):
    """Power iterate to find the largest singular value and vectors of a matrix."""
    m, n = matrix.shape
    device = matrix.device
    dtype = matrix.dtype
    if m < n:
        u = torch.randn((m,), device=device, dtype=dtype)
        u = u / (u.norm() + 1e-12)
        for _ in range(num_iters):
            w = matrix @ (matrix.T @ u)
            u = w / (w.norm() + 1e-12)
        MTu = matrix.T @ u
        sigma = MTu.norm()
        v = MTu / (sigma + 1e-12)
    else:
        v = torch.randn((n,), device=device, dtype=dtype)
        v = v / (v.norm() + 1e-12)
        for _ in range(num_iters):
            w = matrix.T @ (matrix @ v)
            v = w / (w.norm() + 1e-12)
        matrix_v = matrix @ v
        sigma = matrix_v.norm()
        u = matrix_v / (sigma + 1e-12)
    return u, sigma, v


def spectral_normalize(matrix):
    """Normalize the singular values of M to 1."""
    _, sigma_max, _ = power_iterate(matrix)
    sigma_clamped = torch.clamp(sigma_max, min=1.0)
    return matrix / sigma_clamped
