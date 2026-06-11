"""Audio adversarial attack utilities (classification BCE and legacy representation PGD)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, cast

import torch
import torch.nn.functional as F
from torch import Tensor

NormType = Literal["l2", "linf"]
AttackLossType = Literal["normalized_l2", "l2", "squared_l2", "cosine"]
ClassificationLossType = Literal["bce_multihot"]


@dataclass(frozen=True)
class AdversarialAttackConfig:
    """Configuration for bounded input-space audio classification attacks."""

    norm: NormType = "l2"
    epsilon: float = 0.05
    step_size: float = 0.005
    num_steps: int = 100
    num_random_starts: int = 1
    clamp_range: Tuple[float, float] = (-1.0, 1.0)
    seed: Optional[int] = None
    grad_eps: float = 1e-12
    classification_loss: ClassificationLossType = "bce_multihot"


@dataclass(frozen=True)
class RepresentationAttackConfig:
    """Configuration for bounded representation-space audio attacks (legacy)."""

    norm: NormType = "l2"
    epsilon: float = 0.05
    step_size: float = 0.005
    num_steps: int = 100
    num_random_starts: int = 1
    loss_type: AttackLossType = "normalized_l2"
    clamp_range: Tuple[float, float] = (-1.0, 1.0)
    seed: Optional[int] = None
    grad_eps: float = 1e-12


def _flatten_batch(x: Tensor) -> Tensor:
    return x.view(x.shape[0], -1)


def build_multihot_target(
    num_classes: int,
    label_indices: Sequence[int],
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build a multi-hot target vector for multilabel BCE attacks."""
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    if not label_indices:
        raise ValueError("true_label_indices must be non-empty for classification attacks.")
    target = torch.zeros(num_classes, device=device, dtype=dtype)
    for label_idx in label_indices:
        idx = int(label_idx)
        if idx < 0 or idx >= num_classes:
            raise ValueError(
                f"Label index {idx} out of range for num_classes={num_classes}."
            )
        target[idx] = 1.0
    return target


def classification_loss(logits: Tensor, target_multihot: Tensor) -> Tensor:
    """Multilabel BCE loss (mean over classes) for batch size 1 or B."""
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if target_multihot.dim() == 1:
        target_multihot = target_multihot.unsqueeze(0)
    if logits.shape != target_multihot.shape:
        raise ValueError(
            f"Mismatched logits and target shapes: {tuple(logits.shape)} vs "
            f"{tuple(target_multihot.shape)}"
        )
    return F.binary_cross_entropy_with_logits(logits, target_multihot, reduction="mean")


def representation_distance(
    current: Tensor,
    reference: Tensor,
    loss_type: AttackLossType = "normalized_l2",
    eps: float = 1e-12,
    reduction: Literal["none", "mean"] = "mean",
) -> Tensor:
    """Compute representation-space distance between tensors."""
    if current.shape[0] != reference.shape[0]:
        raise ValueError(
            f"Mismatched batch dimensions: current={current.shape}, reference={reference.shape}"
        )

    current_flat = _flatten_batch(current)
    reference_flat = _flatten_batch(reference)

    if loss_type == "normalized_l2":
        numerator = torch.norm(current_flat - reference_flat, p=2, dim=1)
        denominator = torch.norm(reference_flat, p=2, dim=1) + eps
        per_sample = numerator / denominator
    elif loss_type == "l2":
        per_sample = torch.norm(current_flat - reference_flat, p=2, dim=1)
    elif loss_type == "squared_l2":
        per_sample = torch.sum((current_flat - reference_flat) ** 2, dim=1)
    elif loss_type == "cosine":
        cosine = F.cosine_similarity(current_flat, reference_flat, dim=1, eps=eps)
        per_sample = 1.0 - cosine
    else:
        raise ValueError(f"Unknown loss_type={loss_type!r}")

    if reduction == "mean":
        return per_sample.mean()
    return per_sample


def project_l2(delta: Tensor, epsilon: float, eps: float = 1e-12) -> Tensor:
    """Project perturbations into an L2 ball of radius ``epsilon``."""
    if epsilon <= 0:
        return torch.zeros_like(delta)
    delta_flat = _flatten_batch(delta)
    norms = torch.norm(delta_flat, p=2, dim=1, keepdim=True)
    scale = torch.clamp(epsilon / (norms + eps), max=1.0)
    return (delta_flat * scale).view_as(delta)


def project_linf(delta: Tensor, epsilon: float) -> Tensor:
    """Project perturbations into an Linf ball of radius ``epsilon``."""
    if epsilon <= 0:
        return torch.zeros_like(delta)
    return delta.clamp(-epsilon, epsilon)


def perturbation_norms(delta: Tensor) -> Dict[str, Tensor]:
    """Return per-sample perturbation norms."""
    delta_flat = _flatten_batch(delta)
    return {
        "l2": torch.norm(delta_flat, p=2, dim=1),
        "linf": torch.norm(delta_flat, p=float("inf"), dim=1),
    }


class _BoundedWaveformPGD:
    """Shared PGD mechanics for waveform perturbations."""

    def __init__(self, config: AdversarialAttackConfig | RepresentationAttackConfig, device: str):
        self.config = config
        self.device = device

    def _project(self, delta: Tensor) -> Tensor:
        if self.config.norm == "l2":
            return project_l2(delta, epsilon=self.config.epsilon, eps=self.config.grad_eps)
        if self.config.norm == "linf":
            return project_linf(delta, epsilon=self.config.epsilon)
        raise ValueError(f"Unsupported norm={self.config.norm!r}")

    def _step_update(self, grad: Tensor) -> Tensor:
        if self.config.norm == "linf":
            return self.config.step_size * grad.sign()
        grad_flat = _flatten_batch(grad)
        grad_norm = torch.norm(grad_flat, p=2, dim=1, keepdim=True)
        normalized = grad_flat / (grad_norm + self.config.grad_eps)
        return self.config.step_size * normalized.view_as(grad)

    def _initial_delta(self, shape: torch.Size, start_idx: int) -> Tensor:
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed + start_idx)
        noise = torch.randn(shape, device=self.device)
        if self.config.norm == "linf":
            return project_linf(noise * self.config.epsilon, epsilon=self.config.epsilon)
        projected = project_l2(noise, epsilon=self.config.epsilon, eps=self.config.grad_eps)
        random_scale = torch.rand((shape[0],) + (1,) * (len(shape) - 1), device=self.device)
        return projected * random_scale


class AudioAdversarialAttacker(_BoundedWaveformPGD):
    """Untargeted PGD on classifier logits via multilabel BCE."""

    def __init__(
        self,
        model: torch.nn.Module,
        sample_rate: int,
        config: AdversarialAttackConfig,
        device: str = "cuda",
    ) -> None:
        super().__init__(config=config, device=device)
        self.model = model
        self.sample_rate = int(sample_rate)

    def _get_logits(self, waveform: Tensor) -> Tensor:
        waveform = waveform.to(self.device)
        logits = cast(Any, self.model).get_classifier_logits(waveform, sr=self.sample_rate)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        return logits

    def _resolve_num_classes(self, waveform: Tensor) -> int:
        logits = self._get_logits(waveform)
        return int(logits.shape[-1])

    def _run_single_attack(
        self,
        source_waveform: Tensor,
        target_multihot: Tensor,
        start_idx: int,
    ) -> Tuple[Tensor, Dict]:
        clamp_min, clamp_max = self.config.clamp_range
        source_waveform = source_waveform.detach().to(self.device)
        target_multihot = target_multihot.detach().to(self.device)

        delta = self._initial_delta(source_waveform.shape, start_idx=start_idx)
        delta = self._project(delta)
        step_records = []

        for step in range(self.config.num_steps):
            adv_waveform = (source_waveform + delta).clamp(clamp_min, clamp_max)
            adv_waveform = adv_waveform.detach().requires_grad_(True)

            logits = self._get_logits(adv_waveform)
            loss = classification_loss(logits, target_multihot)
            grad = torch.autograd.grad(loss, adv_waveform)[0]

            with torch.no_grad():
                delta = delta + self._step_update(grad)
                delta = self._project(delta)
                clipped_adv = (source_waveform + delta).clamp(clamp_min, clamp_max)
                delta = clipped_adv - source_waveform
                norms = perturbation_norms(delta)
                step_records.append(
                    {
                        "step": step,
                        "classification_loss": float(loss.detach().item()),
                        "delta_l2_mean": float(norms["l2"].mean().item()),
                        "delta_linf_mean": float(norms["linf"].mean().item()),
                    }
                )

        final_adv = (source_waveform + delta).clamp(clamp_min, clamp_max).detach()
        final_logits = self._get_logits(final_adv).detach()
        final_loss = classification_loss(final_logits, target_multihot)
        final_norms = perturbation_norms(final_adv - source_waveform)
        metadata = {
            "start_idx": start_idx,
            "final_classification_loss": float(final_loss.item()),
            "final_delta_l2_mean": float(final_norms["l2"].mean().item()),
            "final_delta_linf_mean": float(final_norms["linf"].mean().item()),
            "steps": step_records,
        }
        return final_adv, metadata

    def attack_untargeted(
        self,
        source_waveform: Tensor,
        true_label_indices: Sequence[int],
    ) -> Tuple[Tensor, Dict]:
        """Maximize multilabel BCE against true-class multi-hot targets."""
        source_waveform = source_waveform.to(self.device)
        num_classes = self._resolve_num_classes(source_waveform)
        target_multihot = build_multihot_target(
            num_classes=num_classes,
            label_indices=true_label_indices,
            device=self.device,
        )

        with torch.no_grad():
            initial_logits = self._get_logits(source_waveform)
            initial_loss = classification_loss(initial_logits, target_multihot)

        best_adv: Optional[Tensor] = None
        best_meta: Optional[Dict] = None
        best_score: Optional[float] = None

        num_starts = max(1, self.config.num_random_starts)
        for start_idx in range(num_starts):
            candidate_adv, candidate_meta = self._run_single_attack(
                source_waveform=source_waveform,
                target_multihot=target_multihot,
                start_idx=start_idx,
            )
            score = float(candidate_meta["final_classification_loss"])
            if best_score is None or score > best_score:
                best_adv = candidate_adv
                best_meta = candidate_meta
                best_score = score

        if best_adv is None or best_meta is None:
            raise RuntimeError("Attack failed to produce a candidate adversarial waveform.")

        result = {
            "best_start_idx": best_meta["start_idx"],
            "best_final_classification_loss": best_meta["final_classification_loss"],
            "initial_classification_loss": float(initial_loss.item()),
            "best_delta_l2_mean": best_meta["final_delta_l2_mean"],
            "best_delta_linf_mean": best_meta["final_delta_linf_mean"],
            "num_random_starts": num_starts,
            "norm": self.config.norm,
            "epsilon": self.config.epsilon,
            "step_size": self.config.step_size,
            "num_steps": self.config.num_steps,
            "classification_loss": self.config.classification_loss,
            "steps": best_meta["steps"],
        }
        return best_adv, result


class AudioRepresentationAttacker(_BoundedWaveformPGD):
    """Legacy projected-gradient attacks on intermediate audio representations."""

    def __init__(
        self,
        model: torch.nn.Module,
        layer_name: str,
        sample_rate: int,
        config: RepresentationAttackConfig,
        device: str = "cuda",
    ) -> None:
        super().__init__(config=config, device=device)
        self.model = model
        self.layer_name = layer_name
        self.sample_rate = int(sample_rate)

    def extract_representation(self, waveform: Tensor) -> Tensor:
        waveform = waveform.to(self.device)
        _, all_outputs = cast(Any, self.model).forward_with_representations(
            waveform, sr=self.sample_rate, fake_relu=False
        )
        if self.layer_name not in all_outputs:
            available_layers = sorted(all_outputs.keys())
            raise KeyError(
                f"Layer '{self.layer_name}' not found. Available: {available_layers}"
            )
        return all_outputs[self.layer_name]

    def _run_single_attack(
        self,
        source_waveform: Tensor,
        reference_rep: Tensor,
        minimize_distance: bool,
        start_idx: int,
    ) -> Tuple[Tensor, Dict]:
        clamp_min, clamp_max = self.config.clamp_range
        source_waveform = source_waveform.detach().to(self.device)
        reference_rep = reference_rep.detach().to(self.device)
        rep_config = cast(RepresentationAttackConfig, self.config)

        delta = self._initial_delta(source_waveform.shape, start_idx=start_idx)
        delta = self._project(delta)
        step_records = []

        for step in range(rep_config.num_steps):
            adv_waveform = (source_waveform + delta).clamp(clamp_min, clamp_max)
            adv_waveform = adv_waveform.detach().requires_grad_(True)

            current_rep = self.extract_representation(adv_waveform)
            distance = representation_distance(
                current=current_rep,
                reference=reference_rep,
                loss_type=rep_config.loss_type,
                eps=rep_config.grad_eps,
                reduction="mean",
            )
            objective = -distance if minimize_distance else distance
            grad = torch.autograd.grad(objective, adv_waveform)[0]

            with torch.no_grad():
                delta = delta + self._step_update(grad)
                delta = self._project(delta)
                clipped_adv = (source_waveform + delta).clamp(clamp_min, clamp_max)
                delta = clipped_adv - source_waveform
                norms = perturbation_norms(delta)
                step_records.append(
                    {
                        "step": step,
                        "distance": float(distance.detach().item()),
                        "delta_l2_mean": float(norms["l2"].mean().item()),
                        "delta_linf_mean": float(norms["linf"].mean().item()),
                    }
                )

        final_adv = (source_waveform + delta).clamp(clamp_min, clamp_max).detach()
        final_rep = self.extract_representation(final_adv).detach()
        final_distance = representation_distance(
            current=final_rep,
            reference=reference_rep,
            loss_type=rep_config.loss_type,
            eps=rep_config.grad_eps,
            reduction="mean",
        )
        final_norms = perturbation_norms(final_adv - source_waveform)
        metadata = {
            "start_idx": start_idx,
            "final_distance": float(final_distance.item()),
            "final_delta_l2_mean": float(final_norms["l2"].mean().item()),
            "final_delta_linf_mean": float(final_norms["linf"].mean().item()),
            "steps": step_records,
        }
        return final_adv, metadata

    def _attack(
        self,
        source_waveform: Tensor,
        reference_rep: Tensor,
        minimize_distance: bool,
    ) -> Tuple[Tensor, Dict]:
        rep_config = cast(RepresentationAttackConfig, self.config)
        best_adv: Optional[Tensor] = None
        best_meta: Optional[Dict] = None
        best_score: Optional[float] = None

        num_starts = max(1, rep_config.num_random_starts)
        for start_idx in range(num_starts):
            candidate_adv, candidate_meta = self._run_single_attack(
                source_waveform=source_waveform,
                reference_rep=reference_rep,
                minimize_distance=minimize_distance,
                start_idx=start_idx,
            )
            score = float(candidate_meta["final_distance"])
            if best_score is None:
                pick = True
            elif minimize_distance:
                pick = score < best_score
            else:
                pick = score > best_score
            if pick:
                best_adv = candidate_adv
                best_meta = candidate_meta
                best_score = score

        if best_adv is None or best_meta is None:
            raise RuntimeError("Attack failed to produce a candidate adversarial waveform.")

        result = {
            "best_start_idx": best_meta["start_idx"],
            "best_final_distance": best_meta["final_distance"],
            "best_delta_l2_mean": best_meta["final_delta_l2_mean"],
            "best_delta_linf_mean": best_meta["final_delta_linf_mean"],
            "num_random_starts": num_starts,
            "norm": rep_config.norm,
            "epsilon": rep_config.epsilon,
            "step_size": rep_config.step_size,
            "num_steps": rep_config.num_steps,
            "loss_type": rep_config.loss_type,
            "steps": best_meta["steps"],
        }
        return best_adv, result

    def attack_untargeted(
        self, source_waveform: Tensor, source_rep: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict]:
        """Maximize representation distance from the original sample."""
        if source_rep is None:
            source_rep = self.extract_representation(source_waveform).detach()
        return self._attack(
            source_waveform=source_waveform,
            reference_rep=source_rep,
            minimize_distance=False,
        )

    def attack_targeted(
        self,
        source_waveform: Tensor,
        target_waveform: Optional[Tensor] = None,
        target_rep: Optional[Tensor] = None,
        source_rep: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict]:
        """Minimize representation distance to a target example."""
        rep_config = cast(RepresentationAttackConfig, self.config)
        if target_rep is None:
            if target_waveform is None:
                raise ValueError("Provide either target_waveform or target_rep for targeted attack.")
            target_rep = self.extract_representation(target_waveform).detach()
        if source_rep is None:
            source_rep = self.extract_representation(source_waveform).detach()

        adv_waveform, metadata = self._attack(
            source_waveform=source_waveform,
            reference_rep=target_rep,
            minimize_distance=True,
        )
        adv_rep = self.extract_representation(adv_waveform).detach()
        source_distance = representation_distance(
            adv_rep,
            source_rep,
            loss_type=rep_config.loss_type,
            eps=rep_config.grad_eps,
            reduction="mean",
        )
        target_distance = representation_distance(
            adv_rep,
            target_rep,
            loss_type=rep_config.loss_type,
            eps=rep_config.grad_eps,
            reduction="mean",
        )
        metadata["adv_to_source_distance"] = float(source_distance.item())
        metadata["adv_to_target_distance"] = float(target_distance.item())
        return adv_waveform, metadata


def audit_waveform_gradient(
    model: torch.nn.Module,
    waveform: Tensor,
    sample_rate: int,
    layer_name: str | None = None,
    true_label_indices: Sequence[int] | None = None,
    device: str = "cuda",
    use_logits: bool = False,
) -> Dict[str, Any]:
    """Check that gradients flow from logits or a layer back to waveform input."""
    waveform = waveform.detach().to(device).requires_grad_(True)

    if use_logits:
        if true_label_indices is None:
            raise ValueError("true_label_indices required when use_logits=True.")
        logits = cast(Any, model).get_classifier_logits(waveform, sr=sample_rate)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        num_classes = int(logits.shape[-1])
        target = build_multihot_target(
            num_classes=num_classes,
            label_indices=true_label_indices,
            device=device,
            dtype=logits.dtype,
        )
        if target.dim() == 1:
            target = target.unsqueeze(0)
        scalar = classification_loss(logits, target)
        audit_layer = "logits"
    else:
        if layer_name is None:
            raise ValueError("layer_name is required when use_logits=False.")
        _, reps = cast(Any, model).forward_with_representations(
            waveform, sr=sample_rate, fake_relu=False
        )
        if layer_name not in reps:
            available_layers = sorted(reps.keys())
            raise KeyError(f"Unknown layer '{layer_name}'. Available: {available_layers}")
        scalar = reps[layer_name].sum()
        audit_layer = layer_name

    scalar.backward()
    grad = waveform.grad
    if grad is None:
        raise RuntimeError("Expected non-None waveform gradient, got None.")
    nonzero_count = int((grad.abs() > 0).sum().item())
    finite = bool(torch.isfinite(grad).all().item())
    return {
        "layer_name": audit_layer,
        "use_logits": use_logits,
        "gradient_is_finite": finite,
        "gradient_nonzero_count": nonzero_count,
        "gradient_shape": list(grad.shape),
        "gradient_abs_mean": float(grad.abs().mean().item()),
        "gradient_abs_max": float(grad.abs().max().item()),
    }
