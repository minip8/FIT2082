"""Boosting objectives: logits -> (gradient, hessian).

Swap in a different objective to change what the booster fits without touching
the hashing/accumulation core in `fit2082.boost.tables`.
"""

from typing import Protocol

import torch
import torch.nn.functional as F

# == protocol ==================================================================


class Objective(Protocol):
    """Maps raw logits to probabilities and to first/second order statistics."""

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """(n, k) logits -> (n, k) probabilities."""
        ...

    def gradients(
        self, probabilities: torch.Tensor, Y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(n, k) probabilities + (n,) labels -> (gradient, hessian), both (n, k)."""
        ...


# == softmax cross entropy =====================================================


class SoftmaxObjective:
    """Multiclass softmax cross entropy.

    Matches the original numba implementation: the gradient is `p - onehot(y)`
    and the hessian is `p * (1 - p)` with a floor for numerical stability.
    """

    def __init__(self, num_classes: int, hessian_floor: float = 1e-3) -> None:

        self.num_classes = num_classes
        self.hessian_floor = hessian_floor

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:

        return torch.softmax(logits, dim=-1)

    def gradients(
        self, probabilities: torch.Tensor, Y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        onehot = F.one_hot(Y, self.num_classes).to(probabilities.dtype)

        gradient = probabilities - onehot
        hessian = probabilities * (1 - probabilities) + self.hessian_floor

        return gradient, hessian
