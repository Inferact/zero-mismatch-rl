"""Shared contracts for K31 vLLM-visible training bridges (copied from DS4)."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def own_visible_tensor(value: torch.Tensor) -> torch.Tensor:
    """Make a deployment result eligible for an outer autograd owner.

    Cloning an inference tensor after the deployment call preserves its value
    while avoiding inference-mode tensor restrictions. Ordinary tensors are
    returned as-is so an exact/path gate observes no extra numerical work.
    """

    if not isinstance(value, torch.Tensor):
        raise TypeError("a vLLM training bridge must own a tensor output")
    return value.clone() if value.is_inference() else value


def parameter_versions(parameters: Iterable[torch.Tensor]) -> tuple[int, ...]:
    # vLLM audit/eager inference constructs inference tensors, which
    # intentionally have no version counter. Training tensors still retain
    # the mutation guard used by custom backward.
    return tuple(
        -1 if parameter.is_inference() else parameter._version
        for parameter in parameters
    )


def check_parameter_versions(
    parameters: Iterable[torch.Tensor], expected: tuple[int, ...]
) -> None:
    parameters = tuple(parameters)
    actual = parameter_versions(parameters)
    if actual != expected:
        changed = [
            index
            for index, (before, after) in enumerate(zip(expected, actual, strict=True))
            if before != after
        ]
        raise RuntimeError(
            "K31 vLLM master parameter changed between forward and backward; "
            f"changed dependency indices={changed}, versions={expected}->{actual}"
        )


__all__ = [
    "check_parameter_versions",
    "own_visible_tensor",
    "parameter_versions",
]
