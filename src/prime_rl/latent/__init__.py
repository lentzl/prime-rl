"""Trainable latent modules around immutable language-model policies."""

from prime_rl.latent.local_depth import LocalDepthCodec, LocalDepthInputs, compose_local_depth_inputs
from prime_rl.latent.recurrent import (
    OneShotFeedForwardSidecar,
    RecurrentDiagnostics,
    RecurrentState,
    TimestepFreeRecurrentSidecar,
    diagnose_recurrent_states,
)

__all__ = [
    "LocalDepthCodec",
    "LocalDepthInputs",
    "OneShotFeedForwardSidecar",
    "RecurrentDiagnostics",
    "RecurrentState",
    "TimestepFreeRecurrentSidecar",
    "compose_local_depth_inputs",
    "diagnose_recurrent_states",
]
