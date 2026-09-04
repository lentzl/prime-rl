"""Trainable latent modules around immutable language-model policies."""

from prime_rl.latent.recurrent import (
    OneShotFeedForwardSidecar,
    RecurrentDiagnostics,
    RecurrentState,
    TimestepFreeRecurrentSidecar,
    diagnose_recurrent_states,
)

__all__ = [
    "OneShotFeedForwardSidecar",
    "RecurrentDiagnostics",
    "RecurrentState",
    "TimestepFreeRecurrentSidecar",
    "diagnose_recurrent_states",
]
