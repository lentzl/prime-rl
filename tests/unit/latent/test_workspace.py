from dataclasses import replace

import pytest
import torch

from prime_rl.latent.workspace import (
    DeliveryContext,
    WorkspaceEnvelope,
    WorkspaceValidationError,
    validate_workspace_delivery,
)

_E33 = "e33bd4cdbfd92eb22844dbbde2764aa7fa00e1cd25ca7045f91ce22210499e47"
_BRIDGE = "1" * 64
_CAPTURE = "2" * 64


def _envelope(**overrides: object) -> WorkspaceEnvelope:
    values: dict[str, object] = {
        "producer_session_id": "coordinator-parent",
        "producer_checkpoint_hash": _E33,
        "bridge_checkpoint_hash": _BRIDGE,
        "capture_spec_hash": _CAPTURE,
        "source_task_id": "task-1",
        "intended_task_id": "task-1",
        "scope_id": "scope-1",
        "parent_session_id": "coordinator-root",
        "creation_depth": 1,
        "allowed_receiver_sessions": ("coordinator-child",),
        "source_resource_labels": ("public:test",),
        "taint_labels": ("synthetic",),
        "created_at_utc": "2026-09-04T10:00:00Z",
        "expires_at_utc": "2026-09-04T11:00:00Z",
    }
    values.update(overrides)
    return WorkspaceEnvelope.create(torch.arange(12, dtype=torch.float32).reshape(3, 4), **values)


def _context(**overrides: object) -> DeliveryContext:
    values: dict[str, object] = {
        "receiver_session_id": "coordinator-child",
        "receiver_role": "coordinator",
        "task_id": "task-1",
        "scope_id": "scope-1",
        "expected_producer_session_id": "coordinator-parent",
        "expected_producer_parent_session_id": "coordinator-root",
        "expected_producer_checkpoint_hash": _E33,
        "expected_bridge_checkpoint_hash": _BRIDGE,
        "expected_capture_spec_hash": _CAPTURE,
        "now_utc": "2026-09-04T10:30:00Z",
    }
    values.update(overrides)
    return DeliveryContext(**values)


def test_valid_operational_workspace_is_deterministic_and_deliverable() -> None:
    first = _envelope()
    second = _envelope()

    assert first.metadata.workspace_id == second.metadata.workspace_id
    assert first.metadata.tensor_checksum == second.metadata.tensor_checksum
    validate_workspace_delivery(first, _context())


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("receiver_session_id", "sibling", "not authorized"),
        ("task_id", "task-2", "intended task"),
        ("scope_id", "scope-2", "scope"),
        ("expected_bridge_checkpoint_hash", "3" * 64, "bridge checkpoint"),
        ("now_utc", "2026-09-04T11:00:00Z", "expired"),
    ],
)
def test_delivery_fails_closed(field: str, value: str, match: str) -> None:
    with pytest.raises(WorkspaceValidationError, match=match):
        validate_workspace_delivery(_envelope(), _context(**{field: value}))


def test_tensor_tampering_fails_integrity_check() -> None:
    envelope = _envelope()
    tampered = WorkspaceEnvelope(metadata=envelope.metadata, tensor=envelope.tensor + 1)

    with pytest.raises(WorkspaceValidationError, match="checksum"):
        tampered.validate_integrity()


def test_operational_cross_task_workspace_is_rejected() -> None:
    with pytest.raises(WorkspaceValidationError, match="cross task"):
        _envelope(source_task_id="donor-task")


def test_moth_requires_explicit_causal_audit_context() -> None:
    envelope = _envelope(
        source_task_id="donor-task",
        delivery_mode="causal_audit",
        audit_arm="MOTH",
    )

    with pytest.raises(WorkspaceValidationError, match="delivery mode"):
        validate_workspace_delivery(envelope, _context())

    validate_workspace_delivery(
        envelope,
        _context(delivery_mode="causal_audit", expected_audit_arm="MOTH"),
    )


def test_metadata_identity_tampering_fails() -> None:
    envelope = _envelope()
    tampered = WorkspaceEnvelope(
        metadata=replace(envelope.metadata, allowed_receiver_sessions=("another-child",)),
        tensor=envelope.tensor,
    )

    with pytest.raises(WorkspaceValidationError, match="identity checksum"):
        tampered.validate_integrity()
