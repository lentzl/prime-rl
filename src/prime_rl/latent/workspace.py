from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import Literal

import torch

LATENT_WORKSPACE_SCHEMA_VERSION = "prime-rl/latent-workspace/v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_DTYPE_TO_NAME = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}
_METADATA_FIELDS = {
    "schema_version",
    "workspace_id",
    "producer_session_id",
    "producer_role",
    "producer_checkpoint_hash",
    "bridge_checkpoint_hash",
    "capture_spec_hash",
    "source_task_id",
    "intended_task_id",
    "scope_id",
    "parent_session_id",
    "creation_depth",
    "allowed_receiver_sessions",
    "source_resource_labels",
    "taint_labels",
    "created_at_utc",
    "expires_at_utc",
    "shape",
    "dtype",
    "tensor_checksum",
    "delivery_mode",
    "audit_arm",
}

DeliveryMode = Literal["operational", "causal_audit"]
AuditArm = Literal["MOTH", "MSELF", "MCUR", "ZERO", "NOISE"]


class WorkspaceValidationError(ValueError):
    """Raised when a latent workspace fails a control-plane invariant."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _require_identifier(name: str, value: str) -> None:
    if not value or len(value) > 256 or value.strip() != value:
        raise WorkspaceValidationError(f"{name} must be a non-empty canonical identifier")


def _require_sha256(name: str, value: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise WorkspaceValidationError(f"{name} must be a lowercase SHA-256 digest")


def _parse_utc(name: str, value: str) -> datetime:
    if not _UTC_RE.fullmatch(value):
        raise WorkspaceValidationError(f"{name} must use second-precision UTC RFC3339")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def tensor_checksum(tensor: torch.Tensor) -> str:
    if tensor.dtype not in _DTYPE_TO_NAME:
        raise WorkspaceValidationError(f"unsupported workspace dtype: {tensor.dtype}")
    if tensor.ndim != 2:
        raise WorkspaceValidationError("workspace tensor must have rank 2")
    if not torch.isfinite(tensor).all().item():
        raise WorkspaceValidationError("workspace tensor must contain only finite values")
    canonical = tensor.detach().to(device="cpu").contiguous()
    header = _canonical_json({"dtype": _DTYPE_TO_NAME[canonical.dtype], "shape": list(canonical.shape)})
    payload = canonical.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(header + b"\0" + payload).hexdigest()


@dataclass(frozen=True, slots=True)
class WorkspaceMetadata:
    schema_version: str
    workspace_id: str
    producer_session_id: str
    producer_role: Literal["coordinator"]
    producer_checkpoint_hash: str
    bridge_checkpoint_hash: str
    capture_spec_hash: str
    source_task_id: str
    intended_task_id: str
    scope_id: str
    parent_session_id: str
    creation_depth: int
    allowed_receiver_sessions: tuple[str, ...]
    source_resource_labels: tuple[str, ...]
    taint_labels: tuple[str, ...]
    created_at_utc: str
    expires_at_utc: str
    shape: tuple[int, int]
    dtype: str
    tensor_checksum: str
    delivery_mode: DeliveryMode = "operational"
    audit_arm: AuditArm | None = None

    def validate(self) -> None:
        if self.schema_version != LATENT_WORKSPACE_SCHEMA_VERSION:
            raise WorkspaceValidationError("unknown workspace schema version")
        _require_sha256("workspace_id", self.workspace_id)
        _require_sha256("producer_checkpoint_hash", self.producer_checkpoint_hash)
        _require_sha256("bridge_checkpoint_hash", self.bridge_checkpoint_hash)
        _require_sha256("capture_spec_hash", self.capture_spec_hash)
        _require_sha256("tensor_checksum", self.tensor_checksum)
        for name in (
            "producer_session_id",
            "source_task_id",
            "intended_task_id",
            "scope_id",
            "parent_session_id",
        ):
            _require_identifier(name, getattr(self, name))
        if self.producer_role != "coordinator":
            raise WorkspaceValidationError("v1 permits coordinator producers only")
        if self.creation_depth < 0:
            raise WorkspaceValidationError("creation_depth must be non-negative")
        if not self.allowed_receiver_sessions:
            raise WorkspaceValidationError("receiver allowlist must not be empty")
        for receiver in self.allowed_receiver_sessions:
            _require_identifier("allowed_receiver_session", receiver)
        if len(set(self.allowed_receiver_sessions)) != len(self.allowed_receiver_sessions):
            raise WorkspaceValidationError("receiver allowlist must not contain duplicates")
        if tuple(sorted(self.allowed_receiver_sessions)) != self.allowed_receiver_sessions:
            raise WorkspaceValidationError("receiver allowlist must be canonically sorted")
        if tuple(sorted(set(self.source_resource_labels))) != self.source_resource_labels:
            raise WorkspaceValidationError("source resource labels must be unique and sorted")
        if tuple(sorted(set(self.taint_labels))) != self.taint_labels:
            raise WorkspaceValidationError("taint labels must be unique and sorted")
        if len(self.shape) != 2 or not (1 <= self.shape[0] <= 64 and 1 <= self.shape[1] <= 4096):
            raise WorkspaceValidationError("workspace shape is outside the v1 bounds")
        if self.dtype not in set(_DTYPE_TO_NAME.values()):
            raise WorkspaceValidationError("workspace dtype is unsupported")
        created_at = _parse_utc("created_at_utc", self.created_at_utc)
        expires_at = _parse_utc("expires_at_utc", self.expires_at_utc)
        if expires_at <= created_at:
            raise WorkspaceValidationError("workspace expiry must follow creation")
        if self.delivery_mode == "operational":
            if self.audit_arm is not None:
                raise WorkspaceValidationError("operational workspaces cannot declare an audit arm")
            if self.source_task_id != self.intended_task_id:
                raise WorkspaceValidationError("operational workspaces cannot cross task boundaries")
        elif self.delivery_mode == "causal_audit":
            if self.audit_arm not in {"MOTH", "MSELF", "MCUR", "ZERO", "NOISE"}:
                raise WorkspaceValidationError("causal-audit workspace has an invalid audit arm")
            if self.audit_arm == "MOTH" and self.source_task_id == self.intended_task_id:
                raise WorkspaceValidationError("MOTH must identify a distinct source task")
            if self.audit_arm != "MOTH" and self.source_task_id != self.intended_task_id:
                raise WorkspaceValidationError("only MOTH may use a distinct source task")
        else:
            raise WorkspaceValidationError("unknown workspace delivery mode")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        for field in (
            "allowed_receiver_sessions",
            "source_resource_labels",
            "taint_labels",
            "shape",
        ):
            value[field] = list(value[field])
        return value

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> WorkspaceMetadata:
        unknown = set(value) - _METADATA_FIELDS
        missing = _METADATA_FIELDS - set(value)
        if unknown or missing:
            raise WorkspaceValidationError(
                f"workspace metadata fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        converted = dict(value)
        for field in (
            "allowed_receiver_sessions",
            "source_resource_labels",
            "taint_labels",
            "shape",
        ):
            raw = converted[field]
            if not isinstance(raw, list):
                raise WorkspaceValidationError(f"{field} must be a JSON array")
            converted[field] = tuple(raw)
        metadata = cls(**converted)  # type: ignore[arg-type]
        metadata.validate()
        return metadata


@dataclass(frozen=True, slots=True)
class WorkspaceEnvelope:
    metadata: WorkspaceMetadata
    tensor: torch.Tensor

    @classmethod
    def create(
        cls,
        tensor: torch.Tensor,
        *,
        producer_session_id: str,
        producer_checkpoint_hash: str,
        bridge_checkpoint_hash: str,
        capture_spec_hash: str,
        source_task_id: str,
        intended_task_id: str,
        scope_id: str,
        parent_session_id: str,
        creation_depth: int,
        allowed_receiver_sessions: tuple[str, ...],
        source_resource_labels: tuple[str, ...],
        taint_labels: tuple[str, ...],
        created_at_utc: str,
        expires_at_utc: str,
        delivery_mode: DeliveryMode = "operational",
        audit_arm: AuditArm | None = None,
    ) -> WorkspaceEnvelope:
        checksum = tensor_checksum(tensor)
        metadata = WorkspaceMetadata(
            schema_version=LATENT_WORKSPACE_SCHEMA_VERSION,
            workspace_id="0" * 64,
            producer_session_id=producer_session_id,
            producer_role="coordinator",
            producer_checkpoint_hash=producer_checkpoint_hash,
            bridge_checkpoint_hash=bridge_checkpoint_hash,
            capture_spec_hash=capture_spec_hash,
            source_task_id=source_task_id,
            intended_task_id=intended_task_id,
            scope_id=scope_id,
            parent_session_id=parent_session_id,
            creation_depth=creation_depth,
            allowed_receiver_sessions=tuple(sorted(allowed_receiver_sessions)),
            source_resource_labels=tuple(sorted(set(source_resource_labels))),
            taint_labels=tuple(sorted(set(taint_labels))),
            created_at_utc=created_at_utc,
            expires_at_utc=expires_at_utc,
            shape=(tensor.shape[0], tensor.shape[1]),
            dtype=_DTYPE_TO_NAME.get(tensor.dtype, str(tensor.dtype)),
            tensor_checksum=checksum,
            delivery_mode=delivery_mode,
            audit_arm=audit_arm,
        )
        identity_payload = metadata.to_dict()
        identity_payload.pop("workspace_id")
        workspace_id = hashlib.sha256(_canonical_json(identity_payload)).hexdigest()
        metadata = replace(metadata, workspace_id=workspace_id)
        envelope = cls(metadata=metadata, tensor=tensor)
        envelope.validate_integrity()
        return envelope

    def validate_integrity(self) -> None:
        self.metadata.validate()
        if tuple(self.tensor.shape) != self.metadata.shape:
            raise WorkspaceValidationError("workspace tensor shape does not match metadata")
        actual_dtype = _DTYPE_TO_NAME.get(self.tensor.dtype)
        if actual_dtype != self.metadata.dtype:
            raise WorkspaceValidationError("workspace tensor dtype does not match metadata")
        if tensor_checksum(self.tensor) != self.metadata.tensor_checksum:
            raise WorkspaceValidationError("workspace tensor checksum does not match metadata")
        identity_payload = self.metadata.to_dict()
        identity_payload.pop("workspace_id")
        expected_id = hashlib.sha256(_canonical_json(identity_payload)).hexdigest()
        if expected_id != self.metadata.workspace_id:
            raise WorkspaceValidationError("workspace identity checksum does not match metadata")


@dataclass(frozen=True, slots=True)
class DeliveryContext:
    receiver_session_id: str
    receiver_role: Literal["coordinator"]
    task_id: str
    scope_id: str
    expected_producer_session_id: str
    expected_producer_parent_session_id: str
    expected_producer_checkpoint_hash: str
    expected_bridge_checkpoint_hash: str
    expected_capture_spec_hash: str
    now_utc: str
    delivery_mode: DeliveryMode = "operational"
    expected_audit_arm: AuditArm | None = None


def validate_workspace_delivery(envelope: WorkspaceEnvelope, context: DeliveryContext) -> None:
    envelope.validate_integrity()
    metadata = envelope.metadata
    if context.receiver_role != "coordinator":
        raise WorkspaceValidationError("v1 permits coordinator receivers only")
    if context.receiver_session_id not in metadata.allowed_receiver_sessions:
        raise WorkspaceValidationError("receiver is not authorized for this workspace")
    if context.task_id != metadata.intended_task_id:
        raise WorkspaceValidationError("workspace intended task does not match receiver task")
    if context.scope_id != metadata.scope_id:
        raise WorkspaceValidationError("workspace scope does not match receiver scope")
    if context.expected_producer_session_id != metadata.producer_session_id:
        raise WorkspaceValidationError("workspace producer is not the expected parent session")
    if context.expected_producer_parent_session_id != metadata.parent_session_id:
        raise WorkspaceValidationError("producer ancestry does not match the accepted session graph")
    if context.expected_producer_checkpoint_hash != metadata.producer_checkpoint_hash:
        raise WorkspaceValidationError("producer checkpoint is not approved")
    if context.expected_bridge_checkpoint_hash != metadata.bridge_checkpoint_hash:
        raise WorkspaceValidationError("bridge checkpoint is not approved")
    if context.expected_capture_spec_hash != metadata.capture_spec_hash:
        raise WorkspaceValidationError("capture specification is not approved")
    if context.delivery_mode != metadata.delivery_mode:
        raise WorkspaceValidationError("delivery mode does not match")
    if context.expected_audit_arm != metadata.audit_arm:
        raise WorkspaceValidationError("audit arm does not match")
    now = _parse_utc("now_utc", context.now_utc)
    created_at = _parse_utc("created_at_utc", metadata.created_at_utc)
    expires_at = _parse_utc("expires_at_utc", metadata.expires_at_utc)
    if now < created_at:
        raise WorkspaceValidationError("workspace creation time is in the future")
    if now >= expires_at:
        raise WorkspaceValidationError("workspace has expired")
