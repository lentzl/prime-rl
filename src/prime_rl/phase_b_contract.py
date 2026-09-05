"""Pure-stdlib launch contracts for the Phase B fixed-depth smoke.

This module deliberately does not import Torch or Transformers.  The host runner
uses it before importing either dependency, so an unresolved carrier receipt or
an unauthorized plan cannot allocate a model or touch CUDA.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_A0C_PREDICATES = (
    "receipt_status_exact",
    "receipt_claim_exact",
    "receipt_plan_sha256_exact",
    "receipt_execution_commit_exact",
    "receipt_dependency_scope_exact",
    "four_probes_completed",
    "probe_status_complete_all",
    "hard_bypass_bitwise_all",
    "hard_bypass_contract_exact_all",
    "capture_detached_all",
    "capture_finite_all",
    "capture_deterministic_all",
    "capture_visible_indices_valid_all",
    "capture_content_exact_all",
    "insertion_eight_slots_all",
    "insertion_original_content_preserved_all",
    "insertion_attention_one_all",
    "insertion_positions_sequential_all",
    "insertion_labels_masked_all",
    "insertion_no_other_masking_all",
    "soft_loss_finite_all",
    "soft_logits_finite_all",
    "workspace_gradient_finite_nonzero_all",
    "gate_gradient_finite_nonzero_all",
    "e33_gradients_absent_all",
    "e33_file_hash_before_exact",
    "e33_file_hash_after_exact",
    "e33_metadata_before_exact",
    "e33_metadata_after_exact",
)


class PhaseBContractError(RuntimeError):
    """The prospective Phase B execution contract is not satisfied."""


@dataclass(frozen=True)
class ValidatedA0CBinding:
    binding: dict[str, Any]
    receipt: dict[str, Any]
    binding_path: Path
    binding_hash_path: Path
    receipt_path: Path
    binding_file_sha256: str
    receipt_file_sha256: str
    receipt_canonical_sha256: str


def file_sha256(path: Path) -> str:
    """Hash one direct regular file and reject symlink indirection."""

    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise PhaseBContractError(f"expected an absolute, direct regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_json_file(path: Path) -> dict[str, Any]:
    file_sha256(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PhaseBContractError(f"invalid JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise PhaseBContractError(f"JSON root must be an object: {path}")
    return value


def validate_hash_sidecar(file_path: Path, sidecar_path: Path) -> str:
    """Validate a one-line ``<sha256>  <basename>`` immutable sidecar."""

    actual = file_sha256(file_path)
    file_sha256(sidecar_path)
    fields = sidecar_path.read_text(encoding="ascii").strip().split()
    if len(fields) != 2 or fields[1] != file_path.name:
        raise PhaseBContractError(f"malformed hash sidecar for {file_path.name}")
    if fields[0] != actual:
        raise PhaseBContractError(f"hash sidecar mismatch for {file_path.name}")
    return actual


def validate_plan_authorization(plan: dict[str, Any]) -> None:
    """Accept only a future prospectively frozen A0C-bound plan.

    The checked-in v1 plan intentionally fails this check.  It binds the broader
    A0 receipt that never existed and remains marked not authorized.  Root must
    prospectively freeze a replacement after the A0C schema and receipt hash are
    known; the runner cannot reinterpret the old plan after the fact.
    """

    if plan.get("status") != "frozen_bound_a0c_authorized":
        raise PhaseBContractError("Phase B plan is not prospectively A0C-bound and authorized")
    dependency = plan.get("a0c_dependency")
    if not isinstance(dependency, dict) or dependency.get("required_before_launch") is not True:
        raise PhaseBContractError("Phase B plan lacks its required A0C dependency")


def validate_a0c_binding(binding_path: Path, hash_path: Path) -> ValidatedA0CBinding:
    """Validate an exact, prospective A0C receipt binding.

    Receipt details are expressed by ``predicate_paths`` in the prospectively
    frozen binding.  This keeps the interface explicit while refusing to invent
    A0C's still-unfrozen receipt schema here.
    """

    binding_file_hash = validate_hash_sidecar(binding_path, hash_path)
    binding = load_json_file(binding_path)
    if binding.get("schema_version") != "q35-2b-phase-b-a0c-binding/v1":
        raise PhaseBContractError("unsupported Phase B A0C binding schema")
    if binding.get("status") != "bound":
        raise PhaseBContractError("A0C binding is unresolved")
    if binding.get("required_claim") != "four_probe_carrier_only_for_phase_b":
        raise PhaseBContractError("A0C binding claim is not the carrier-only Phase B claim")

    receipt_text = binding.get("receipt_absolute_path")
    if not isinstance(receipt_text, str):
        raise PhaseBContractError("A0C binding lacks an absolute receipt path")
    receipt_path = Path(receipt_text)
    receipt_file_hash = file_sha256(receipt_path)
    if receipt_file_hash != binding.get("receipt_file_sha256"):
        raise PhaseBContractError("A0C receipt file hash differs from the binding")
    receipt = load_json_file(receipt_path)
    receipt_canonical_hash = canonical_json_sha256(receipt)
    if receipt_canonical_hash != binding.get("receipt_canonical_sha256"):
        raise PhaseBContractError("A0C receipt canonical hash differs from the binding")
    if receipt.get("schema_version") != binding.get("receipt_schema_version"):
        raise PhaseBContractError("A0C receipt schema differs from the binding")

    predicates = binding.get("predicate_paths")
    if not isinstance(predicates, dict) or set(predicates) != set(REQUIRED_A0C_PREDICATES):
        raise PhaseBContractError("A0C binding does not map the exact required carrier predicates")
    for semantic_name, predicate in predicates.items():
        if not isinstance(predicate, dict) or set(predicate) != {"path", "expected"}:
            raise PhaseBContractError(f"A0C predicate mapping {semantic_name!r} is malformed")
        dotted_path = predicate["path"]
        expected = predicate["expected"]
        if not isinstance(dotted_path, str) or not dotted_path:
            raise PhaseBContractError("A0C receipt predicate path must be a nonempty string")
        actual = _resolve_dotted_path(receipt, dotted_path)
        if type(actual) is not type(expected) or actual != expected:
            raise PhaseBContractError(
                f"A0C receipt predicate {semantic_name!r} at {dotted_path!r} was {actual!r}, expected {expected!r}"
            )

    return ValidatedA0CBinding(
        binding=binding,
        receipt=receipt,
        binding_path=binding_path,
        binding_hash_path=hash_path,
        receipt_path=receipt_path,
        binding_file_sha256=binding_file_hash,
        receipt_file_sha256=receipt_file_hash,
        receipt_canonical_sha256=receipt_canonical_hash,
    )


def atomic_exclusive_json(
    output_dir: Path,
    name: str,
    value: dict[str, Any],
    *,
    maximum_directory_bytes: int,
) -> None:
    """Publish the sole success or failure receipt without replacement."""

    if name not in {"SUCCESS.json", "FAILURE.json"}:
        raise ValueError("Phase B receipt must be SUCCESS.json or FAILURE.json")
    if not output_dir.is_absolute() or output_dir.is_symlink() or not output_dir.is_dir():
        raise PhaseBContractError("Phase B output must be an absolute, direct directory")
    if maximum_directory_bytes <= 0:
        raise ValueError("maximum_directory_bytes must be positive")

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY | nofollow)
    try:
        existing_receipts = {entry for entry in os.listdir(directory_fd) if entry in {"SUCCESS.json", "FAILURE.json"}}
        if existing_receipts:
            raise FileExistsError(f"Phase B already has a terminal receipt: {sorted(existing_receipts)}")
        encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        current_bytes = _directory_bytes(output_dir)
        if current_bytes + len(encoded) > maximum_directory_bytes:
            raise PhaseBContractError("Phase B artifact cap would be exceeded by the terminal receipt")
        temporary = f".{name}.{os.getpid()}.tmp"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            try:
                view = memoryview(encoded)
                while view:
                    written = os.write(descriptor, view)
                    if not written:
                        raise OSError("short write while publishing Phase B receipt")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            # link() is an atomic no-replace publish. rename() would replace an
            # independently published terminal receipt on POSIX.
            os.link(
                temporary,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        finally:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.fsync(directory_fd)
        terminal = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        receipts = {entry for entry in os.listdir(directory_fd) if entry in {"SUCCESS.json", "FAILURE.json"}}
        if receipts != {name} or not stat.S_ISREG(terminal.st_mode):
            raise PhaseBContractError("Phase B terminal receipt postflight failed")
    finally:
        os.close(directory_fd)


def _resolve_dotted_path(value: dict[str, Any], dotted_path: str) -> Any:
    def resolve(current: Any, components: list[str]) -> Any:
        if not components:
            return current
        component, *remaining = components
        if component == "*":
            if not isinstance(current, list):
                raise PhaseBContractError(f"A0C receipt wildcard is not a list in {dotted_path!r}")
            return [resolve(item, remaining) for item in current]
        if not isinstance(current, dict) or component not in current:
            raise PhaseBContractError(f"A0C receipt lacks predicate path {dotted_path!r}")
        return resolve(current[component], remaining)

    return resolve(value, dotted_path.split("."))


def _directory_bytes(path: Path) -> int:
    total = 0
    for root, directories, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        if any((root_path / directory).is_symlink() for directory in directories):
            raise PhaseBContractError("Phase B output contains a symlinked directory")
        for filename in files:
            item = root_path / filename
            if item.is_symlink() or not item.is_file():
                raise PhaseBContractError("Phase B output contains a non-regular artifact")
            total += item.stat().st_size
    return total
