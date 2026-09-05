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
from copy import deepcopy
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


def normalize_assistant_tool_call_arguments(
    messages: list[dict[str, Any]],
    *,
    expected_action: str,
) -> tuple[list[dict[str, Any]], tuple[dict[str, Any], ...]]:
    """Decode only assistant tool-call argument JSON strings into mappings.

    Qwen's pinned template iterates over ``function.arguments|items``. OpenAI
    wire rows store that value as a JSON string, so rendering requires one
    explicit boundary conversion. The source object is never mutated.
    """

    allowed_actions = {"solve_owned", "delegate_terminal", "delegate_coordinator"}
    if expected_action not in allowed_actions:
        raise PhaseBContractError("row action is outside the exact cognitive-action grammar")
    if not messages or not isinstance(messages[-1], dict) or messages[-1].get("role") != "assistant":
        raise PhaseBContractError("final chat message must be the assistant target")
    for message in messages[:-1]:
        if not isinstance(message, dict):
            raise PhaseBContractError("chat message must be a mapping")
        if message.get("role") == "assistant" and message.get("tool_calls"):
            raise PhaseBContractError("only the single final assistant target may contain tool calls")

    normalized = deepcopy(messages)
    message_index = len(normalized) - 1
    tool_calls = normalized[-1].get("tool_calls")
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        raise PhaseBContractError("final assistant target must contain exactly one tool call")
    tool_call = tool_calls[0]
    if not isinstance(tool_call, dict) or not isinstance(tool_call.get("function"), dict):
        raise PhaseBContractError("assistant tool call function must be a mapping")
    function = tool_call["function"]
    if function.get("name") != "select_cognitive_action":
        raise PhaseBContractError("final assistant target must call select_cognitive_action")
    raw_arguments = function.get("arguments")
    if not isinstance(raw_arguments, str):
        raise PhaseBContractError("final assistant tool-call arguments must be an OpenAI JSON string")
    decoded = _strict_json_object(raw_arguments)
    if set(decoded) != {"action"} or decoded["action"] != expected_action:
        raise PhaseBContractError("decoded tool-call action does not exactly equal the row action")
    function["arguments"] = decoded
    record = {
        "message_index": message_index,
        "call_index": 0,
        "function_name": function["name"],
        "source_kind": "json_string",
        "modified_path": f"messages.{message_index}.tool_calls.0.function.arguments",
        "raw_arguments_sha256": hashlib.sha256(raw_arguments.encode()).hexdigest(),
        "normalized_arguments_sha256": canonical_json_sha256(decoded),
        "normalized_arguments": decoded,
    }
    return normalized, (record,)


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
    receipt_whole_object_sha256: str


@dataclass(frozen=True)
class ValidatedFailedStartEvidence:
    binding: dict[str, Any]
    binding_path: Path
    binding_hash_path: Path
    failure_path: Path
    log_path: Path
    binding_file_sha256: str
    failure_file_sha256: str
    log_file_sha256: str
    failure: dict[str, Any]


@dataclass(frozen=True)
class ValidatedPreflightRejectionEvidence:
    binding: dict[str, Any]
    binding_path: Path
    binding_hash_path: Path
    manifest_path: Path
    log_path: Path
    binding_file_sha256: str
    manifest_file_sha256: str
    log_file_sha256: str
    manifest: dict[str, Any]


def file_sha256(path: Path) -> str:
    """Hash one direct regular file and reject symlink indirection."""

    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise PhaseBContractError(f"expected an absolute, direct regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any, *, omitted_fields: tuple[str, ...] = ()) -> str:
    if not isinstance(value, dict):
        canonical = value
    else:
        canonical = {key: item for key, item in value.items() if key not in omitted_fields}
    return hashlib.sha256(_canonical_json_bytes(canonical)).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _strict_json_object(raw: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-JSON constant {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        decoded = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_object)
    except (json.JSONDecodeError, ValueError) as error:
        raise PhaseBContractError(f"assistant tool-call arguments are malformed JSON: {error}") from error
    if not isinstance(decoded, dict):
        raise PhaseBContractError("assistant tool-call arguments JSON must decode to an object")
    return decoded


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
    receipt_canonical_hash = canonical_json_sha256(receipt, omitted_fields=("receipt_sha256",))
    if receipt.get("receipt_sha256") != receipt_canonical_hash:
        raise PhaseBContractError("A0C receipt internal canonical hash is missing or invalid")
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
        receipt_whole_object_sha256=canonical_json_sha256(receipt),
    )


def validate_failed_start_evidence(binding_path: Path, hash_path: Path) -> ValidatedFailedStartEvidence:
    """Bind the exact failed B start that prospectively motivates B-R."""

    binding_file_hash = validate_hash_sidecar(binding_path, hash_path)
    binding = load_json_file(binding_path)
    if binding.get("schema_version") != "q35-2b-phase-b-failed-start-binding/v1":
        raise PhaseBContractError("unsupported Phase B failed-start binding schema")
    if binding.get("status") != "bound_infrastructure_invalid":
        raise PhaseBContractError("Phase B failed-start evidence is not bound")

    failure_path = Path(binding.get("failure_absolute_path", ""))
    log_path = Path(binding.get("log_absolute_path", ""))
    failure_hash = file_sha256(failure_path)
    log_hash = file_sha256(log_path)
    if failure_hash != binding.get("failure_file_sha256"):
        raise PhaseBContractError("prior Phase B FAILURE hash differs from its binding")
    if log_hash != binding.get("log_file_sha256"):
        raise PhaseBContractError("prior Phase B log hash differs from its binding")
    failure = load_json_file(failure_path)

    predicates = binding.get("failure_predicates")
    if not isinstance(predicates, dict) or not predicates:
        raise PhaseBContractError("prior Phase B failure binding lacks exact predicates")
    for semantic_name, predicate in predicates.items():
        if not isinstance(predicate, dict) or set(predicate) != {"path", "expected"}:
            raise PhaseBContractError(f"failed-start predicate {semantic_name!r} is malformed")
        actual = _resolve_dotted_path(failure, predicate["path"])
        expected = predicate["expected"]
        if type(actual) is not type(expected) or actual != expected:
            raise PhaseBContractError(
                f"failed-start predicate {semantic_name!r} at {predicate['path']!r} was "
                f"{actual!r}, expected {expected!r}"
            )

    try:
        log_text = log_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise PhaseBContractError(f"prior Phase B log is not valid UTF-8: {error}") from error
    markers = binding.get("required_log_markers")
    if not isinstance(markers, list) or not markers or any(not isinstance(marker, str) for marker in markers):
        raise PhaseBContractError("prior Phase B log markers are malformed")
    if any(marker not in log_text for marker in markers):
        raise PhaseBContractError("prior Phase B log lacks a bound failure marker")

    return ValidatedFailedStartEvidence(
        binding=binding,
        binding_path=binding_path,
        binding_hash_path=hash_path,
        failure_path=failure_path,
        log_path=log_path,
        binding_file_sha256=binding_file_hash,
        failure_file_sha256=failure_hash,
        log_file_sha256=log_hash,
        failure=failure,
    )


def validate_preflight_rejection_evidence(
    binding_path: Path, hash_path: Path
) -> ValidatedPreflightRejectionEvidence:
    """Bind the exact no-model B-R preflight rejection motivating B-R2."""

    binding_file_hash = validate_hash_sidecar(binding_path, hash_path)
    binding = load_json_file(binding_path)
    if binding.get("schema_version") != "q35-2b-phase-b-preflight-rejection-binding/v1":
        raise PhaseBContractError("unsupported Phase B preflight rejection binding schema")
    if binding.get("status") != "bound_pre_model_rejection":
        raise PhaseBContractError("Phase B preflight rejection evidence is not bound")

    manifest_path = Path(binding.get("manifest_absolute_path", ""))
    log_path = Path(binding.get("log_absolute_path", ""))
    manifest_hash = file_sha256(manifest_path)
    log_hash = file_sha256(log_path)
    if manifest_hash != binding.get("manifest_file_sha256"):
        raise PhaseBContractError("prior Phase B preflight manifest hash differs from its binding")
    if log_hash != binding.get("log_file_sha256"):
        raise PhaseBContractError("prior Phase B preflight log hash differs from its binding")
    manifest = load_json_file(manifest_path)

    predicates = binding.get("manifest_predicates")
    if not isinstance(predicates, dict) or not predicates:
        raise PhaseBContractError("prior Phase B preflight binding lacks exact predicates")
    for semantic_name, predicate in predicates.items():
        if not isinstance(predicate, dict) or set(predicate) != {"path", "expected"}:
            raise PhaseBContractError(f"preflight-rejection predicate {semantic_name!r} is malformed")
        actual = _resolve_dotted_path(manifest, predicate["path"])
        expected = predicate["expected"]
        if type(actual) is not type(expected) or actual != expected:
            raise PhaseBContractError(
                f"preflight-rejection predicate {semantic_name!r} at {predicate['path']!r} was "
                f"{actual!r}, expected {expected!r}"
            )

    try:
        log_text = log_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise PhaseBContractError(f"prior Phase B preflight log is not valid UTF-8: {error}") from error
    markers = binding.get("required_log_markers")
    if not isinstance(markers, list) or not markers or any(not isinstance(marker, str) for marker in markers):
        raise PhaseBContractError("prior Phase B preflight log markers are malformed")
    if any(marker not in log_text for marker in markers):
        raise PhaseBContractError("prior Phase B preflight log lacks a bound marker")

    return ValidatedPreflightRejectionEvidence(
        binding=binding,
        binding_path=binding_path,
        binding_hash_path=hash_path,
        manifest_path=manifest_path,
        log_path=log_path,
        binding_file_sha256=binding_file_hash,
        manifest_file_sha256=manifest_hash,
        log_file_sha256=log_hash,
        manifest=manifest,
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
