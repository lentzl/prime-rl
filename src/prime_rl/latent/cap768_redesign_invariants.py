from __future__ import annotations

import ast
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path


class InvariantViolation(ValueError):
    pass


@dataclass(frozen=True)
class StaticGuardEvidence:
    runner_sha256: str
    forbidden_calls: tuple[str, ...]
    forbidden_identifiers: tuple[str, ...]
    forbidden_imports: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        return {key: list(item) if isinstance(item, tuple) else item for key, item in value.items()}


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def inspect_no_training_runner(
    path: Path,
    *,
    forbidden_call_names: frozenset[str] = frozenset({"backward", "generate", "step"}),
    forbidden_identifier_names: frozenset[str] = frozenset({"AdamW", "WorkspaceBridge"}),
) -> StaticGuardEvidence:
    if path.is_symlink() or not path.is_file():
        raise InvariantViolation("runner is absent or symlinked")
    source_bytes = path.read_bytes()
    try:
        tree = ast.parse(source_bytes, filename=str(path))
    except (SyntaxError, ValueError) as error:
        raise InvariantViolation(f"runner is not valid Python: {error}") from error

    calls = sorted(
        {
            name
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for name in [_call_name(node)]
            if name in forbidden_call_names
        }
    )
    identifiers = sorted(
        {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id in forbidden_identifier_names}
        | {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr in forbidden_identifier_names
        }
    )
    imports = sorted(
        {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
            if alias.name.rsplit(".", 1)[-1] in forbidden_identifier_names
        }
    )
    evidence = StaticGuardEvidence(
        runner_sha256=hashlib.sha256(source_bytes).hexdigest(),
        forbidden_calls=tuple(calls),
        forbidden_identifiers=tuple(identifiers),
        forbidden_imports=tuple(imports),
    )
    if calls or identifiers or imports:
        raise InvariantViolation(f"runner contains forbidden training/generation syntax: {evidence.as_dict()}")
    return evidence


def require_pre_model_static_guard(
    path: Path,
    *,
    run_function: str,
    guard_function: str,
    model_loader_attribute: str = "from_pretrained",
) -> None:
    if path.is_symlink() or not path.is_file():
        raise InvariantViolation("runner is absent or symlinked")
    tree = ast.parse(path.read_bytes(), filename=str(path))
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == run_function
        ),
        None,
    )
    if function is None:
        raise InvariantViolation("run function is absent")
    guard_lines = [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == guard_function
    ]
    model_lines = [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == model_loader_attribute
    ]
    if len(guard_lines) != 1 or not model_lines or guard_lines[0] >= min(model_lines):
        raise InvariantViolation("static guard is not invoked exactly once before model loading")


def validate_comparison_partition(
    comparisons: list[dict[str, object]],
    *,
    gated_names: tuple[str, ...],
    descriptive_names: tuple[str, ...],
) -> None:
    if not gated_names or not descriptive_names or set(gated_names) & set(descriptive_names):
        raise InvariantViolation("comparison partition is empty or overlapping")
    expected = list(gated_names) + list(descriptive_names)
    if [row.get("name") for row in comparisons] != expected:
        raise InvariantViolation("comparison order or partition changed")
    for index, row in enumerate(comparisons):
        expected_role = "gate" if index < len(gated_names) else "descriptive_only"
        if set(row) != {"name", "role", "lhs_shape", "rhs_shape", "torch_equal"}:
            raise InvariantViolation("comparison schema changed")
        if row.get("role") != expected_role:
            raise InvariantViolation("comparison role changed")
        left_shape = row.get("lhs_shape")
        right_shape = row.get("rhs_shape")
        if not isinstance(left_shape, list) or not isinstance(right_shape, list):
            raise InvariantViolation("comparison shape evidence changed")
        if expected_role == "gate" and left_shape != right_shape:
            raise InvariantViolation("gated comparison is not same-shape")
        if not isinstance(row.get("torch_equal"), bool):
            raise InvariantViolation("comparison equality evidence changed")


def validate_exact_failure_binding(
    *,
    failure_path: Path,
    log_path: Path,
    expected_failure_sha256: str,
    expected_log_sha256: str,
) -> None:
    for path, expected in (
        (failure_path, expected_failure_sha256),
        (log_path, expected_log_sha256),
    ):
        if path.is_symlink() or not path.is_file():
            raise InvariantViolation("bound failure evidence is absent or symlinked")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise InvariantViolation("bound failure evidence hash changed")
