"""Compare Prime Agent runtimes with identical model weights and frozen tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from scripts.compare_natural_yield_sdpo_gates_v1 import (
    GATES,
    LOCAL_WORK_DIAGNOSTICS,
    MATERIAL_FORBIDDEN_TRANSITION_REDUCTION,
    TARGET_DIAGNOSTICS,
    ComparisonFailure,
    _task_signature,
)
from scripts.summarize_procedural_harness_master_v1 import _traces


def _normalized_runtime_gate(root: Path, label: str, suffix: str) -> dict[str, Any]:
    run = root / f"{label}-{suffix}" / "train-admission"
    summary_path = run / "SUMMARY.json"
    if not summary_path.is_file():
        raise ComparisonFailure(f"missing gate summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    if summary.get("rescored") is not True or summary.get("episodes") != 8:
        raise ComparisonFailure(f"gate is not an eight-episode rescored run: {run}")
    if summary.get("errors") != 0:
        raise ComparisonFailure(f"gate contains rollout errors: {run}")

    traces = _traces(run)
    if len(traces) != 8:
        raise ComparisonFailure(f"gate contains {len(traces)} traces instead of eight")
    task_signatures = {_task_signature(trace) for trace in traces}
    if len(task_signatures) != 1:
        raise ComparisonFailure(f"gate contains multiple task specifications: {run}")

    config_path = run / "configs" / "eval.json"
    if not config_path.is_file():
        raise ComparisonFailure(f"missing resolved evaluation config: {config_path}")
    config = json.loads(config_path.read_text())
    try:
        harness = config["env"]["agent"]["harness"]
        runtime = config["env"]["agent"]["runtime"]
        version = harness["version"]
        image = runtime["image"]
    except (KeyError, TypeError) as error:
        raise ComparisonFailure(f"gate does not expose a Prime Agent runtime: {run}") from error
    if not isinstance(version, str) or not version:
        raise ComparisonFailure(f"gate has no pinned Prime Agent version: {run}")
    expected_image = f"rlm-prime-agent-runtime:{version}-node22.19.0"
    if image != expected_image:
        raise ComparisonFailure(
            f"gate runtime image does not match its Prime Agent version: {run}"
        )

    for key in ("model", "output_dir"):
        config.pop(key, None)
    client = config.get("client")
    if isinstance(client, dict) and isinstance(client.get("base_url"), str):
        endpoint = urlsplit(client["base_url"])
        if not endpoint.scheme or not endpoint.netloc:
            raise ComparisonFailure(f"evaluation client has an invalid base URL: {run}")
        client["base_url"] = {
            "path": endpoint.path or "/",
            "query": endpoint.query,
            "fragment": endpoint.fragment,
        }
    harness["version"] = "<PRIME_AGENT_VERSION>"
    runtime["image"] = "<PRIME_AGENT_RUNTIME_IMAGE>"
    config_signature = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "task_signature": task_signatures.pop(),
        "normalized_config_signature": config_signature,
        "prime_agent_version": version,
        "runtime_image": image,
        "passed": summary.get("harness", {}).get("passed"),
        "rate": summary.get("harness", {}).get("rate"),
        "diagnostic_means": summary.get("diagnostic_means", {}),
    }


def compare_runtimes(
    root: Path,
    base_label: str,
    candidate_label: str,
    *,
    expected_base_version: str | None = None,
    expected_candidate_version: str | None = None,
) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    versions: dict[str, set[str]] = {"base": set(), "candidate": set()}
    for name, suffix in GATES.items():
        base = _normalized_runtime_gate(root, base_label, suffix)
        candidate = _normalized_runtime_gate(root, candidate_label, suffix)
        if base["task_signature"] != candidate["task_signature"]:
            raise ComparisonFailure(f"{name} task specifications differ")
        if base["normalized_config_signature"] != candidate["normalized_config_signature"]:
            raise ComparisonFailure(
                f"{name} resolved configs differ beyond Prime Agent runtime identity"
            )
        versions["base"].add(base["prime_agent_version"])
        versions["candidate"].add(candidate["prime_agent_version"])
        diagnostic_keys = sorted(
            set(base["diagnostic_means"]) | set(candidate["diagnostic_means"])
        )
        gates[name] = {
            "base": {key: value for key, value in base.items() if key != "task_signature"},
            "candidate": {
                key: value for key, value in candidate.items() if key != "task_signature"
            },
            "delta_passed": candidate["passed"] - base["passed"],
            "diagnostic_deltas": {
                key: candidate["diagnostic_means"].get(key, 0.0)
                - base["diagnostic_means"].get(key, 0.0)
                for key in diagnostic_keys
            },
        }

    if any(len(observed) != 1 for observed in versions.values()):
        raise ComparisonFailure("each runtime arm must use one version across all gates")
    base_version = next(iter(versions["base"]))
    candidate_version = next(iter(versions["candidate"]))
    if base_version == candidate_version:
        raise ComparisonFailure("runtime comparison arms use the same Prime Agent version")
    if expected_base_version is not None and base_version != expected_base_version:
        raise ComparisonFailure(
            f"base runtime is {base_version}, expected {expected_base_version}"
        )
    if expected_candidate_version is not None and candidate_version != expected_candidate_version:
        raise ComparisonFailure(
            f"candidate runtime is {candidate_version}, expected {expected_candidate_version}"
        )

    target = gates["natural_yield"]
    local_work = gates["natural_yield_local_work"]
    for label in ("base", "candidate"):
        missing_target = TARGET_DIAGNOSTICS - set(target[label]["diagnostic_means"])
        if missing_target:
            raise ComparisonFailure(
                f"{label} natural-yield gate is missing diagnostics: "
                + ", ".join(sorted(missing_target))
            )
        missing_local = LOCAL_WORK_DIAGNOSTICS - set(
            local_work[label]["diagnostic_means"]
        )
        if missing_local:
            raise ComparisonFailure(
                f"{label} local-work gate is missing diagnostics: "
                + ", ".join(sorted(missing_local))
            )

    target_hard_improved = target["delta_passed"] > 0
    target_forbidden_reduced = (
        target["diagnostic_deltas"]["forbidden_post_spawn_tool_before_child"]
        <= -MATERIAL_FORBIDDEN_TRANSITION_REDUCTION
    )
    target_connected = target_hard_improved or target_forbidden_reduced
    exact_not_regressed = (
        target["diagnostic_deltas"].get("final_answer_exact", 0.0) >= 0
    )
    prerequisites_retained = all(
        gates[name]["delta_passed"] >= 0 for name in ("atomic_state", "atomic_send")
    )
    local_work_retained = (
        local_work["delta_passed"] >= 0
        and local_work["diagnostic_deltas"].get("local_work_before_yield", 0.0) >= 0
        and local_work["diagnostic_deltas"].get(
            "premature_yield_before_local_work", 0.0
        )
        <= 0
    )
    eligible = (
        target_connected
        and exact_not_regressed
        and prerequisites_retained
        and local_work_retained
    )
    return {
        "schema_version": "prime-agent/runtime-natural-yield-comparison/v1",
        "base": {"label": base_label, "version": base_version},
        "candidate": {"label": candidate_label, "version": candidate_version},
        "decision": {
            "current_runtime_connects_natural_yield": target_connected,
            "target_hard_improved": target_hard_improved,
            "target_forbidden_transition_reduced_materially": target_forbidden_reduced,
            "target_exact_answer_not_regressed": exact_not_regressed,
            "prerequisites_retained": prerequisites_retained,
            "spawn_then_local_work_retained": local_work_retained,
            "eligible_for_current_runtime_replication": eligible,
            "weights_changed": False,
            "note": "A passing screen authorizes fresh runtime replication, not model promotion.",
        },
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("base_label")
    parser.add_argument("candidate_label")
    parser.add_argument("--expected-base-version")
    parser.add_argument("--expected-candidate-version")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = compare_runtimes(
            args.root,
            args.base_label,
            args.candidate_label,
            expected_base_version=args.expected_base_version,
            expected_candidate_version=args.expected_candidate_version,
        )
    except (ComparisonFailure, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"Prime Agent runtime comparison failed: {error}") from error
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
