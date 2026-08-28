#!/usr/bin/env python3
"""Resumable autonomous executor for the dual-policy dense SPADE loop."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import combine_q35_2b_role_replay_sft_v1 as replay_builder
import q35_2b_spade_dual_dense_loop_v1 as controller
from build_q35_2b_environment_bootstrap_context_v1 import LEAK_LADDER

RUNNER_SCHEMA_VERSION = "qwen35-2b-spade-dual-dense-runner/v1"
JOURNAL_SCHEMA_VERSION = "qwen35-2b-spade-dual-dense-journal-event/v1"
TASK_AXIS = "natural_n1a"
SAMPLING_SEED = "20260823"
MODEL_WEIGHT = "model.safetensors"
PHASE_TAGS = {
    "e0_full_actions": "e0",
    "e0c_natural_child": "e0c",
    "e0c2_natural_child_no_template": "e0c2",
    "e0c25_inline_evidence": "e0c25",
    "e0c275_inline_location": "e0c275",
    "e0c28_inline_only": "e0c28",
    "e0c29_evidence_available": "e0c29",
    "e0c3_natural_child_minimal": "e0c3",
    "e0d2_capped_yield_exact_child": "e0d2x",
    "e0d3_uncapped_yield_exact_child": "e0d3x",
    "e0d3_uncapped_yield": "e0d3",
}


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_text_once_or_match(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != value:
            raise ValueError(f"refusing to replace a different durable artifact: {path}")
        return
    path.write_text(value, encoding="utf-8")


def _write_json_once_or_match(path: Path, value: dict[str, Any]) -> None:
    _write_text_once_or_match(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _append_journal(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        existing = [json.loads(line) for line in handle if line.strip()]
        event = {
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "sequence": len(existing),
            "previous_event_sha256": existing[-1]["event_sha256"] if existing else None,
            "recorded_at_utc": _now(),
            **payload,
        }
        event["event_sha256"] = hashlib.sha256(_canonical_json(event).encode()).hexdigest()
        handle.seek(0, 2)
        handle.write(_canonical_json(event) + "\n")
        handle.flush()


@dataclass(frozen=True)
class RunnerConfig:
    repo_root: Path
    events: Path
    artifacts_root: Path
    results_root: Path
    output_root: Path
    experiment_dir: Path
    journal: Path
    stop_file: Path
    uv_bin: str
    learning_rate: float
    max_evaluations: int
    max_update_pairs: int
    max_actions: int
    open_ended: bool
    dry_run: bool
    coevolution: bool = False


class AutonomousDualDenseRunner:
    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        self.evaluations = 0
        self.update_pairs = 0
        self.actions = 0

    def status(self) -> dict[str, Any]:
        return controller.project(controller._load_events(self.config.events))

    def _run(
        self,
        command: list[str | Path],
        *,
        env: dict[str, str] | None = None,
        stdout_path: Path | None = None,
    ) -> None:
        resolved = [str(item) for item in command]
        _append_journal(self.config.journal, {"kind": "command_started", "command": resolved})
        try:
            if stdout_path is None:
                subprocess.run(resolved, cwd=self.config.repo_root, env=env, check=True)
            else:
                stdout_path.parent.mkdir(parents=True, exist_ok=True)
                with stdout_path.open("a", encoding="utf-8") as handle:
                    subprocess.run(
                        resolved,
                        cwd=self.config.repo_root,
                        env=env,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
        except (OSError, subprocess.CalledProcessError) as error:
            _append_journal(
                self.config.journal,
                {
                    "kind": "command_failed",
                    "command": resolved,
                    "error": str(error),
                    "returncode": getattr(error, "returncode", None),
                },
            )
            raise
        _append_journal(self.config.journal, {"kind": "command_completed", "command": resolved})

    @staticmethod
    def _external_model(candidates: dict[str, dict[str, Any]]) -> str:
        coordinator = candidates["coordinator"]["model_sha256"][:12]
        child = candidates["child"]["model_sha256"][:12]
        return f"q35-dual-c{coordinator}-k{child}"

    @staticmethod
    def _verify_candidate(candidate: dict[str, Any]) -> Path:
        path = Path(candidate["model_path"])
        weight = path / MODEL_WEIGHT
        if not path.is_absolute() or not (path / "STABLE").is_file() or not weight.is_file():
            raise ValueError(f"dense candidate is incomplete: {path}")
        if _sha256_file(weight) != candidate["model_sha256"]:
            raise ValueError(f"dense candidate SHA-256 mismatch: {path}")
        return path

    @staticmethod
    def _gpus_idle() -> None:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            raise RuntimeError("refusing to start an action while a GPU process is active")

    @staticmethod
    def _pair_label(candidates: dict[str, dict[str, Any]]) -> str:
        return f"{candidates['coordinator']['label'].lower()}-{candidates['child']['label'].lower()}"

    def _bank_base_label(self, candidates: dict[str, dict[str, Any]], arm: dict[str, Any]) -> str:
        return (
            f"{self._pair_label(candidates)}-{arm['track']}-{PHASE_TAGS[arm['phase']]}-"
            f"{arm['start_index']}-n{arm['tasks']}"
        )

    @staticmethod
    def _designer_leak_level(candidates: dict[str, dict[str, Any]], arm: dict[str, Any]) -> str:
        if arm["track"] != "yield":
            return LEAK_LADDER[0]
        metadata = (candidates["coordinator"].get("replay") or {}).get("environment_designer")
        if not isinstance(metadata, dict):
            return LEAK_LADDER[0]
        mode = metadata.get("mode")
        coevolution_stages = {
            "paired_hint_regret": "delayed_reward_filtered_coevolution",
            "scaffolded_repair": "scaffolded_schema_and_safety_repair",
        }
        if mode in coevolution_stages:
            if (
                metadata.get("training_stage") != coevolution_stages[mode]
                or not isinstance(metadata.get("trained_batch_ids"), list)
                or not all(isinstance(item, str) and item for item in metadata["trained_batch_ids"])
                or not isinstance(metadata.get("selected_environment_ids"), list)
                or not all(
                    isinstance(item, str) and item for item in metadata["selected_environment_ids"]
                )
            ):
                raise ValueError("coordinator candidate has invalid coevolution Designer metadata")
            return LEAK_LADDER[0]
        if mode not in {None, "static_leak_ladder"}:
            raise ValueError("coordinator candidate has an unknown environment-designer mode")
        if metadata.get("ladder") != list(LEAK_LADDER) or metadata.get("promotion_step_size") != 1:
            raise ValueError("coordinator candidate has an invalid environment-designer ladder")
        key = "trained_stage_index" if arm.get("reason") == "verify_exact_parent_retention_rung" else "next_stage_index"
        stage = metadata.get(key)
        if not isinstance(stage, int) or not 0 <= stage < len(LEAK_LADDER):
            raise ValueError("coordinator candidate has an invalid environment-designer stage")
        return LEAK_LADDER[stage]

    def _bootstrap_path(self, candidates: dict[str, dict[str, Any]], arm: dict[str, Any]) -> Path:
        leak_level = self._designer_leak_level(candidates, arm)
        return self.config.artifacts_root / (
            f"dual-dense-{self._bank_base_label(candidates, arm)}-{leak_level}-bootstrap.json"
        )

    def _ensure_bootstrap(self, candidates: dict[str, dict[str, Any]], arm: dict[str, Any]) -> Path:
        path = self._bootstrap_path(candidates, arm)
        leak_level = self._designer_leak_level(candidates, arm)
        if path.exists():
            payload = _json(path)
            records = payload.get("records") or []
            if (
                payload.get("axes") != [{"name": TASK_AXIS, "start_index": arm["start_index"]}]
                or payload.get("tasks_per_axis") != arm["tasks"]
                or payload.get("gradient_updates") != 0
                or payload.get("leak_level") != leak_level
                or payload.get("leak_stage_index") != LEAK_LADDER.index(leak_level)
                or payload.get("leak_ladder") != list(LEAK_LADDER)
                or len(records) != arm["tasks"]
                or any(record.get("final_answer_in_context") is not False for record in records)
            ):
                raise ValueError(f"existing bootstrap does not match planned arm: {path}")
            return path
        self._run(
            [
                self.config.uv_bin,
                "run",
                "--no-sync",
                self.config.repo_root / "scripts/build_q35_2b_environment_bootstrap_context_v1.py",
                "--output",
                path,
                "--axis",
                f"{TASK_AXIS}:{arm['start_index']}",
                "--tasks-per-axis",
                str(arm["tasks"]),
                "--leak-level",
                leak_level,
            ]
        )
        return self._ensure_bootstrap(candidates, arm)

    @staticmethod
    def _trace_count(path: Path) -> int:
        if not path.is_file():
            return 0
        with path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    @staticmethod
    def _snapshot_abort_evidence(source: Path, destination: Path) -> Path:
        if not source.is_file():
            raise FileNotFoundError(f"aborted evaluation lacks evidence: {source}")
        _write_text_once_or_match(destination, source.read_text(encoding="utf-8"))
        return destination

    def _evaluation_attempt(self, base: str, tasks: int) -> tuple[str, Path, bool]:
        for attempt in range(1, 100):
            label = base if attempt == 1 else f"{base}-attempt{attempt}"
            run = self.config.results_root / label
            summary = run / "INTERACTION_SUMMARY.json"
            traces = run / TASK_AXIS / "traces.jsonl"
            if summary.is_file():
                return label, run, True
            if not run.exists() or self._trace_count(traces) == tasks:
                return label, run, False
        raise RuntimeError("exhausted dual-policy evaluation attempt labels")

    def collect(self, status: dict[str, Any], arm: dict[str, Any]) -> None:
        candidates = status["candidates"]
        paths = {role: self._verify_candidate(value) for role, value in candidates.items()}
        bootstrap = self._ensure_bootstrap(candidates, arm)
        label, run, summarized = self._evaluation_attempt(self._bank_base_label(candidates, arm), arm["tasks"])
        traces = run / TASK_AXIS / "traces.jsonl"
        summary = run / "INTERACTION_SUMMARY.json"
        versions = run / "VERSIONS.txt"
        routing = run / "ROUTING_AUDIT.jsonl"
        _append_journal(
            self.config.journal,
            {
                "kind": "evaluation_started",
                "event_head_sha256": status["event_head_sha256"],
                "candidates": candidates,
                "arm": arm,
                "bootstrap_path": str(bootstrap),
                "bootstrap_sha256": _sha256_file(bootstrap),
            },
        )
        if not run.exists():
            self._gpus_idle()
            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{Path(self.config.uv_bin).parent}:{self.config.repo_root / '.venv/bin'}:{environment.get('PATH', '')}",
                    "NCCL_P2P_DISABLE": "1",
                    "QWEN38_QUALIFICATION_OUTPUT_ROOT": str(self.config.results_root),
                    "QWEN38_QUALIFICATION_AXES": TASK_AXIS,
                    "QWEN38_QUALIFICATION_NUM_TASKS": str(arm["tasks"]),
                    "QWEN38_QUALIFICATION_NUM_ROLLOUTS": "1",
                    "QWEN38_QUALIFICATION_MAX_CONCURRENT": str(arm["tasks"]),
                    "QWEN38_QUALIFICATION_EVAL_MAX_ADDRESS_SPACE_BYTES": str(32 * 1024**3),
                    "QWEN38_QUALIFICATION_START_INDEX": str(arm["start_index"]),
                    "QUALIFICATION_REASONING_EFFORT": "high",
                    "QUALIFICATION_SAMPLING_SEED": SAMPLING_SEED,
                    "QUALIFICATION_SAMPLING_TEMPERATURE": "0.6",
                    "QUALIFICATION_PRIVILEGED_BOOTSTRAP_PATH": str(bootstrap),
                    "PROCEDURAL_INTERACTION_CURRICULUM": arm["phase"],
                    "DUAL_EXTERNAL_MODEL": self._external_model(candidates),
                }
            )
            try:
                self._run(
                    [
                        self.config.repo_root / "scripts/run_q35_2b_dual_policy_mastery_v1.sh",
                        paths["coordinator"],
                        paths["child"],
                        label,
                        status["model_revision"],
                    ],
                    env=environment,
                )
            except subprocess.CalledProcessError:
                partial_traces = self._snapshot_abort_evidence(traces, run / "PARTIAL_TRACES.jsonl")
                partial_routing = self._snapshot_abort_evidence(routing, run / "PARTIAL_ROUTING_AUDIT.jsonl")
                namespace = argparse.Namespace(
                    events=self.config.events,
                    track=arm["track"],
                    phase=arm["phase"],
                    start_index=arm["start_index"],
                    bank_id=label,
                    versions=versions,
                    traces=partial_traces,
                    bootstrap=bootstrap,
                    routing_audit=partial_routing,
                    reason="evaluation_command_failed_after_gate_mathematically_closed",
                    recorded_at=None,
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    controller._abort_evaluation(namespace)
                _append_journal(
                    self.config.journal,
                    {
                        "kind": "evaluation_aborted",
                        "bank_id": label,
                        "event_head_sha256": self.status()["event_head_sha256"],
                    },
                )
                return
        if not summarized:
            if self._trace_count(traces) != arm["tasks"]:
                raise ValueError("dual-policy evaluation did not produce the planned trace count")
            self._run(
                [
                    self.config.uv_bin,
                    "run",
                    "--no-sync",
                    self.config.repo_root / "scripts/summarize_q35_2b_interaction_curriculum_v1.py",
                    traces,
                    "--phase",
                    arm["phase"],
                    "--output",
                    summary,
                ]
            )
        for required in (traces, summary, versions, routing):
            if not required.is_file():
                raise FileNotFoundError(f"completed dual evaluation lacks artifact: {required}")
        namespace = argparse.Namespace(
            events=self.config.events,
            track=arm["track"],
            phase=arm["phase"],
            start_index=arm["start_index"],
            bank_id=label,
            summary=summary,
            versions=versions,
            traces=traces,
            bootstrap=bootstrap,
            routing_audit=routing,
            recorded_at=None,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            controller._record_evaluation(namespace)
        _append_journal(
            self.config.journal,
            {
                "kind": "evaluation_recorded",
                "bank_id": label,
                "summary_sha256": _sha256_file(summary),
                "event_head_sha256": self.status()["event_head_sha256"],
            },
        )

    def _source_event(self, bank_id: str) -> dict[str, Any]:
        for event in reversed(controller._load_events(self.config.events)):
            if event.get("kind") == "evaluation_recorded" and event["bank"]["id"] == bank_id:
                return event
        raise ValueError(f"admitted source bank is absent from event log: {bank_id}")

    def _export_role(
        self,
        *,
        status: dict[str, Any],
        source: dict[str, Any],
        role: str,
        output_dir: Path,
        selection_count: int,
    ) -> None:
        candidate = status["candidates"][role]
        if output_dir.exists():
            manifest = _json(output_dir / "MANIFEST.json")
            if (
                manifest.get("selected_roles") != [role]
                or manifest.get("rows_by_role") != {role: selection_count}
                or manifest.get("student", {}).get("weight_sha256") != candidate["model_sha256"]
                or manifest.get("source", {}).get("summary_sha256") != source["artifacts"]["summary_sha256"]
            ):
                raise ValueError(f"existing role corpus does not match authorization: {output_dir}")
            return
        self._run(
            [
                self.config.uv_bin,
                "run",
                "--no-sync",
                self.config.repo_root / "scripts/export_q35_2b_interaction_sft_v1.py",
                "--traces",
                source["artifacts"]["traces_path"],
                "--summary",
                source["artifacts"]["summary_path"],
                "--versions",
                source["artifacts"]["versions_path"],
                "--output-dir",
                output_dir,
                "--phase",
                source["phase"],
                "--sampled-model",
                self._external_model(status["candidates"]),
                "--student-snapshot",
                candidate["model_path"],
                "--student-revision",
                status["model_revision"],
                "--student-weight-sha",
                candidate["model_sha256"],
                "--roles",
                role,
                "--selection-count",
                str(selection_count),
                "--dense-weight-mutated",
            ],
            stdout_path=output_dir.parent / f"{output_dir.name}-export.log",
        )

    def _export_positive_prefixes(
        self,
        *,
        status: dict[str, Any],
        source: dict[str, Any],
        role: str,
        output_dir: Path,
    ) -> Path | None:
        candidate = status["candidates"][role]
        if not output_dir.exists():
            self._run(
                [
                    self.config.uv_bin,
                    "run",
                    "--no-sync",
                    self.config.repo_root / "scripts/export_q35_2b_positive_prefix_sft_v1.py",
                    "--traces",
                    source["artifacts"]["traces_path"],
                    "--summary",
                    source["artifacts"]["summary_path"],
                    "--output-dir",
                    output_dir,
                    "--phase",
                    source["phase"],
                    "--sampled-model",
                    self._external_model(status["candidates"]),
                    "--student-snapshot",
                    candidate["model_path"],
                    "--student-revision",
                    status["model_revision"],
                    "--student-weight-sha",
                    candidate["model_sha256"],
                    "--role",
                    role,
                    "--max-rows",
                    "4",
                ],
                stdout_path=output_dir.parent / f"{output_dir.name}-export.log",
            )
        manifest = output_dir / "MANIFEST.json"
        if not manifest.is_file():
            if not (output_dir / "SKIPPED.json").is_file():
                raise ValueError(f"positive-prefix export is incomplete: {output_dir}")
            return None
        value = _json(manifest)
        if (
            value.get("schema_version") != replay_builder.POSITIVE_PREFIX_SOURCE_SCHEMA_VERSION
            or value.get("role") != role
            or value.get("student", {}).get("weight_sha256") != candidate["model_sha256"]
            or value.get("source", {}).get("summary_sha256") != source["artifacts"]["summary_sha256"]
        ):
            raise ValueError(f"positive-prefix corpus does not match the source: {output_dir}")
        return output_dir

    def _export_designer(
        self,
        *,
        status: dict[str, Any],
        source: dict[str, Any],
        output_dir: Path,
        selection_count: int,
    ) -> None:
        candidate = status["candidates"]["coordinator"]
        if output_dir.exists():
            manifest = _json(output_dir / "MANIFEST.json")
            if (
                manifest.get("role") != "coordinator"
                or manifest.get("objective") != "environment_designer"
                or manifest.get("rows") != selection_count
                or manifest.get("exact_answer_rows") != 0
                or manifest.get("student", {}).get("weight_sha256") != candidate["model_sha256"]
                or manifest.get("source", {}).get("summary_sha256") != source["artifacts"]["summary_sha256"]
                or manifest.get("source", {}).get("bootstrap_sha256") != source["artifacts"]["bootstrap_sha256"]
            ):
                raise ValueError(f"existing designer corpus does not match authorization: {output_dir}")
            return
        self._run(
            [
                self.config.uv_bin,
                "run",
                "--no-sync",
                self.config.repo_root / "scripts/export_q35_2b_environment_designer_sft_v1.py",
                "--bootstrap",
                source["artifacts"]["bootstrap_path"],
                "--summary",
                source["artifacts"]["summary_path"],
                "--output-dir",
                output_dir,
                "--phase",
                source["phase"],
                "--student-snapshot",
                candidate["model_path"],
                "--student-revision",
                status["model_revision"],
                "--student-weight-sha",
                candidate["model_sha256"],
                "--selection-count",
                str(selection_count),
            ],
            stdout_path=output_dir.parent / f"{output_dir.name}-export.log",
        )

    def _build_replay(
        self,
        *,
        candidate: dict[str, Any],
        role: str,
        source_dir: Path,
        output_dir: Path,
        expected_new_rows: int = 4,
        designer_source_dir: Path | None = None,
        auxiliary_source_dirs: list[Path] | None = None,
    ) -> dict[str, Any]:
        auxiliary_source_dirs = auxiliary_source_dirs or []
        if not output_dir.exists():
            command: list[str | Path] = [
                self.config.uv_bin,
                "run",
                "--no-sync",
                self.config.repo_root / "scripts/combine_q35_2b_role_replay_sft_v1.py",
                "--new-source",
                source_dir,
                "--output-dir",
                output_dir,
                "--role",
                role,
                "--max-rows",
                "16" if role == "coordinator" else "12",
            ]
            if designer_source_dir is not None:
                command.extend(["--designer-source", designer_source_dir])
            for auxiliary in auxiliary_source_dirs:
                command.extend(["--auxiliary-source", auxiliary])
            prior = candidate.get("replay")
            if prior is not None:
                command.extend(["--prior-replay", prior["path"]])
            self._run(command, stdout_path=output_dir.parent / f"{output_dir.name}-combine.log")
        manifest = _json(output_dir / "MANIFEST.json")
        if (
            manifest.get("schema_version") not in replay_builder.SUPPORTED_REPLAY_SCHEMA_VERSIONS
            or manifest.get("role") != role
            or manifest.get("new_rows") != expected_new_rows
            or manifest.get("new_auxiliary_rows", 0)
            != sum(_json(path / "MANIFEST.json")["rows_by_role"][role] for path in auxiliary_source_dirs)
            or manifest.get("new_designer_rows", 0)
            != ((_json(designer_source_dir / "MANIFEST.json")["rows"]) if designer_source_dir is not None else 0)
            or not 1 <= manifest.get("rows", 0) <= (16 if role == "coordinator" else 12)
        ):
            raise ValueError(f"invalid role replay corpus: {output_dir}")
        return manifest

    @staticmethod
    def _next_label(label: str) -> str:
        match = re.fullmatch(r"(.*?)([0-9]+)", label)
        if match is None:
            raise ValueError(f"candidate label does not end in an integer: {label}")
        return f"{match.group(1)}{int(match.group(2)) + 1}"

    @staticmethod
    def _quote(value: str | Path) -> str:
        return json.dumps(str(value))

    def _training_config(self, *, run_name: str, model_path: Path, replay: Path, rows: int) -> str:
        quote = self._quote
        if not 1 <= rows <= 16:
            raise ValueError("dense role replay must contain between one and sixteen rows")
        distributed_batch_size = rows + rows % 2
        return f"""max_steps = 1
output_dir = {quote(self.config.output_root)}
clean = false

[run]
name = {quote(run_name)}
dir = {quote(run_name)}

[deployment]
type = "single_node"
gpus_per_node = 2
num_gpus = 2

[model]
name = {quote(model_path)}
impl = "custom"
optimization_dtype = "bfloat16"
reduce_dtype = "bfloat16"

[model.vlm]
vision_encoder_attr = "model.visual"
language_model_attr = "model.language_model"
freeze_vision_encoder = true

[model.compile]
fullgraph = false

[model.ac]
mode = "full"
freq = 1
targets = ["norm"]

[tokenizer]
name = {quote(model_path)}

[renderer]
name = "qwen3.5"
enable_thinking = false

[data]
type = "sft"
name = {quote(replay)}
batch_size = {distributed_batch_size}
micro_batch_size = 1
seq_len = 16384
shuffle = false
seed = 20260823

[data.loss_mask]
system = false
user = false
assistant = true
tool = false

[optim]
type = "adamw"
lr = {self.config.learning_rate:.12g}
weight_decay = 0.01
max_norm = 1.0
betas1 = 0.9
betas2 = 0.999

[scheduler]
type = "constant"

[ckpt]
interval = 1
keep_last = 1
weights_only = true

[ckpt.weights]
save_sharded = true
save_format = "safetensors"

[file_monitor]
filename = "metrics.jsonl"
"""

    def _training_attempt(self, base: str) -> tuple[str, Path]:
        for attempt in range(1, 100):
            name = base if attempt == 1 else f"{base}-attempt{attempt}"
            output = self.config.output_root / name
            weight = output / "weights/step_1/model.safetensors"
            metrics = output / "metrics.jsonl"
            if weight.is_file() and metrics.is_file() and (weight.parent / "STABLE").is_file():
                return name, output
            if not output.exists():
                return name, output
        raise RuntimeError("exhausted dense training attempt labels")

    @staticmethod
    def _metrics(path: Path) -> dict[str, Any]:
        metrics = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    metrics.update(json.loads(line))
        required = ("loss/mean", "loss/nan_count", "optim/grad_norm", "time/step")
        if any(key not in metrics for key in required) or metrics["loss/nan_count"] != 0:
            raise ValueError("dense training metrics are incomplete or non-finite")
        return metrics

    def _trained_coevolution_batches(self) -> set[str]:
        trained: set[str] = set()
        for manifest_path in self.config.artifacts_root.glob("*-replay/MANIFEST.json"):
            metadata = _json(manifest_path).get("environment_designer")
            if isinstance(metadata, dict):
                trained.update(metadata.get("trained_batch_ids") or [])
        return trained

    def _delayed_designer_source(
        self,
        *,
        status: dict[str, Any],
        output_dir: Path,
    ) -> Path | None:
        current_sha = status["candidates"]["coordinator"]["model_sha256"]
        trained = self._trained_coevolution_batches()
        eligible: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
        for score_path in self.config.artifacts_root.glob("spade-coevolution-*/SCORE.json"):
            generation_path = score_path.parent / "generation/GENERATION.json"
            if not generation_path.is_file():
                continue
            score = _json(score_path)
            generation = _json(generation_path)
            if (
                score.get("selected_environment_ids")
                and score.get("batch_id") not in trained
                and generation.get("designer_model", {}).get("weight_sha256") != current_sha
            ):
                eligible.append((score_path, score, generation))
        if not eligible:
            return None
        score_path, score, generation = sorted(eligible, key=lambda item: item[1]["batch_id"])[0]
        generation_path = score_path.parent / "generation/GENERATION.json"
        if not output_dir.exists():
            candidate = status["candidates"]["coordinator"]
            self._run(
                [
                    self.config.uv_bin,
                    "run",
                    "--no-sync",
                    self.config.repo_root / "scripts/q35_2b_spade_coevolution_v1.py",
                    "export-designer",
                    "--generation",
                    generation_path,
                    "--score",
                    score_path,
                    "--output-dir",
                    output_dir,
                    "--student-snapshot",
                    candidate["model_path"],
                    "--student-revision",
                    status["model_revision"],
                    "--student-weight-sha",
                    candidate["model_sha256"],
                    "--max-rows",
                    "2",
                ],
                stdout_path=output_dir.parent / f"{output_dir.name}-export.log",
            )
        manifest = _json(output_dir / "MANIFEST.json")
        if manifest.get("batch_id") != generation["batch_id"]:
            raise ValueError("delayed Designer corpus does not match its scored batch")
        return output_dir

    def _designer_repair_source(
        self,
        *,
        status: dict[str, Any],
        rejections: Path,
        output_dir: Path,
        track: str,
        phase: str,
    ) -> Path:
        candidate = status["candidates"]["coordinator"]
        if not output_dir.exists():
            self._run(
                [
                    self.config.uv_bin,
                    "run",
                    "--no-sync",
                    self.config.repo_root / "scripts/q35_2b_spade_coevolution_v1.py",
                    "export-repairs",
                    "--rejections",
                    rejections,
                    "--output-dir",
                    output_dir,
                    "--student-snapshot",
                    candidate["model_path"],
                    "--student-revision",
                    status["model_revision"],
                    "--student-weight-sha",
                    candidate["model_sha256"],
                    "--track",
                    track,
                    "--phase",
                    phase,
                    "--max-rows",
                    "2",
                ],
                stdout_path=output_dir.parent / f"{output_dir.name}-export.log",
            )
        manifest = _json(output_dir / "MANIFEST.json")
        if manifest.get("batch_id") != _json(rejections)["batch_id"]:
            raise ValueError("Designer repair corpus does not match its rejected batch")
        return output_dir

    def _coevolution_sources(
        self,
        *,
        status: dict[str, Any],
        sources: dict[str, dict[str, Any]],
        active_roles: tuple[str, ...],
        prefix: str,
    ) -> tuple[dict[str, list[Path]], Path | None, dict[str, Any]]:
        if not self.config.coevolution:
            return {}, None, {}
        prior_scores = list(self.config.artifacts_root.glob("spade-coevolution-*/SCORE.json"))
        available = [role for role in ("child", "coordinator") if role in active_roles]
        if not available:
            raise ValueError("coevolution requires an active role")
        preferred = ("child", "coordinator")[len(prior_scores) % 2]
        role = preferred if preferred in available else available[0]
        track = replay_builder.TRACK_FOR_ROLE[role]
        phase = sources[role]["phase"]
        fingerprint = status["event_head_sha256"][:12]
        batch_id = f"spade-{self._pair_label(status['candidates'])}-{track}-{fingerprint}"
        batch_dir = self.config.artifacts_root / f"spade-coevolution-{batch_id}"
        start_index = 8_000_000 + (int(fingerprint[:8], 16) % 100_000) * 10
        complete = batch_dir / "PAIRED_EVALUATIONS_COMPLETE"
        if not batch_dir.exists():
            self._gpus_idle()
            self._run(
                [
                    self.config.repo_root / "scripts/run_q35_2b_spade_coevolution_batch_v1.sh",
                    status["candidates"]["coordinator"]["model_path"],
                    status["candidates"]["child"]["model_path"],
                    batch_dir,
                    self.config.results_root,
                    batch_id,
                    status["model_revision"],
                    track,
                    phase,
                    str(start_index),
                    "6",
                    self.config.experiment_dir / "coevolution-memory.jsonl",
                    self._external_model(status["candidates"]),
                ]
            )
        rejections = batch_dir / "generation/REJECTIONS.json"
        if rejections.is_file() and not complete.is_file():
            repair = self._designer_repair_source(
                status=status,
                rejections=rejections,
                output_dir=self.config.artifacts_root / f"{prefix}-designer-repair-source",
                track=track,
                phase=phase,
            )
            metadata = {
                "batch_id": batch_id,
                "track": track,
                "phase": phase,
                "status": "designer_rejected_nonfatal",
                "rejections_path": str(rejections),
                "rejections_sha256": _sha256_file(rejections),
                "repair_source": str(repair),
                "paired_evaluations_run": False,
            }
            _append_journal(self.config.journal, {"kind": "coevolution_batch_repaired", **metadata})
            return {}, repair, metadata
        if not complete.is_file():
            raise RuntimeError(f"coevolution batch is partial and requires recovery: {batch_dir}")
        summaries: dict[str, Path] = {}
        for arm in ("no-hint", "hint"):
            run = self.config.results_root / f"{batch_id}-{arm}"
            traces = run / TASK_AXIS / "traces.jsonl"
            summary = batch_dir / f"{arm.upper()}_SUMMARY.json"
            if not summary.is_file():
                self._run(
                    [
                        self.config.uv_bin,
                        "run",
                        "--no-sync",
                        self.config.repo_root / "scripts/summarize_q35_2b_interaction_curriculum_v1.py",
                        traces,
                        "--phase",
                        phase,
                        "--output",
                        summary,
                    ]
                )
            summaries[arm] = summary
        score_path = batch_dir / "SCORE.json"
        self._run(
            [
                self.config.uv_bin,
                "run",
                "--no-sync",
                self.config.repo_root / "scripts/q35_2b_spade_coevolution_v1.py",
                "score",
                "--generation",
                batch_dir / "generation/GENERATION.json",
                "--no-hint-summary",
                summaries["no-hint"],
                "--hint-summary",
                summaries["hint"],
                "--memory",
                self.config.experiment_dir / "coevolution-memory.jsonl",
                "--output",
                score_path,
            ]
        )
        counts = {
            arm: len(_json(summary).get("qualifying") or [])
            for arm, summary in summaries.items()
        }
        best_arm = max(("hint", "no-hint"), key=lambda arm: counts[arm])
        auxiliary: dict[str, list[Path]] = {}
        if counts[best_arm] > 0:
            run = self.config.results_root / f"{batch_id}-{best_arm}"
            source_dir = self.config.artifacts_root / f"{prefix}-{role}-coevolution-source"
            source = {
                "phase": phase,
                "artifacts": {
                    "traces_path": str(run / TASK_AXIS / "traces.jsonl"),
                    "summary_path": str(summaries[best_arm]),
                    "summary_sha256": _sha256_file(summaries[best_arm]),
                    "versions_path": str(run / "VERSIONS.txt"),
                },
            }
            self._export_role(
                status=status,
                source=source,
                role=role,
                output_dir=source_dir,
                selection_count=min(4, counts[best_arm]),
            )
            auxiliary[role] = [source_dir]
        delayed = self._delayed_designer_source(
            status=status,
            output_dir=self.config.artifacts_root / f"{prefix}-rewarded-designer-source",
        )
        metadata = {
            "batch_id": batch_id,
            "track": track,
            "phase": phase,
            "score_path": str(score_path),
            "score_sha256": _sha256_file(score_path),
            "qualifying_by_arm": counts,
            "replayed_arm": best_arm if counts[best_arm] else None,
            "delayed_designer_source": str(delayed) if delayed else None,
        }
        _append_journal(self.config.journal, {"kind": "coevolution_batch_scored", **metadata})
        return auxiliary, delayed, metadata

    def train_roles(self, status: dict[str, Any]) -> None:
        action = status["next_action"]
        if action.get("kind") not in {"train_pair", "train_roles"}:
            raise ValueError("runner was asked to train without dense role authorization")
        optimizer_steps = action.get("full_optimizer_steps_authorized")
        if optimizer_steps not in (
            {"child": 1, "coordinator": 1},
            {"child": 0, "coordinator": 1},
            {"child": 1, "coordinator": 0},
        ):
            raise ValueError("runner received an invalid dense role authorization")
        active_roles = tuple(role for role in ("coordinator", "child") if optimizer_steps[role] == 1)
        for candidate in status["candidates"].values():
            self._verify_candidate(candidate)
        sources = {
            controller.ROLE_FOR_TRACK[track]: self._source_event(source["bank_id"])
            for track, source in action["sources"].items()
        }
        fingerprint = status["event_head_sha256"][:12]
        prefix = f"dual-dense-{self._pair_label(status['candidates'])}-{fingerprint}"
        source_dirs = {role: self.config.artifacts_root / f"{prefix}-{role}-source" for role in sources}
        partial_source_dirs = {
            role: self.config.artifacts_root / f"{prefix}-{role}-positive-prefix-source" for role in sources
        }
        replay_dirs = {role: self.config.artifacts_root / f"{prefix}-{role}-replay" for role in sources}
        designer_source_dir = self.config.artifacts_root / f"{prefix}-environment-designer-source"
        replay_manifests = {}
        if set(sources) != set(active_roles):
            raise ValueError("admitted sources do not match the authorized roles")
        auxiliary_sources, rewarded_designer_source, coevolution_metadata = self._coevolution_sources(
            status=status,
            sources=sources,
            active_roles=active_roles,
            prefix=prefix,
        )
        failed_trajectory_rows = 0
        for role in active_roles:
            selection_count = min(
                4,
                sources[role]["admission"]["qualifying_trajectories"],
                sources[role]["admission"]["distinct_task_keys"],
            )
            partial = self._export_positive_prefixes(
                status=status,
                source=sources[role],
                role=role,
                output_dir=partial_source_dirs[role],
            )
            partial_rows = _json(partial / "MANIFEST.json")["rows"] if partial is not None else 0
            failed_trajectory_rows += partial_rows
            role_auxiliary = list(auxiliary_sources.get(role, []))
            if selection_count:
                self._export_role(
                    status=status,
                    source=sources[role],
                    role=role,
                    output_dir=source_dirs[role],
                    selection_count=selection_count,
                )
                primary_source = source_dirs[role]
                expected_new_rows = selection_count
                if partial is not None:
                    role_auxiliary.append(partial)
            elif partial is not None:
                primary_source = partial
                expected_new_rows = partial_rows
            else:
                raise ValueError("training authorization lacks complete or positive-prefix rows")
            role_designer_source = None
            if role == "coordinator" and not self.config.coevolution:
                self._export_designer(
                    status=status,
                    source=sources[role],
                    output_dir=designer_source_dir,
                    selection_count=selection_count,
                )
                role_designer_source = designer_source_dir
            elif role == "coordinator":
                role_designer_source = rewarded_designer_source
            replay_manifests[role] = self._build_replay(
                candidate=status["candidates"][role],
                role=role,
                source_dir=primary_source,
                output_dir=replay_dirs[role],
                designer_source_dir=role_designer_source,
                auxiliary_source_dirs=role_auxiliary,
                expected_new_rows=expected_new_rows,
            )

        outputs = {}
        results = {}
        for role in active_roles:
            candidate = status["candidates"][role]
            next_label = self._next_label(candidate["label"])
            base_name = f"dual-dense-{role}-{candidate['label'].lower()}-to-{next_label.lower()}-{fingerprint}"
            run_name, output = self._training_attempt(base_name)
            config_path = self.config.experiment_dir / f"{run_name}.toml"
            config_text = self._training_config(
                run_name=run_name,
                model_path=Path(candidate["model_path"]),
                replay=replay_dirs[role],
                rows=replay_manifests[role]["rows"],
            )
            _write_text_once_or_match(config_path, config_text)
            output_model = output / "weights/step_1"
            output_weight = output_model / MODEL_WEIGHT
            if not output_weight.is_file():
                self._gpus_idle()
                environment = os.environ.copy()
                environment.update({"NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "0"})
                self._run(
                    [self.config.uv_bin, "run", "--no-sync", "sft", "@", config_path],
                    env=environment,
                )
            metrics_path = output / "metrics.jsonl"
            metrics = self._metrics(metrics_path)
            output_sha = _sha256_file(output_weight)
            if output_sha == candidate["model_sha256"]:
                raise ValueError(f"{role} full update did not change dense weights")
            outputs[role] = {
                "label": next_label,
                "model": f"q35-2b-{role}-{next_label.lower()}-{output_sha[:12]}",
                "model_path": str(output_model.resolve()),
                "model_sha256": output_sha,
                "replay": {
                    "role": role,
                    "path": str(replay_dirs[role].resolve()),
                    "manifest_sha256": _sha256_file(replay_dirs[role] / "MANIFEST.json"),
                    "train_parquet_sha256": _sha256_file(replay_dirs[role] / "train.parquet"),
                    "rows": replay_manifests[role]["rows"],
                    "new_rows": replay_manifests[role]["new_rows"],
                    "new_designer_rows": replay_manifests[role].get("new_designer_rows", 0),
                    "new_auxiliary_rows": replay_manifests[role].get("new_auxiliary_rows", 0),
                    "new_partial_rows": replay_manifests[role].get("new_partial_rows", 0),
                    "environment_designer": replay_manifests[role].get("environment_designer"),
                },
            }
            results[role] = {
                "config_path": str(config_path),
                "config_sha256": _sha256_file(config_path),
                "metrics_path": str(metrics_path),
                "metrics_sha256": _sha256_file(metrics_path),
                "loss_mean": metrics["loss/mean"],
                "loss_nan_count": metrics["loss/nan_count"],
                "gradient_norm": metrics["optim/grad_norm"],
                "time_per_step_seconds": metrics["time/step"],
            }
        projected_outputs = {**status["candidates"], **outputs}
        if projected_outputs["coordinator"]["model_sha256"] == projected_outputs["child"]["model_sha256"]:
            raise ValueError("role-specific full updates produced identical dense checkpoints")
        receipt = {
            "schema_version": controller.UPDATE_SCHEMA_VERSION,
            "input_model_sha256": {role: candidate["model_sha256"] for role, candidate in status["candidates"].items()},
            "optimizer_steps": optimizer_steps,
            "dense_model_updates": optimizer_steps,
            "lora_updates": 0,
            "failed_trajectory_rows": failed_trajectory_rows,
            "optimizer": {
                "type": "adamw",
                "learning_rate": self.config.learning_rate,
                "weight_decay": 0.01,
                "max_norm": 1.0,
                "scheduler": "constant",
            },
            "source_summary_sha256": {track: source["summary_sha256"] for track, source in action["sources"].items()},
            "coevolution": coevolution_metadata or None,
            "outputs": outputs,
        }
        receipt_path = self.config.experiment_dir / f"{prefix}-receipt.json"
        result_path = self.config.experiment_dir / f"{prefix}-result.json"
        _write_json_once_or_match(receipt_path, receipt)
        _write_json_once_or_match(
            result_path,
            {
                "schema_version": RUNNER_SCHEMA_VERSION,
                "status": "complete",
                "event_head_sha256": status["event_head_sha256"],
                "receipt_path": str(receipt_path),
                "receipt_sha256": _sha256_file(receipt_path),
                "roles": results,
            },
        )
        namespace = argparse.Namespace(events=self.config.events, receipt=receipt_path, recorded_at=None)
        with contextlib.redirect_stdout(io.StringIO()):
            controller._record_update(namespace)
        _append_journal(
            self.config.journal,
            {
                "kind": "update_pair_recorded" if len(active_roles) == 2 else "update_roles_recorded",
                "output_model_sha256": {role: output["model_sha256"] for role, output in outputs.items()},
                "event_head_sha256": self.status()["event_head_sha256"],
            },
        )

    def run(self) -> dict[str, Any]:
        initial = self.status()
        if self.config.dry_run:
            return {"schema_version": RUNNER_SCHEMA_VERSION, "status": "dry_run", "controller": initial}
        _append_journal(
            self.config.journal,
            {
                "kind": "runner_started",
                "event_head_sha256": initial["event_head_sha256"],
                "budgets": {
                    "open_ended": self.config.open_ended,
                    "max_evaluations": self.config.max_evaluations,
                    "max_update_pairs": self.config.max_update_pairs,
                    "max_actions": self.config.max_actions,
                },
                "training_recipe": {
                    "full_parameter_updates": True,
                    "causal_environment_coevolution": self.config.coevolution,
                    "designer_update": (
                        "one-update-delayed reward-filtered SFT" if self.config.coevolution else "static leak ladder SFT"
                    ),
                    "learning_rate": self.config.learning_rate,
                    "optimizer": "adamw",
                    "scheduler": "constant",
                },
            },
        )
        stop_reason = "no_action"
        while True:
            if self.config.stop_file.exists():
                stop_reason = "stop_file_present"
                break
            if not self.config.open_ended and self.actions >= self.config.max_actions:
                stop_reason = "action_budget_exhausted"
                break
            status = self.status()
            action = status["next_action"]
            if action["kind"] == "collect":
                if not self.config.open_ended and self.evaluations >= self.config.max_evaluations:
                    stop_reason = "evaluation_budget_exhausted"
                    break
                self.collect(status, action["arms"][0])
                self.evaluations += 1
            elif action["kind"] in {"train_pair", "train_roles"}:
                if not self.config.open_ended and self.update_pairs >= self.config.max_update_pairs:
                    stop_reason = "update_pair_budget_exhausted"
                    break
                self.train_roles(status)
                self.update_pairs += 1
            elif action["kind"] in {"select_pair", "select_roles"}:
                namespace = argparse.Namespace(events=self.config.events, recorded_at=None)
                with contextlib.redirect_stdout(io.StringIO()):
                    controller._record_selection(namespace)
                _append_journal(
                    self.config.journal,
                    {
                        "kind": "candidate_roles_selected",
                        "input_candidate_sha256": {
                            role: candidate["model_sha256"] for role, candidate in status["candidates"].items()
                        },
                        "selected_candidate_sha256": {
                            role: candidate["model_sha256"] for role, candidate in action["selected_candidates"].items()
                        },
                        "event_head_sha256": self.status()["event_head_sha256"],
                    },
                )
            else:
                raise ValueError(f"unsupported controller action: {action['kind']}")
            self.actions += 1
        final = self.status()
        report = {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "status": "stopped_cleanly",
            "stop_reason": stop_reason,
            "actions_completed": self.actions,
            "evaluations_completed": self.evaluations,
            "update_pairs_completed": self.update_pairs,
            "controller": final,
        }
        _append_journal(
            self.config.journal,
            {
                "kind": "runner_stopped",
                "stop_reason": stop_reason,
                "actions_completed": self.actions,
                "evaluations_completed": self.evaluations,
                "update_pairs_completed": self.update_pairs,
                "event_head_sha256": final["event_head_sha256"],
            },
        )
        return report


def _nonnegative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("budget must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive finite float")
    return parsed


def main() -> None:
    repo_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_default)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path)
    local_uv = Path.home() / ".local/bin/uv"
    parser.add_argument(
        "--uv-bin",
        default=shutil.which("uv") or (str(local_uv) if local_uv.is_file() else "uv"),
    )
    parser.add_argument("--max-evaluations", type=_nonnegative, default=12)
    parser.add_argument("--max-update-pairs", type=_nonnegative, default=2)
    parser.add_argument("--max-actions", type=_nonnegative, default=14)
    parser.add_argument(
        "--open-ended",
        action="store_true",
        help="Ignore numeric budgets and continue until the stop file is present.",
    )
    parser.add_argument("--learning-rate", type=_positive_float, default=5e-6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--coevolution",
        action="store_true",
        help="Run paired generated environments and delayed reward-filtered Designer updates before each role update.",
    )
    args = parser.parse_args()
    if not Path(args.uv_bin).is_file() and shutil.which(args.uv_bin) is None:
        raise FileNotFoundError(f"uv executable does not exist: {args.uv_bin}")
    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    config = RunnerConfig(
        repo_root=args.repo_root.resolve(),
        events=args.events.resolve(),
        artifacts_root=args.artifacts_root.resolve(),
        results_root=args.results_root.resolve(),
        output_root=args.output_root.resolve(),
        experiment_dir=(
            args.experiment_dir.resolve()
            if args.experiment_dir
            else args.repo_root.resolve() / "experiments/qwen35-2b-self-bootstrap-dual-dense-v1"
        ),
        journal=args.journal.resolve(),
        stop_file=args.stop_file.resolve(),
        uv_bin=args.uv_bin,
        learning_rate=args.learning_rate,
        max_evaluations=args.max_evaluations,
        max_update_pairs=args.max_update_pairs,
        max_actions=args.max_actions,
        open_ended=args.open_ended,
        dry_run=args.dry_run,
        coevolution=args.coevolution,
    )
    with args.lock_file.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another dual-dense SPADE runner owns the lock") from error
        report = AutonomousDualDenseRunner(config).run()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
