#!/usr/bin/env python3
"""Resumable unattended executor for the Qwen3.5-2B SPADE interaction loop."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import q35_2b_spade_interaction_loop_v1 as controller

RUNNER_SCHEMA_VERSION = "qwen35-2b-spade-autonomous-runner/v1"
JOURNAL_SCHEMA_VERSION = "qwen35-2b-spade-autonomous-journal-event/v1"
BASE_WEIGHT_NAME = "model.safetensors"
ADAPTER_WEIGHT_NAME = "adapter_model.safetensors"
TASK_AXIS = "natural_n1a"
SAMPLING_SEED = "20260822"

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


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


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


def _journal_digest(event: dict[str, Any]) -> str:
    return _sha256_text(_canonical_json(event))


def _append_journal(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        existing = [json.loads(line) for line in handle if line.strip()]
        previous = existing[-1]["event_sha256"] if existing else None
        event = {
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "sequence": len(existing),
            "previous_event_sha256": previous,
            "recorded_at_utc": _now(),
            **payload,
        }
        event["event_sha256"] = _journal_digest(event)
        handle.seek(0, 2)
        handle.write(_canonical_json(event) + "\n")
        handle.flush()


@dataclass(frozen=True)
class RunnerConfig:
    repo_root: Path
    events: Path
    base_model: Path
    initial_adapter_path: Path | None
    artifacts_root: Path
    results_root: Path
    output_root: Path
    experiment_dir: Path
    journal: Path
    stop_file: Path
    uv_bin: str
    max_evaluations: int
    max_updates: int
    max_actions: int
    dry_run: bool


class AutonomousRunner:
    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        self.evaluations = 0
        self.updates = 0
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
        resolved_command = [str(item) for item in command]
        _append_journal(
            self.config.journal,
            {"kind": "command_started", "command": resolved_command},
        )
        if stdout_path is None:
            subprocess.run(
                resolved_command,
                cwd=self.config.repo_root,
                env=env,
                check=True,
            )
        else:
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            # Keep prior failed-attempt output and append the retry. The journal
            # brackets each invocation, so a stale log must not block recovery.
            with stdout_path.open("a", encoding="utf-8") as handle:
                subprocess.run(
                    resolved_command,
                    cwd=self.config.repo_root,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        _append_journal(
            self.config.journal,
            {"kind": "command_completed", "command": resolved_command},
        )

    def _adapter_path(self, candidate: dict[str, Any]) -> Path:
        stored = candidate.get("adapter_path")
        path = Path(stored) if isinstance(stored, str) else self.config.initial_adapter_path
        if path is None:
            raise ValueError("current candidate has no adapter path; supply --initial-adapter-path")
        weight = path / ADAPTER_WEIGHT_NAME
        if not path.is_absolute() or not weight.is_file():
            raise ValueError(f"current candidate adapter is incomplete: {path}")
        if _sha256_file(weight) != candidate["adapter_sha256"]:
            raise ValueError("current candidate adapter path does not match the event log")
        return path

    def _verify_base(self, candidate: dict[str, Any]) -> None:
        weight = self.config.base_model / BASE_WEIGHT_NAME
        if not (self.config.base_model / "STABLE").is_file() or not weight.is_file():
            raise ValueError("immutable base is incomplete or lacks STABLE")
        if _sha256_file(weight) != candidate["base_sha256"]:
            raise ValueError("immutable base SHA-256 differs from the event log")

    def _gpus_idle(self) -> None:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            raise RuntimeError("refusing to start an action while a GPU process is active")

    @staticmethod
    def _bank_base_label(candidate: dict[str, Any], arm: dict[str, Any]) -> str:
        tag = PHASE_TAGS[arm["phase"]]
        return (
            f"{candidate['label'].lower()}-{arm['track']}-{tag}-"
            f"{arm['start_index']}-n{arm['tasks']}"
        )

    def _bootstrap_path(self, candidate: dict[str, Any], arm: dict[str, Any]) -> Path:
        return self.config.artifacts_root / f"spade-loop-{self._bank_base_label(candidate, arm)}-bootstrap.json"

    def _ensure_bootstrap(self, candidate: dict[str, Any], arm: dict[str, Any]) -> Path:
        path = self._bootstrap_path(candidate, arm)
        if path.exists():
            payload = _json(path)
            expected_axis = [{"name": TASK_AXIS, "start_index": arm["start_index"]}]
            records = payload.get("records") or []
            if (
                payload.get("axes") != expected_axis
                or payload.get("tasks_per_axis") != arm["tasks"]
                or payload.get("gradient_updates") != 0
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
                "action_scaffold",
            ]
        )
        return self._ensure_bootstrap(candidate, arm)

    @staticmethod
    def _trace_count(path: Path) -> int:
        if not path.is_file():
            return 0
        with path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    def _evaluation_run(self, base_label: str, tasks: int) -> tuple[str, Path, bool]:
        for attempt in range(1, 100):
            label = base_label if attempt == 1 else f"{base_label}-attempt{attempt}"
            run = self.config.results_root / label
            summary = run / "INTERACTION_SUMMARY.json"
            traces = run / TASK_AXIS / "traces.jsonl"
            if summary.is_file():
                return label, run, True
            if run.exists() and self._trace_count(traces) >= tasks:
                return label, run, False
            if not run.exists():
                return label, run, False
        raise RuntimeError("exhausted evaluation attempt labels")

    def _probe_environment(
        self,
        *,
        candidate: dict[str, Any],
        adapter_path: Path,
        bootstrap: Path,
        arm: dict[str, Any],
    ) -> dict[str, str]:
        label, run, summarized = self._evaluation_run(
            self._bank_base_label(candidate, arm), arm["tasks"]
        )
        traces = run / TASK_AXIS / "traces.jsonl"
        summary = run / "INTERACTION_SUMMARY.json"
        versions = run / "VERSIONS.txt"
        if not run.exists():
            self._gpus_idle()
            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{Path(self.config.uv_bin).parent}:{self.config.repo_root / '.venv/bin'}:{environment.get('PATH', '')}",
                    "NCCL_P2P_DISABLE": "1",
                    "EVAL_DRIVER": "scripts/run_qwen38_27b_prime_harness_qualification_v1.sh",
                    "EVAL_EXPERIMENT_DIR": "experiments/qwen38-27b-prime-harness-qualification-v1",
                    "PRIME_MASTERY_OUTPUT_ROOT": str(self.config.results_root),
                    "QWEN38_QUALIFICATION_OUTPUT_ROOT": str(self.config.results_root),
                    "QWEN38_QUALIFICATION_AXES": TASK_AXIS,
                    "QWEN38_QUALIFICATION_NUM_TASKS": str(arm["tasks"]),
                    "QWEN38_QUALIFICATION_NUM_ROLLOUTS": "1",
                    "QWEN38_QUALIFICATION_MAX_CONCURRENT": str(arm["tasks"]),
                    "QWEN38_QUALIFICATION_START_INDEX": str(arm["start_index"]),
                    "QUALIFICATION_REASONING_EFFORT": "high",
                    "QUALIFICATION_SAMPLING_SEED": SAMPLING_SEED,
                    "QUALIFICATION_SAMPLING_TEMPERATURE": "0.6",
                    "QUALIFICATION_PRIVILEGED_BOOTSTRAP_PATH": str(bootstrap),
                    "PROCEDURAL_INTERACTION_CURRICULUM": arm["phase"],
                    "EVAL_CUDA_VISIBLE_DEVICES": "0",
                    "EVAL_TENSOR_PARALLEL_SIZE": "1",
                    "EVAL_DTYPE": "bfloat16",
                    "EVAL_MAX_MODEL_LEN": "32768",
                    "EVAL_LANGUAGE_MODEL_ONLY": "true",
                    "EVAL_DISABLE_CUSTOM_ALL_REDUCE": "false",
                    "EVAL_GPU_MEMORY_UTILIZATION": "0.80",
                    "EVAL_MAX_NUM_SEQS": "8",
                    "EVAL_MAX_NUM_BATCHED_TOKENS": "4096",
                    "EVAL_LORA_NAME": candidate["model"],
                    "EVAL_SERVED_MODEL": candidate["model"],
                    "EVAL_LORA_PATH": str(adapter_path),
                    "EVAL_MAX_LORA_RANK": "16",
                }
            )
            self._run(
                [
                    self.config.repo_root / "scripts/run_qwen35_27b_prime_agent_mastery_baseline_v2.sh",
                    self.config.base_model,
                    label,
                    candidate["model_revision"],
                ],
                env=environment,
            )
        if not summarized:
            if self._trace_count(traces) != arm["tasks"]:
                raise ValueError(f"evaluation did not produce exactly {arm['tasks']} traces: {run}")
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
        for required in (summary, versions, traces):
            if not required.is_file():
                raise FileNotFoundError(f"completed evaluation lacks artifact: {required}")
        return {
            "label": label,
            "run": str(run),
            "summary": str(summary),
            "versions": str(versions),
            "traces": str(traces),
        }

    def collect(self, status: dict[str, Any], arm: dict[str, Any]) -> None:
        candidate = status["candidate"]
        self._verify_base(candidate)
        adapter_path = self._adapter_path(candidate)
        bootstrap = self._ensure_bootstrap(candidate, arm)
        _append_journal(
            self.config.journal,
            {
                "kind": "evaluation_started",
                "event_head_sha256": status["event_head_sha256"],
                "candidate": candidate,
                "arm": arm,
                "bootstrap_path": str(bootstrap),
                "bootstrap_sha256": _sha256_file(bootstrap),
            },
        )
        artifacts = self._probe_environment(
            candidate=candidate,
            adapter_path=adapter_path,
            bootstrap=bootstrap,
            arm=arm,
        )
        namespace = argparse.Namespace(
            events=self.config.events,
            track=arm["track"],
            phase=arm["phase"],
            start_index=arm["start_index"],
            bank_id=artifacts["label"],
            summary=Path(artifacts["summary"]),
            versions=Path(artifacts["versions"]),
            traces=Path(artifacts["traces"]),
            bootstrap=bootstrap,
            recorded_at=None,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            controller._record_evaluation(namespace)
        _append_journal(
            self.config.journal,
            {
                "kind": "evaluation_recorded",
                "bank_id": artifacts["label"],
                "summary_sha256": _sha256_file(Path(artifacts["summary"])),
                "event_head_sha256": self.status()["event_head_sha256"],
            },
        )

    def _source_event(self, bank_id: str) -> dict[str, Any]:
        for event in reversed(controller._load_events(self.config.events)):
            if event.get("kind") == "evaluation_recorded" and event["bank"]["id"] == bank_id:
                return event
        raise ValueError(f"admitted source bank is absent from event log: {bank_id}")

    @staticmethod
    def _next_candidate_label(label: str) -> str:
        match = re.fullmatch(r"(.*Y)([0-9]+)", label)
        if match is None:
            raise ValueError(f"candidate label does not end in Y<number>: {label}")
        return f"{match.group(1)}{int(match.group(2)) + 1}"

    def _export_source(
        self,
        *,
        candidate: dict[str, Any],
        adapter_path: Path,
        source: dict[str, Any],
        role: str,
        output_dir: Path,
    ) -> None:
        if output_dir.exists():
            manifest = _json(output_dir / "MANIFEST.json")
            if (
                manifest.get("selected_roles") != [role]
                or manifest.get("rows_by_role") != {role: 4}
                or manifest.get("source", {}).get("summary_sha256")
                != source["artifacts"]["summary_sha256"]
            ):
                raise ValueError(f"existing source corpus does not match authorization: {output_dir}")
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
                candidate["model"],
                "--student-snapshot",
                self.config.base_model,
                "--student-revision",
                candidate["model_revision"],
                "--student-weight-sha",
                candidate["base_sha256"],
                "--roles",
                role,
                "--selection-count",
                "4",
                "--initial-adapter-path",
                adapter_path,
                "--initial-adapter-sha256",
                candidate["adapter_sha256"],
            ],
            stdout_path=output_dir.parent / f"{output_dir.name}-export.log",
        )

    def _combine_sources(
        self,
        *,
        child_dir: Path,
        yield_dir: Path,
        output_dir: Path,
        child_phase: str,
        yield_phase: str,
    ) -> None:
        if output_dir.exists():
            manifest = _json(output_dir / "MANIFEST.json")
            if manifest.get("rows") != 8 or manifest.get("rows_by_phase_and_role") != {
                f"{child_phase}:child": 4,
                f"{yield_phase}:orchestrator": 4,
            }:
                raise ValueError(f"existing joint corpus does not match authorization: {output_dir}")
            return
        self._run(
            [
                self.config.uv_bin,
                "run",
                "--no-sync",
                self.config.repo_root / "scripts/combine_q35_2b_split_frontier_sft_v1.py",
                "--child-corpus",
                child_dir,
                "--yield-corpus",
                yield_dir,
                "--output-dir",
                output_dir,
                "--child-phase",
                child_phase,
                "--yield-phase",
                yield_phase,
            ],
            stdout_path=output_dir.parent / f"{output_dir.name}-combine.log",
        )

    @staticmethod
    def _toml_string(value: str | Path) -> str:
        return json.dumps(str(value))

    def _training_config(
        self,
        *,
        run_name: str,
        adapter_path: Path,
        adapter_sha256: str,
        corpus: Path,
    ) -> str:
        quote = self._toml_string
        return f'''max_steps = 1
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
name = {quote(self.config.base_model)}
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

[model.lora]
rank = 16
alpha = 32.0
dropout = 0.0
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
initial_adapter_path = {quote(adapter_path)}
initial_adapter_sha256 = {quote(adapter_sha256)}

[tokenizer]
name = {quote(self.config.base_model)}

[renderer]
name = "qwen3.5"
enable_thinking = false

[data]
type = "sft"
name = {quote(corpus)}
batch_size = 8
micro_batch_size = 1
seq_len = 16384
shuffle = false
seed = 20260822

[data.loss_mask]
system = false
user = false
assistant = true
tool = false

[optim]
type = "adamw"
lr = 1e-5
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
save_adapter_separately = true

[file_monitor]
filename = "metrics.jsonl"
'''

    def _training_attempt(self, base_name: str) -> tuple[str, Path]:
        for attempt in range(1, 100):
            name = base_name if attempt == 1 else f"{base_name}-attempt{attempt}"
            output = self.config.output_root / name
            adapter = output / "weights/step_1/lora_adapters" / ADAPTER_WEIGHT_NAME
            metrics = output / "metrics.jsonl"
            if adapter.is_file() and metrics.is_file():
                return name, output
            if not output.exists():
                return name, output
        raise RuntimeError("exhausted training attempt labels")

    @staticmethod
    def _metrics(path: Path) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    metrics.update(json.loads(line))
        required = ("loss/mean", "loss/nan_count", "optim/grad_norm", "time/step")
        if any(key not in metrics for key in required) or metrics["loss/nan_count"] != 0:
            raise ValueError("training metrics are incomplete or non-finite")
        return metrics

    def train(self, status: dict[str, Any]) -> None:
        action = status["next_action"]
        if action.get("kind") != "train" or action.get("optimizer_steps_authorized") != 1:
            raise ValueError("runner was asked to train without one-step authorization")
        candidate = status["candidate"]
        self._verify_base(candidate)
        adapter_path = self._adapter_path(candidate)
        sources = {
            track: self._source_event(source["bank_id"])
            for track, source in action["sources"].items()
        }
        next_label = self._next_candidate_label(candidate["label"])
        fingerprint = status["event_head_sha256"][:12]
        prefix = f"spade-auto-{candidate['label'].lower()}-{fingerprint}"
        child_dir = self.config.artifacts_root / f"{prefix}-child-sft"
        yield_dir = self.config.artifacts_root / f"{prefix}-yield-sft"
        joint_dir = self.config.artifacts_root / f"{prefix}-joint-sft"
        self._export_source(
            candidate=candidate,
            adapter_path=adapter_path,
            source=sources["child"],
            role="child",
            output_dir=child_dir,
        )
        self._export_source(
            candidate=candidate,
            adapter_path=adapter_path,
            source=sources["yield"],
            role="orchestrator",
            output_dir=yield_dir,
        )
        self._combine_sources(
            child_dir=child_dir,
            yield_dir=yield_dir,
            output_dir=joint_dir,
            child_phase=sources["child"]["phase"],
            yield_phase=sources["yield"]["phase"],
        )
        joint_manifest = _json(joint_dir / "MANIFEST.json")
        if joint_manifest.get("rows") != 8:
            raise ValueError("joint corpus is not the required eight success-only rows")
        base_run_name = f"spade-auto-{candidate['label'].lower()}-to-{next_label.lower()}-{fingerprint}"
        run_name, output = self._training_attempt(base_run_name)
        config_path = self.config.experiment_dir / f"{run_name}.toml"
        config_text = self._training_config(
            run_name=run_name,
            adapter_path=adapter_path,
            adapter_sha256=candidate["adapter_sha256"],
            corpus=joint_dir,
        )
        _write_text_once_or_match(config_path, config_text)
        plan = {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "status": "authorized",
            "event_head_sha256": status["event_head_sha256"],
            "input_candidate": candidate,
            "output_candidate_label": next_label,
            "optimizer_steps": 1,
            "dense_base_updates": 0,
            "failed_trajectory_rows": 0,
            "sources": action["sources"],
            "joint_corpus": {
                "path": str(joint_dir),
                "train_parquet_sha256": _sha256_file(joint_dir / "train.parquet"),
                "manifest_sha256": _sha256_file(joint_dir / "MANIFEST.json"),
            },
            "config_path": str(config_path),
            "config_sha256": _sha256_file(config_path),
        }
        plan_path = self.config.experiment_dir / f"{run_name}-plan.json"
        _write_json_once_or_match(plan_path, plan)
        output_adapter = output / "weights/step_1/lora_adapters"
        output_weight = output_adapter / ADAPTER_WEIGHT_NAME
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
        if output_sha == candidate["adapter_sha256"]:
            raise ValueError("bounded update did not produce a distinct adapter")
        self._verify_base(candidate)
        output_model = f"q35-2b-{next_label.lower()}-{output_sha[:12]}"
        receipt = {
            "schema_version": controller.UPDATE_SCHEMA_VERSION,
            "initial_adapter_sha256": candidate["adapter_sha256"],
            "base_sha256_before": candidate["base_sha256"],
            "base_sha256_after": candidate["base_sha256"],
            "optimizer_steps": 1,
            "dense_base_updates": 0,
            "failed_trajectory_rows": 0,
            "output_adapter_sha256": output_sha,
            "output_adapter_path": str(output_adapter.resolve()),
            "output_candidate_label": next_label,
            "output_model": output_model,
            "source_summary_sha256": {
                track: source["summary_sha256"] for track, source in action["sources"].items()
            },
        }
        receipt_path = self.config.experiment_dir / f"{run_name}-receipt.json"
        _write_json_once_or_match(receipt_path, receipt)
        result = {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "status": "complete",
            "plan_path": str(plan_path),
            "receipt_path": str(receipt_path),
            "output_adapter_sha256": output_sha,
            "metrics_sha256": _sha256_file(metrics_path),
            "loss_mean": metrics["loss/mean"],
            "loss_nan_count": metrics["loss/nan_count"],
            "gradient_norm": metrics["optim/grad_norm"],
            "time_per_step_seconds": metrics["time/step"],
        }
        result_path = self.config.experiment_dir / f"{run_name}-result.json"
        _write_json_once_or_match(result_path, result)
        namespace = argparse.Namespace(
            events=self.config.events,
            receipt=receipt_path,
            recorded_at=None,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            controller._record_update(namespace)
        _append_journal(
            self.config.journal,
            {
                "kind": "update_recorded",
                "input_candidate": candidate["label"],
                "output_candidate": next_label,
                "output_adapter_sha256": output_sha,
                "event_head_sha256": self.status()["event_head_sha256"],
            },
        )

    def run(self) -> dict[str, Any]:
        initial = self.status()
        if self.config.dry_run:
            return {
                "schema_version": RUNNER_SCHEMA_VERSION,
                "status": "dry_run",
                "controller": initial,
            }
        _append_journal(
            self.config.journal,
            {
                "kind": "runner_started",
                "event_head_sha256": initial["event_head_sha256"],
                "budgets": {
                    "max_evaluations": self.config.max_evaluations,
                    "max_updates": self.config.max_updates,
                    "max_actions": self.config.max_actions,
                },
            },
        )
        stop_reason = "no_action"
        while True:
            if self.config.stop_file.exists():
                stop_reason = "stop_file_present"
                break
            if self.actions >= self.config.max_actions:
                stop_reason = "action_budget_exhausted"
                break
            status = self.status()
            action = status["next_action"]
            if action["kind"] == "collect":
                if not action["arms"]:
                    stop_reason = "no_collection_arms"
                    break
                if self.evaluations >= self.config.max_evaluations:
                    stop_reason = "evaluation_budget_exhausted"
                    break
                self.collect(status, action["arms"][0])
                self.evaluations += 1
            elif action["kind"] == "train":
                if self.updates >= self.config.max_updates:
                    stop_reason = "update_budget_exhausted"
                    break
                self.train(status)
                self.updates += 1
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
            "updates_completed": self.updates,
            "controller": final,
        }
        _append_journal(
            self.config.journal,
            {
                "kind": "runner_stopped",
                "stop_reason": stop_reason,
                "actions_completed": self.actions,
                "evaluations_completed": self.evaluations,
                "updates_completed": self.updates,
                "event_head_sha256": final["event_head_sha256"],
            },
        )
        return report


def _positive_or_zero(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("budget must be non-negative")
    return parsed


def _config(args: argparse.Namespace) -> RunnerConfig:
    repo_root = args.repo_root.resolve()
    return RunnerConfig(
        repo_root=repo_root,
        events=args.events.resolve(),
        base_model=args.base_model.resolve(),
        initial_adapter_path=(
            args.initial_adapter_path.resolve() if args.initial_adapter_path is not None else None
        ),
        artifacts_root=args.artifacts_root.resolve(),
        results_root=args.results_root.resolve(),
        output_root=args.output_root.resolve(),
        experiment_dir=(repo_root / "experiments/qwen35-2b-self-bootstrap-v1"),
        journal=args.journal.resolve(),
        stop_file=args.stop_file.resolve(),
        uv_bin=args.uv_bin,
        max_evaluations=args.max_evaluations,
        max_updates=args.max_updates,
        max_actions=args.max_actions,
        dry_run=args.dry_run,
    )


def main() -> None:
    repo_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_default)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--initial-adapter-path", type=Path)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--uv-bin", default=shutil.which("uv") or "uv")
    parser.add_argument("--max-evaluations", type=_positive_or_zero, default=12)
    parser.add_argument("--max-updates", type=_positive_or_zero, default=1)
    parser.add_argument("--max-actions", type=_positive_or_zero, default=13)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.events.is_file():
        raise FileNotFoundError(f"event log does not exist: {args.events}")
    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    with args.lock_file.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another autonomous SPADE runner owns the lock") from error
        report = AutonomousRunner(_config(args)).run()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
