#!/usr/bin/env python3
"""Prove the H-ITER Phase-0 terminal-recovery census without scientific work."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

SCHEMA = "prime-rl/latent-h-iter-phase0-object-inventory-robustness-proof/v1"
STATUS = "h_iter_phase0_object_inventory_robustness_validated"
RUN_ID = "h-iter-phase0-object-inventory-robustness-proof-run1"
OUTPUT = Path("/home/ubuntu/rlm/outputs/q35-2b-h-iter-phase0-object-inventory-robustness-proof-run1")
PRODUCTION_OUTPUT = Path(
    "/home/ubuntu/rlm/outputs/q35-2b-h-iter-phase0-deterministic-terminal-recovery-run1"
)
ANTECEDENT_OUTPUT = Path("/home/ubuntu/rlm/outputs/q35-2b-h-iter-phase0-generator-locality-run1")
ARTIFACT_REL = Path(
    "experiments/qwen35-2b-latent-workspace-v1/h-iter-phase0-generator-locality-v1"
)
RUNNER_REL = Path("scripts/latent/run_h_iter_phase0_generator_locality_v1.py")
MODULE_REL = Path("src/prime_rl/latent/h_iter_phase0.py")
BUILDER_REL = ARTIFACT_REL / "scientific-surface-builder.py"
BASELINE_REL = ARTIFACT_REL / "scientific-surface-b5cdb53.json"
ANTECEDENT_REL = ARTIFACT_REL / "deterministic-recovery-antecedent.json"
BUILDER_SHA256 = "2a560b609639efa56e5b19616c7597f942ea4ade24fbce6058ad9181ff577ee7"
BASELINE_FILE_SHA256 = "3e34288a365ed52e0ae6f5d3b8a2245aec246f5b577cb81aebc7d365d99632e2"
BASELINE_INTERNAL_SHA256 = "bef2a3772c61ef40cfd886d8d9ea408bd75ef59764b100136effeb16edec9a4d"
BASELINE_BYTES = 215904
ANTECEDENT_SHA256 = "ccda7a124238f4ba28912f62d82872942ca293dbd782a3b54b99f11679ac6777"
EXPECTED_EXECUTABLE = "/home/ubuntu/rlm/prime-rl/.venv/bin/python3"
EXPECTED_PREFIX = "/home/ubuntu/rlm/prime-rl/.venv"
EXPECTED_DISTRIBUTIONS = {
    "torch": "2.11.0+cu128",
    "transformers": "5.6.2",
    "tokenizers": "0.22.2",
    "pyarrow": "24.0.0",
}
INVENTORY_KEYS = {
    "transformers_modeling_modules",
    "pretrained_model_objects",
    "tokenizer_objects",
    "optimizer_objects",
    "uninspectable_count",
    "census_errors",
    "output_inventory",
    "candidate_files",
    "checkpoint_files",
    "cuda_initialized",
    "object_census_method",
    "cuda_observation_method",
    "relevant_modules_absent_for_preimport_inference",
}


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def strict_loads(value: bytes) -> dict[str, object]:
    def reject_constant(token: str) -> object:
        raise ValueError(f"nonfinite JSON token: {token}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    parsed = json.loads(
        value.decode("utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(parsed, dict):
        raise ValueError("proof JSON is not an object")
    return parsed


def run_git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, stdout=subprocess.PIPE, text=True
    ).stdout.strip()


def import_runner(repo: Path, name: str):
    path = repo / RUNNER_REL
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("runner import spec is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_zero_inventory(value: dict[str, object], *, torch_present: bool) -> None:
    if set(value) != INVENTORY_KEYS:
        raise RuntimeError("object inventory fields differ")
    if value["transformers_modeling_modules"] != []:
        raise RuntimeError("modeling module appeared")
    for key in ("pretrained_model_objects", "tokenizer_objects", "optimizer_objects", "uninspectable_count"):
        if value[key] != 0 or isinstance(value[key], bool):
            raise RuntimeError(f"object inventory count differs: {key}")
    if value["census_errors"] != [] or value["output_inventory"] != []:
        raise RuntimeError("object inventory is not clean")
    if value["candidate_files"] != [] or value["checkpoint_files"] != []:
        raise RuntimeError("candidate/checkpoint appeared")
    if value["cuda_initialized"] is not False:
        raise RuntimeError("CUDA initialized during robustness proof")
    if value["object_census_method"] != "gc_mro_scan_without_importing_model_tokenizer_or_optimizer_classes":
        raise RuntimeError("object census method differs")
    expected_cuda_method = (
        "torch.cuda.is_initialized"
        if torch_present
        else "torch_module_absence_proves_no_torch_cuda_runtime_contact"
    )
    if value["cuda_observation_method"] != expected_cuda_method:
        raise RuntimeError("CUDA observation method differs")
    if value["relevant_modules_absent_for_preimport_inference"] is torch_present:
        raise RuntimeError("preimport module-absence evidence differs")


def run_arm(arm: str, repo: Path, scratch: Path) -> dict[str, object]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA is not hidden")
    if not scratch.is_dir() or scratch.is_symlink() or list(scratch.iterdir()):
        raise RuntimeError("arm scratch directory is not exact and empty")
    if any(name in sys.modules for name in ("torch", "transformers", "tokenizers")):
        raise RuntimeError("relevant module imported before runner")
    runner = import_runner(repo, f"hiter_inventory_proof_{arm}")
    if "torch" in sys.modules:
        raise RuntimeError("runner import loaded Torch")
    if arm == "pre_torch":
        inventory = runner.object_inventory(None, scratch)
        validate_zero_inventory(inventory, torch_present=False)
        result = {
            "arm": arm,
            "real_gc_inventory": inventory,
            "torch_imported": False,
            "cuda_initialized_before": False,
            "cuda_initialized_after": False,
            "nonstring_module_accepted": None,
            "raising_metadata_error_count": 0,
            "cleanup_inventory": inventory,
        }
    elif arm == "torch_present":
        import torch

        before = torch.cuda.is_initialized()
        real = runner.object_inventory(torch, scratch)
        validate_zero_inventory(real, torch_present=True)
        original_get_objects = runner.gc.get_objects

        class NonStringModuleMeta(type):
            def __getattribute__(cls, name: str):
                if name == "__module__":
                    return object()
                return super().__getattribute__(name)

        class NonStringSentinel(metaclass=NonStringModuleMeta):
            pass

        nonstring_sentinel = NonStringSentinel()
        objects = [item for item in original_get_objects() if item is not nonstring_sentinel]
        runner.gc.get_objects = lambda: [*objects, nonstring_sentinel]
        nonstring = runner.object_inventory(torch, scratch)
        validate_zero_inventory(nonstring, torch_present=True)
        runner.gc.get_objects = original_get_objects
        del objects, nonstring_sentinel

        class RaisingModuleMeta(type):
            def __getattribute__(cls, name: str):
                if name == "__module__":
                    raise RuntimeError("synthetic module metadata failure")
                return super().__getattribute__(name)

        class RaisingSentinel(metaclass=RaisingModuleMeta):
            pass

        raising_sentinel = RaisingSentinel()
        objects = [item for item in original_get_objects() if item is not raising_sentinel]
        runner.gc.get_objects = lambda: [*objects, raising_sentinel]
        negative = runner.object_inventory(torch, scratch)
        if negative["uninspectable_count"] != 1 or len(negative["census_errors"]) != 1:
            raise RuntimeError("raising-metadata negative did not produce one census error")
        error = negative["census_errors"][0]
        if set(error) != {"object_index", "error_type", "error"} or error["error_type"] != "RuntimeError":
            raise RuntimeError("raising-metadata error schema differs")
        if error["error"] != "synthetic module metadata failure":
            raise RuntimeError("raising-metadata error text differs")
        if error["object_index"] != len(objects):
            raise RuntimeError("raising-metadata error index differs")
        runner.gc.get_objects = original_get_objects
        del objects, raising_sentinel
        gc.collect()
        cleanup = runner.object_inventory(torch, scratch)
        validate_zero_inventory(cleanup, torch_present=True)
        after = torch.cuda.is_initialized()
        if before or after:
            raise RuntimeError("CUDA initialized during torch-present arm")
        result = {
            "arm": arm,
            "real_gc_inventory": real,
            "torch_imported": True,
            "torch_runtime": torch.__version__,
            "cuda_initialized_before": before,
            "cuda_initialized_after": after,
            "nonstring_module_accepted": True,
            "raising_metadata_error_count": 1,
            "raising_metadata_error": error,
            "cleanup_inventory": cleanup,
        }
    else:
        raise RuntimeError("unknown proof arm")
    if list(scratch.iterdir()):
        raise RuntimeError("proof arm wrote a scratch artifact")
    return result


def validate_proof(proof: dict[str, object], repo: Path, execution_commit: str) -> None:
    required = {
        "schema_version", "status", "run_identity", "execution_commit", "execution_tree",
        "runtime", "environment", "source_identity", "recovery_antecedent",
        "namespace_preflight", "scientific_surface", "arms", "safety", "full_freeze",
        "proof_sha256",
    }
    if set(proof) != required:
        raise RuntimeError("robustness proof fields differ")
    if (proof["schema_version"], proof["status"], proof["run_identity"]) != (SCHEMA, STATUS, RUN_ID):
        raise RuntimeError("robustness proof identity differs")
    if proof["execution_commit"] != execution_commit or proof["execution_tree"] != run_git(repo, "rev-parse", f"{execution_commit}^{{tree}}"):
        raise RuntimeError("robustness proof commit/tree differs")
    if proof["proof_sha256"] != sha256_bytes(canonical_json({k: v for k, v in proof.items() if k != "proof_sha256"})):
        raise RuntimeError("robustness proof self hash differs")
    if proof["environment"] != {
        "cuda_visible_devices": "", "hf_hub_offline": "1", "transformers_offline": "1",
        "wandb_mode": "offline",
    }:
        raise RuntimeError("robustness proof environment differs")
    if proof["runtime"] != {
        "python": "3.12.14",
        "sys_executable": EXPECTED_EXECUTABLE,
        "sys_prefix": EXPECTED_PREFIX,
        "distributions": EXPECTED_DISTRIBUTIONS,
    }:
        raise RuntimeError("robustness proof runtime differs")
    source = proof["source_identity"]
    if set(source) != {
        "runner_sha256", "module_sha256", "proof_script_sha256",
        "scientific_surface_builder_sha256", "scientific_surface_baseline_file_sha256",
    } or any(not isinstance(value, str) or len(value) != 64 for value in source.values()):
        raise RuntimeError("robustness proof source schema differs")
    if source["scientific_surface_builder_sha256"] != BUILDER_SHA256 or source["scientific_surface_baseline_file_sha256"] != BASELINE_FILE_SHA256:
        raise RuntimeError("robustness proof source identity differs")
    if proof["recovery_antecedent"]["antecedent_sha256"] != ANTECEDENT_SHA256:
        raise RuntimeError("robustness proof antecedent differs")
    if proof["namespace_preflight"] != {
        "antecedent_output_exists_empty": True,
        "production_output_absent": True,
        "proof_output_absent_before_write": True,
    }:
        raise RuntimeError("robustness proof namespace boundary differs")
    surface = proof["scientific_surface"]
    if surface != {
        "builder_sha256": BUILDER_SHA256,
        "baseline_file_sha256": BASELINE_FILE_SHA256,
        "baseline_internal_sha256": BASELINE_INTERNAL_SHA256,
        "baseline_bytes": BASELINE_BYTES,
        "candidate_file_sha256": BASELINE_FILE_SHA256,
        "candidate_internal_sha256": BASELINE_INTERNAL_SHA256,
        "candidate_bytes": BASELINE_BYTES,
        "candidate_byte_identical": True,
        "builder_runtime_python": "3.12.14",
    }:
        raise RuntimeError("scientific surface closure differs")
    arms = proof["arms"]
    if not isinstance(arms, list) or [arm.get("arm") for arm in arms] != ["pre_torch", "torch_present"]:
        raise RuntimeError("robustness proof arm order differs")
    if set(arms[0]) != {
        "arm", "real_gc_inventory", "torch_imported", "cuda_initialized_before",
        "cuda_initialized_after", "nonstring_module_accepted", "raising_metadata_error_count",
        "cleanup_inventory",
    }:
        raise RuntimeError("pre-Torch arm fields differ")
    if set(arms[1]) != {
        "arm", "real_gc_inventory", "torch_imported", "torch_runtime",
        "cuda_initialized_before", "cuda_initialized_after", "nonstring_module_accepted",
        "raising_metadata_error_count", "raising_metadata_error", "cleanup_inventory",
    }:
        raise RuntimeError("Torch-present arm fields differ")
    validate_zero_inventory(arms[0]["real_gc_inventory"], torch_present=False)
    validate_zero_inventory(arms[0]["cleanup_inventory"], torch_present=False)
    validate_zero_inventory(arms[1]["real_gc_inventory"], torch_present=True)
    validate_zero_inventory(arms[1]["cleanup_inventory"], torch_present=True)
    if arms[0]["torch_imported"] is not False or arms[1]["torch_imported"] is not True:
        raise RuntimeError("robustness proof Torch arm evidence differs")
    if arms[1]["nonstring_module_accepted"] is not True or arms[1]["raising_metadata_error_count"] != 1:
        raise RuntimeError("robustness proof sentinel evidence differs")
    if arms[0]["nonstring_module_accepted"] is not None or arms[0]["raising_metadata_error_count"] != 0:
        raise RuntimeError("pre-Torch sentinel evidence differs")
    if arms[1]["torch_runtime"] != EXPECTED_DISTRIBUTIONS["torch"]:
        raise RuntimeError("Torch-present runtime differs")
    negative = arms[1]["raising_metadata_error"]
    if (
        not isinstance(negative, dict)
        or set(negative) != {"object_index", "error_type", "error"}
        or not isinstance(negative["object_index"], int)
        or isinstance(negative["object_index"], bool)
        or negative["object_index"] < 0
        or negative["error_type"] != "RuntimeError"
        or negative["error"] != "synthetic module metadata failure"
    ):
        raise RuntimeError("raising-metadata evidence differs")
    if any(arm["cuda_initialized_before"] or arm["cuda_initialized_after"] for arm in arms):
        raise RuntimeError("robustness proof initialized CUDA")
    if proof["safety"] != {
        "model_loaded": False, "tokenizer_loaded": False, "model_forwards": 0,
        "synthetic_backwards": 0, "optimizer_steps": 0, "scientific_exposure": False,
        "cuda_hidden": True, "cuda_initialized_before_after": False,
    }:
        raise RuntimeError("robustness proof safety differs")
    if proof["full_freeze"] != {
        "head_before": execution_commit, "head_after": execution_commit,
        "tree_before": proof["execution_tree"], "tree_after": proof["execution_tree"],
        "status_before": "", "status_after": "", "source_hashes_before_after_equal": True,
    }:
        raise RuntimeError("robustness proof full-freeze evidence differs")


def atomic_write(output: Path, proof: dict[str, object], repo: Path, execution_commit: str) -> None:
    if output != OUTPUT or output.exists() or output.is_symlink():
        raise RuntimeError("proof output namespace is not exact and fresh")
    output.mkdir(mode=0o700)
    target = output / "proof.json"
    temporary = output / ".proof.json.tmp"
    encoded = canonical_json(proof) + b"\n"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short robustness-proof write")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise
    finally:
        os.close(descriptor)
    os.replace(temporary, target)
    descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if list(path.name for path in output.iterdir()) != ["proof.json"]:
        raise RuntimeError("proof output inventory differs")
    reopened = target.read_bytes()
    if reopened != encoded:
        raise RuntimeError("proof bytes changed after publication")
    validate_proof(strict_loads(reopened), repo, execution_commit)


def parent(args: argparse.Namespace) -> None:
    repo = args.repo.resolve(strict=True)
    if args.output_dir != OUTPUT or not repo.is_dir() or repo.is_symlink():
        raise RuntimeError("proof path differs")
    if platform.python_version() != "3.12.14" or sys.executable != EXPECTED_EXECUTABLE or sys.prefix != EXPECTED_PREFIX:
        raise RuntimeError("proof Python runtime differs")
    if any(name in sys.modules for name in ("torch", "transformers", "tokenizers")):
        raise RuntimeError("proof parent imported a relevant runtime module")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1" or os.environ.get("WANDB_MODE") != "offline":
        raise RuntimeError("proof environment differs")
    for distribution, version in EXPECTED_DISTRIBUTIONS.items():
        observed = importlib.metadata.version(distribution)
        if observed != version:
            raise RuntimeError(f"proof distribution differs: {distribution}={observed}")
    if len(args.execution_commit) != 40 or run_git(repo, "rev-parse", "HEAD") != args.execution_commit:
        raise RuntimeError("proof execution commit differs")
    if run_git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("proof worktree is dirty")
    tree = run_git(repo, "rev-parse", "HEAD^{tree}")
    source_paths = [RUNNER_REL, MODULE_REL, BUILDER_REL, BASELINE_REL, ANTECEDENT_REL]
    source_before = {str(path): file_sha256(repo / path) for path in source_paths}
    if source_before[str(BUILDER_REL)] != BUILDER_SHA256 or source_before[str(BASELINE_REL)] != BASELINE_FILE_SHA256:
        raise RuntimeError("scientific surface source bytes differ")
    baseline_bytes = (repo / BASELINE_REL).read_bytes()
    if len(baseline_bytes) != BASELINE_BYTES:
        raise RuntimeError("scientific surface baseline length differs")
    baseline = strict_loads(baseline_bytes)
    if baseline.get("surface_sha256") != BASELINE_INTERNAL_SHA256:
        raise RuntimeError("scientific surface baseline internal hash differs")
    module = import_runner(repo, "hiter_inventory_proof_parent")
    antecedent = module.load_canonical_json(repo / ANTECEDENT_REL)
    module.validate_recovery_antecedent(antecedent)
    module.validate_recovery_antecedent_assets(repo / ARTIFACT_REL, antecedent)
    if not ANTECEDENT_OUTPUT.is_dir() or ANTECEDENT_OUTPUT.is_symlink() or list(ANTECEDENT_OUTPUT.iterdir()):
        raise RuntimeError("antecedent output namespace was not preserved empty")
    if PRODUCTION_OUTPUT.exists() or PRODUCTION_OUTPUT.is_symlink() or OUTPUT.exists() or OUTPUT.is_symlink():
        raise RuntimeError("recovery/proof output namespace is not fresh")

    with tempfile.TemporaryDirectory(prefix="hiter-object-inventory-proof-") as temporary_name:
        temporary = Path(temporary_name)
        candidate = temporary / "scientific-surface-candidate.json"
        built = subprocess.run(
            [sys.executable, str(repo / BUILDER_REL), str(repo), str(candidate), args.execution_commit],
            cwd=repo,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if built.stderr != "" or built.stdout.splitlines() != [BASELINE_INTERNAL_SHA256, BASELINE_FILE_SHA256]:
            raise RuntimeError("scientific surface builder output differs")
        candidate_bytes = candidate.read_bytes()
        if candidate_bytes != baseline_bytes:
            raise RuntimeError("candidate scientific surface differs from b5cdb53")
        arms = []
        for arm_name in ("pre_torch", "torch_present"):
            scratch = temporary / arm_name
            scratch.mkdir()
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--arm", arm_name, "--repo", str(repo), "--scratch", str(scratch)],
                cwd=repo,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, "PYTHONPATH": str(repo / "src")},
            )
            if completed.stderr != b"" or not completed.stdout.endswith(b"\n"):
                raise RuntimeError(f"proof arm output differs: {arm_name}")
            arm = strict_loads(completed.stdout[:-1])
            if canonical_json(arm) + b"\n" != completed.stdout:
                raise RuntimeError(f"proof arm output is not canonical: {arm_name}")
            arms.append(arm)

    if any(name in sys.modules for name in ("torch", "transformers", "tokenizers")):
        raise RuntimeError("proof parent loaded a relevant runtime module")
    source_after = {str(path): file_sha256(repo / path) for path in source_paths}
    head_after = run_git(repo, "rev-parse", "HEAD")
    tree_after = run_git(repo, "rev-parse", "HEAD^{tree}")
    status_after = run_git(repo, "status", "--porcelain", "--untracked-files=all")
    proof: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "run_identity": RUN_ID,
        "execution_commit": args.execution_commit,
        "execution_tree": tree,
        "runtime": {
            "python": platform.python_version(),
            "sys_executable": sys.executable,
            "sys_prefix": sys.prefix,
            "distributions": {name: importlib.metadata.version(name) for name in EXPECTED_DISTRIBUTIONS},
        },
        "environment": {
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "hf_hub_offline": os.environ["HF_HUB_OFFLINE"],
            "transformers_offline": os.environ["TRANSFORMERS_OFFLINE"],
            "wandb_mode": os.environ["WANDB_MODE"],
        },
        "source_identity": {
            "runner_sha256": source_before[str(RUNNER_REL)],
            "module_sha256": source_before[str(MODULE_REL)],
            "proof_script_sha256": file_sha256(Path(__file__)),
            "scientific_surface_builder_sha256": source_before[str(BUILDER_REL)],
            "scientific_surface_baseline_file_sha256": source_before[str(BASELINE_REL)],
        },
        "recovery_antecedent": {
            "antecedent_sha256": antecedent["antecedent_sha256"],
            "attempt3_log_sha256": antecedent["antecedent_assets"][1]["sha256"],
            "attempt3_exit_sha256": antecedent["antecedent_assets"][0]["sha256"],
            "claim": antecedent["claim"],
            "allowed_recovery_runs": antecedent["allowed_recovery_runs"],
        },
        "namespace_preflight": {
            "antecedent_output_exists_empty": True,
            "production_output_absent": True,
            "proof_output_absent_before_write": True,
        },
        "scientific_surface": {
            "builder_sha256": BUILDER_SHA256,
            "baseline_file_sha256": BASELINE_FILE_SHA256,
            "baseline_internal_sha256": BASELINE_INTERNAL_SHA256,
            "baseline_bytes": BASELINE_BYTES,
            "candidate_file_sha256": sha256_bytes(candidate_bytes),
            "candidate_internal_sha256": strict_loads(candidate_bytes)["surface_sha256"],
            "candidate_bytes": len(candidate_bytes),
            "candidate_byte_identical": candidate_bytes == baseline_bytes,
            "builder_runtime_python": platform.python_version(),
        },
        "arms": arms,
        "safety": {
            "model_loaded": False,
            "tokenizer_loaded": False,
            "model_forwards": 0,
            "synthetic_backwards": 0,
            "optimizer_steps": 0,
            "scientific_exposure": False,
            "cuda_hidden": True,
            "cuda_initialized_before_after": False,
        },
        "full_freeze": {
            "head_before": args.execution_commit,
            "head_after": head_after,
            "tree_before": tree,
            "tree_after": tree_after,
            "status_before": "",
            "status_after": status_after,
            "source_hashes_before_after_equal": source_before == source_after,
        },
        "proof_sha256": "",
    }
    proof["proof_sha256"] = sha256_bytes(canonical_json({k: v for k, v in proof.items() if k != "proof_sha256"}))
    validate_proof(proof, repo, args.execution_commit)
    atomic_write(args.output_dir, proof, repo, args.execution_commit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--execution-commit")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--arm", choices=("pre_torch", "torch_present"))
    parser.add_argument("--scratch", type=Path)
    args = parser.parse_args()
    if args.arm is None and (args.execution_commit is None or args.output_dir is None or args.scratch is not None):
        parser.error("parent mode requires --execution-commit and --output-dir")
    if args.arm is not None and (args.scratch is None or args.execution_commit is not None or args.output_dir is not None):
        parser.error("arm mode requires --scratch only")
    return args


def main() -> None:
    args = parse_args()
    if args.arm is not None:
        result = run_arm(args.arm, args.repo.resolve(strict=True), args.scratch.resolve(strict=True))
        sys.stdout.buffer.write(canonical_json(result) + b"\n")
    else:
        parent(args)


if __name__ == "__main__":
    main()
