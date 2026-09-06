from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"scripts/latent"))

from freeze_h_iter_phase1_t0_plan_v1 import (
    FAILED_START_BINDING, GUARD_CASE_NAMES, GUARD_PROOF_ASSETS,
    guard_plan_value, validate_guard_plan,
)
from run_h_iter_phase1_t0_preflight_r1_guard_proof_v1 import (
    case_results, load_guard, normalized_source, validate_failed_start,
)

RUNNER=ROOT/"scripts/latent/run_h_iter_phase1_t0_v1.py"

def test_failed_start_archive_is_exact() -> None:
    manifest=validate_failed_start(ROOT)
    assert manifest["manifest_sha256"]==FAILED_START_BINDING["evidence_manifest_internal_sha256"]
    assert manifest["claim_boundary"]=={"classification":"infrastructure_invalid","validated_preflight":False,"output_namespace_created":False,"torch_imported":False,"transformers_imported":False,"tokenizer_loaded":False,"model_loaded":False,"cuda_initialized":False,"gpu_used":False,"scientific_exposure":False,"validation_or_heldout_opened":False,"training_or_update":False,"t0_full_authorized":False,"retry_requires_new_freeze":True}

def test_guard_only_normalized_ast_change() -> None:
    baseline=subprocess.check_output(["git","show","67b21d2ccd7cc3189154f67a80fe8db6abd3ec7b:scripts/latent/run_h_iter_phase1_t0_v1.py"],cwd=ROOT).decode()
    repaired=RUNNER.read_text()
    baseline_hash,baseline_nodes=normalized_source(baseline)
    repaired_hash,repaired_nodes=normalized_source(repaired)
    assert baseline_nodes==repaired_nodes
    assert baseline_hash==repaired_hash
    forbidden=("-".join(("validation","bank.json")),"-".join(("heldout","bank.json")))
    assert all(name not in repaired for name in forbidden)

def test_exact_five_guard_cases() -> None:
    source=RUNNER.read_text(); rows,_=case_results(source)
    assert [row["name"] for row in rows]==GUARD_CASE_NAMES
    assert [row["expected_outcome"] for row in rows]==["pass","reject","reject","reject","reject"]
    assert all(row["qualifies"] for row in rows)
    assert rows[0]["observed_error_type"] is None
    assert all(row["observed_error_type"]=="RuntimeError" for row in rows[1:])

def test_direct_guard_rejects_semantic_mutations() -> None:
    guard,_=load_guard(RUNNER.read_text())
    guard(RUNNER.read_text())
    validation_name="-".join(("validation","bank.json")); heldout_name="-".join(("heldout","bank.json"))
    mutations=["\nobject().generate()\n","\nobject().save_pretrained('x')\n",f"\nX={validation_name!r}\n",f"\nX={'/tmp/'+heldout_name!r}\n"]
    for mutation in mutations:
        with pytest.raises(RuntimeError,match="T0 static exposure guard differs"):
            guard(RUNNER.read_text()+mutation)

def test_guard_proof_plan_has_exact_46_assets() -> None:
    assert len(GUARD_PROOF_ASSETS)==len(set(GUARD_PROOF_ASSETS))==46
    value=guard_plan_value(ROOT,"0"*40)
    validate_guard_plan(value,repo=ROOT)
    assert len(value["asset_sha256"])==46
    assert value["execution_authorization"]=={"guard_proof_eligible_after_independent_review":True,"t0_preflight_authorized":False,"t0_full_authorized":False,"model_authorized":False,"gpu_authorized":False,"training_authorized":False}
