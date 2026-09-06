from __future__ import annotations

import subprocess
import sys
import copy
import tempfile
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"scripts/latent"))

from freeze_h_iter_phase1_t0_plan_v1 import (
    FAILED_START_BINDING, GUARD_CASE_NAMES, GUARD_PROOF_ASSETS,
    guard_plan_value, validate_guard_plan,
)
from run_h_iter_phase1_t0_preflight_r1_guard_proof_v1 import (
    FAILURE_SCHEMA, FAILURE_NAME, MEMORY_LABELS, PROOF_DECISION, PROOF_KEYS,
    atomic_terminal, canonical_json, case_results, finalize_terminal, load_guard,
    normalized_source, sha, validate_failed_start, validate_failure,
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
    assert value["execution_authorization"]=={"guard_proof_eligible_after_independent_review":True,"t0_preflight_authorized":False,"t0_full_authorized":False,"model_load_authorized":False,"gpu_authorized":False,"training_authorized":False}

def _runtime_failure() -> dict:
    value={key:None for key in PROOF_KEYS if key!="proof_sha256"}
    timing={"outer_seconds":720,"startup_seconds":60,"compute_seconds":300,"compute_alarm_seconds":299,"audit_seconds":120,"audit_alarm_seconds":119,"failure_seconds":90,"failure_alarm_seconds":89,"terminal_seconds":30,"terminal_alarm_seconds":29,"postexit_seconds":30,"reserve_seconds":90,"success_terminal_entry_max_seconds":480,"compute_failure_terminal_entry_max_seconds":450,"audit_failure_terminal_entry_max_seconds":570,"prior_terminal_failure_entry_max_seconds":600,"compute_enter_ns":0,"compute_exit_ns":1,"compute_duration_ns":1,"audit_enter_ns":None,"audit_exit_ns":None,"audit_duration_ns":None,"failure_enter_ns":1,"failure_exit_ns":2,"failure_duration_ns":1,"terminal_enter_ns":2,"prepublication_elapsed_ns":2}
    value.update({"schema_version":FAILURE_SCHEMA,"status":"infrastructure_invalid","mechanism":"q35-2b-h-iter-phase1-t0-preflight-r1-guard-v1","run_identity":"h-iter-phase1-t0-preflight-r1-guard-proof-run1","execution_commit":"0"*40,"plan_file_sha256":"1"*64,"resources":{"minimum_ram_gib":8,"minimum_disk_gib":8,"maximum_artifact_bytes":1048576,"timing":timing,"observed_rss_peak_bytes":1,"artifact_bytes":0},"memory":{"expected_labels":MEMORY_LABELS,"rows":[{"index":0,"label":"ENTRY","rss_bytes":1}],"label_sha256":sha(canonical_json(MEMORY_LABELS)),"complete":False},"decision_boundary":PROOF_DECISION,"error_type":"InfrastructureInvalid","error_message":"fixture","traceback":"fixture","stage":"runtime","execution_progress":{"stage":"runtime","memory_rows_completed":1},"audit_errors":["fixture"],"failure_sha256":""})
    return finalize_terminal(value,"failure_sha256")

def test_failure_stage_authority_and_artifact_tampers_rejected() -> None:
    value=_runtime_failure(); validate_failure(value)
    mutations=[]
    future=copy.deepcopy(value); future["source_evidence"]={}; mutations.append(future)
    wrong_memory=copy.deepcopy(value); wrong_memory["memory"]["rows"][0]["label"]="RUNTIME_VERIFIED"; mutations.append(wrong_memory)
    wrong_authority=copy.deepcopy(value); wrong_authority["execution_commit"]="short"; mutations.append(wrong_authority)
    wrong_size=copy.deepcopy(value); wrong_size["resources"]["artifact_bytes"]+=1; mutations.append(wrong_size)
    for item in mutations:
        item["failure_sha256"]=sha(canonical_json({key:entry for key,entry in item.items() if key!="failure_sha256"}))
        with pytest.raises(RuntimeError): validate_failure(item)

def test_atomic_failure_replaces_only_owned_temporary_file() -> None:
    value=_runtime_failure()
    with tempfile.TemporaryDirectory() as td:
        output=Path(td)/"terminal"; output.mkdir(); (output/("."+FAILURE_NAME+".tmp")).write_bytes(b"stale")
        atomic_terminal(output,FAILURE_NAME,value,validate_failure)
        assert [path.name for path in output.iterdir()]==[FAILURE_NAME]
        assert (output/FAILURE_NAME).read_bytes()==canonical_json(value)+b"\n"
