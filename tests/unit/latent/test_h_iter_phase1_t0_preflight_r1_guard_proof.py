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
    asset_entries, atomic_terminal, canonical_json, case_results, finalize_terminal,
    load_guard, normalized_source, sha, validate_failed_start, validate_failure,
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
    timing={"outer_seconds":720,"startup_seconds":60,"compute_seconds":300,"compute_alarm_seconds":299,"audit_seconds":120,"audit_alarm_seconds":119,"failure_seconds":90,"failure_alarm_seconds":89,"terminal_seconds":30,"terminal_alarm_seconds":29,"postexit_seconds":30,"reserve_seconds":90,"success_terminal_entry_max_seconds":480,"compute_failure_terminal_entry_max_seconds":450,"audit_failure_terminal_entry_max_seconds":570,"prior_terminal_failure_entry_max_seconds":600,"compute_enter_ns":0,"compute_exit_ns":1,"compute_duration_ns":1,"audit_enter_ns":None,"audit_exit_ns":None,"audit_duration_ns":None,"prior_terminal_enter_ns":None,"prior_terminal_failure_ns":None,"prior_terminal_duration_ns":None,"failure_enter_ns":1,"failure_exit_ns":2,"failure_duration_ns":1,"terminal_enter_ns":2,"prepublication_elapsed_ns":2}
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

def _late_failure(stage:str,memory_count:int,*,with_source:bool,with_cases:bool,with_safety:bool,with_freeze:bool,prior_terminal:bool=False) -> dict:
    value={key:None for key in PROOF_KEYS if key!="proof_sha256"}; entries=asset_entries(ROOT); entry_hash=sha(canonical_json(entries))
    runtime={"python":"3.12.14","sys_executable":"/home/ubuntu/rlm/prime-rl/.venv/bin/python3","sys_prefix":"/home/ubuntu/rlm/prime-rl/.venv","torch":"2.11.0+cu128","transformers":"5.6.2","tokenizers":"0.22.2","flash_linear_attention":"0.5.2","shared_project_pyproject_sha256":"504907808f992f1e6883f54c2695a4814ae77d6b80814239cbfc98d81a543656","shared_project_uv_lock_sha256":"fca5fa6183345b5b68974078c38d58e0320f79eef13a695af11ceab12fdf36d5","cuda_visible_devices":""}
    baseline=subprocess.check_output(["git","show","67b21d2:scripts/latent/run_h_iter_phase1_t0_v1.py"],cwd=ROOT).decode(); repaired=RUNNER.read_text(); bh,bnodes=normalized_source(baseline); rh,rnodes=normalized_source(repaired); _,guard_hash=load_guard(repaired)
    source={"baseline_runner_file_sha256":sha(baseline.encode()),"repaired_runner_file_sha256":sha(repaired.encode()),"baseline_normalized_ast_sha256":bh,"repaired_normalized_ast_sha256":rh,"normalized_ast_equal":bnodes==rnodes,"guard_function_ast_sha256":guard_hash,"added_top_level_functions":["validate_static_exposure_source"],"full_forbidden_literals_absent_from_repaired_source":True,"only_allowed_surface_changed":bnodes==rnodes}
    cases=case_results(repaired)[0]
    safety={"cuda_visible_devices":"","cuda_initialized":False,"torch_imported":False,"transformers_imported":False,"tokenizer_loaded":False,"model_loaded":False,"optimizer_constructed":False,"scientific_forwards":0,"training_operations":0,"train_opens":0,"validation_opens":0,"heldout_opens":0,"network_attempts":0,"output_namespace_fresh_before":True,"object_census_errors":0,"object_census_uninspectable":0}
    timing={"outer_seconds":720,"startup_seconds":60,"compute_seconds":300,"compute_alarm_seconds":299,"audit_seconds":120,"audit_alarm_seconds":119,"failure_seconds":90,"failure_alarm_seconds":89,"terminal_seconds":30,"terminal_alarm_seconds":29,"postexit_seconds":30,"reserve_seconds":90,"success_terminal_entry_max_seconds":480,"compute_failure_terminal_entry_max_seconds":450,"audit_failure_terminal_entry_max_seconds":570,"prior_terminal_failure_entry_max_seconds":600,"compute_enter_ns":0,"compute_exit_ns":1,"compute_duration_ns":1,"audit_enter_ns":1 if stage in {"audit","terminal_publication"} else None,"audit_exit_ns":2 if stage in {"audit","terminal_publication"} else None,"audit_duration_ns":1 if stage in {"audit","terminal_publication"} else None,"prior_terminal_enter_ns":2 if prior_terminal else None,"prior_terminal_failure_ns":3 if prior_terminal else None,"prior_terminal_duration_ns":1 if prior_terminal else None,"failure_enter_ns":3 if prior_terminal else (2 if stage=="audit" else 1),"failure_exit_ns":4 if prior_terminal else (3 if stage=="audit" else 2),"failure_duration_ns":1,"terminal_enter_ns":4 if prior_terminal else (3 if stage=="audit" else 2),"prepublication_elapsed_ns":4 if prior_terminal else (3 if stage=="audit" else 2)}
    rows=[{"index":i,"label":MEMORY_LABELS[i],"rss_bytes":1} for i in range(memory_count)]
    audit={"target_count":46,"pre_entries":entries,"pre_sha256":entry_hash,"post_entries":entries if with_freeze else None,"post_sha256":entry_hash if with_freeze else None,"all_exact":True if with_freeze else None}
    freeze={"head_before":"0"*40,"head_after":"0"*40,"tree_before":"2"*40,"tree_after":"2"*40,"clean_before":True,"clean_after":True,"assets_pre_sha256":entry_hash,"assets_post_sha256":entry_hash,"complete":True} if with_freeze else None
    status="infrastructure_invalid" if stage=="terminal_publication" else "h_iter_phase1_t0_preflight_r1_guard_proof_incomplete"; error_type="InfrastructureInvalid" if status=="infrastructure_invalid" else "T0PreflightR1GuardProofError"
    value.update({"schema_version":FAILURE_SCHEMA,"status":status,"mechanism":"q35-2b-h-iter-phase1-t0-preflight-r1-guard-v1","run_identity":"h-iter-phase1-t0-preflight-r1-guard-proof-run1","execution_commit":"0"*40,"mechanism_code_commit":"1"*40,"tree_sha256":"2"*40,"plan_file_sha256":"3"*64,"plan_sha256":"4"*64,"runtime":runtime,"asset_audit":audit,"failed_start_binding":FAILED_START_BINDING,"source_evidence":source if with_source else None,"case_results":cases if with_cases else None,"safety":safety if with_safety else None,"resources":{"minimum_ram_gib":8,"minimum_disk_gib":8,"maximum_artifact_bytes":1048576,"timing":timing,"observed_rss_peak_bytes":1,"artifact_bytes":0},"memory":{"expected_labels":MEMORY_LABELS,"rows":rows,"label_sha256":sha(canonical_json(MEMORY_LABELS)),"complete":memory_count==14},"full_freeze":freeze,"decision_boundary":PROOF_DECISION,"error_type":error_type,"error_message":"fixture","traceback":"fixture","stage":stage,"execution_progress":{"stage":stage,"memory_rows_completed":memory_count},"audit_errors":["fixture"] if status=="infrastructure_invalid" else [],"failure_sha256":""})
    return finalize_terminal(value,"failure_sha256")

def test_truthful_repaired_cases_audit_and_terminal_failures_validate() -> None:
    repaired=_late_failure("repaired_source",7,with_source=False,with_cases=False,with_safety=False,with_freeze=False); validate_failure(repaired)
    cases=_late_failure("cases",7,with_source=True,with_cases=False,with_safety=False,with_freeze=False); validate_failure(cases)
    partial_cases=_late_failure("cases",9,with_source=True,with_cases=True,with_safety=False,with_freeze=False); partial_cases["case_results"]=partial_cases["case_results"][:2]; partial_cases=finalize_terminal(partial_cases,"failure_sha256"); validate_failure(partial_cases,ROOT)
    audit=_late_failure("audit",13,with_source=True,with_cases=True,with_safety=True,with_freeze=False); validate_failure(audit)
    audit_prewrite_ready=_late_failure("audit",13,with_source=True,with_cases=True,with_safety=True,with_freeze=True); validate_failure(audit_prewrite_ready,ROOT)
    audit_complete=_late_failure("audit",14,with_source=True,with_cases=True,with_safety=True,with_freeze=True); validate_failure(audit_complete,ROOT)
    audit_missing_freeze=copy.deepcopy(audit_complete); audit_missing_freeze["full_freeze"]=None; audit_missing_freeze=finalize_terminal(audit_missing_freeze,"failure_sha256")
    with pytest.raises(RuntimeError): validate_failure(audit_missing_freeze,ROOT)
    terminal=_late_failure("terminal_publication",14,with_source=True,with_cases=True,with_safety=True,with_freeze=True,prior_terminal=True); validate_failure(terminal)
    for field in ("prior_terminal_failure_ns","prior_terminal_duration_ns"):
        changed=copy.deepcopy(terminal); changed["resources"]["timing"][field]+=1; changed["failure_sha256"]=sha(canonical_json({key:item for key,item in changed.items() if key!="failure_sha256"}))
        with pytest.raises(RuntimeError): validate_failure(changed)
    for field in ("prior_terminal_failure_ns","prior_terminal_duration_ns"):
        changed=copy.deepcopy(audit); changed["resources"]["timing"][field]=1; changed=finalize_terminal(changed,"failure_sha256")
        with pytest.raises(RuntimeError): validate_failure(changed)
    partial_compute=copy.deepcopy(audit); partial_compute["resources"]["timing"]["audit_enter_ns"]=None; partial_compute["resources"]["timing"]["audit_duration_ns"]=None; partial_compute=finalize_terminal(partial_compute,"failure_sha256")
    with pytest.raises(RuntimeError): validate_failure(partial_compute)
    partial_terminal=copy.deepcopy(terminal); partial_terminal["resources"]["timing"]["prior_terminal_enter_ns"]=None; partial_terminal["resources"]["timing"]["prior_terminal_failure_ns"]=None; partial_terminal["resources"]["timing"]["prior_terminal_duration_ns"]=None; partial_terminal=finalize_terminal(partial_terminal,"failure_sha256")
    with pytest.raises(RuntimeError): validate_failure(partial_terminal)

def test_source_and_case_coordinated_tampers_rejected() -> None:
    value=_late_failure("audit",13,with_source=True,with_cases=True,with_safety=True,with_freeze=False); validate_failure(value,ROOT)
    mutations=[]
    for field in ("baseline_normalized_ast_sha256","guard_function_ast_sha256","full_forbidden_literals_absent_from_repaired_source","only_allowed_surface_changed"):
        changed=copy.deepcopy(value); changed["source_evidence"][field]=False if isinstance(changed["source_evidence"][field],bool) else "0"*64; mutations.append(changed)
    for field in ("expected_outcome","observed_outcome","observed_error_type","qualifies","mutated_source_sha256"):
        changed=copy.deepcopy(value); changed["case_results"][0][field]="reject" if field in {"expected_outcome","observed_outcome"} else ("RuntimeError" if field=="observed_error_type" else (False if field=="qualifies" else "0"*64)); mutations.append(changed)
    for changed in mutations:
        changed["failure_sha256"]=sha(canonical_json({key:item for key,item in changed.items() if key!="failure_sha256"}))
        with pytest.raises(RuntimeError): validate_failure(changed,ROOT)
