#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from prime_rl.latent.h_iter_phase1_t0 import (
    ACTIONS, ARMS, ARTIFACT_DIR, CAP0_BINDING, CANDIDATE_FILES,
    E33_PATH, E33_STATE_SHA256, E33_TREE_SHA256, H176_PATH,
    H176_TREE_SHA256, MECHANISM, METADATA_SHA256, MF0_BINDING,
    MODEL_FREE_EVIDENCE_BINDING,
    OUTPUT_ROOT, PLAN_SCHEMA, PROOF_OUTPUT_ROOT, PROOF_RUN_ID, RUN_ID,
    TRAIN_BANK_FILE_SHA256, TRAIN_BANK_INTERNAL_SHA256, TRAIN_BANK_PATH,
    build_antecedent_manifest, build_capture_schedule, build_contract,
    build_memory_schedule, build_tamper_schedule, canonical_json,
    sha256_bytes, strict_loads,
    validate_model_free_evidence_binding, validate_plan,
)

MF0_DIR="experiments/qwen35-2b-latent-workspace-v1/h-iter-phase1-train-calibration-v1"
CAP0_DIR="experiments/qwen35-2b-latent-workspace-v1/h-iter-phase1-cap0-r1-v1"
T0_DIR=ARTIFACT_DIR
PROOF_ASSETS=[
 TRAIN_BANK_PATH,
 f"{MF0_DIR}/candidate-module-contract.json",f"{MF0_DIR}/capture-contract.json",f"{MF0_DIR}/metric-gate-contract.json",f"{MF0_DIR}/threshold-builder-contract.json",f"{MF0_DIR}/train-partition.json",f"{MF0_DIR}/training-schedule.json",f"{MF0_DIR}/phase0-evidence-binding.json",f"{MF0_DIR}/mf0-prereg-run1-evidence-manifest.json",f"{MF0_DIR}/mf0-prereg-run1.MF0-PROOF.json",f"{MF0_DIR}/mf0-prereg-run1.launcher.log",f"{MF0_DIR}/mf0-prereg-run1.exit.txt",
 f"{CAP0_DIR}/cap0-r1-run1-evidence-manifest.json",f"{CAP0_DIR}/cap0-r1-run1.CAP0-R1-PROOF.json",f"{CAP0_DIR}/cap0-r1-run1.launcher.log",f"{CAP0_DIR}/cap0-r1-run1.exit.txt",
 f"{T0_DIR}/t0-contract.json",f"{T0_DIR}/t0-capture-schedule.json",f"{T0_DIR}/t0-memory-schedule.json",f"{T0_DIR}/t0-tamper-schedule.json",f"{T0_DIR}/t0-antecedent-evidence-manifest.json",
 "pyproject.toml","uv.lock","src/prime_rl/latent/h_iter_phase1_t0.py","scripts/latent/run_h_iter_phase1_t0_v1.py","scripts/latent/run_h_iter_phase1_t0_v1.sh","scripts/latent/freeze_h_iter_phase1_t0_plan_v1.py","tests/unit/latent/test_h_iter_phase1_t0.py","scripts/latent/run_h_iter_phase1_t0_model_free_proof_v1.py","scripts/latent/run_h_iter_phase1_t0_model_free_proof_v1.sh","tests/unit/latent/test_h_iter_phase1_t0_model_free_proof.py",
]
if len(PROOF_ASSETS)!=31 or len(set(PROOF_ASSETS))!=31: raise RuntimeError("T0 proof asset set differs")
FINAL_ASSETS=[*PROOF_ASSETS,
 f"{T0_DIR}/t0-model-free-proof-plan.json",f"{T0_DIR}/t0-model-free-proof-plan.sha256",
 MODEL_FREE_EVIDENCE_BINDING["evidence_manifest_path"],MODEL_FREE_EVIDENCE_BINDING["proof_path"],MODEL_FREE_EVIDENCE_BINDING["launcher_log_path"],MODEL_FREE_EVIDENCE_BINDING["exit_path"],
]
if len(FINAL_ASSETS)!=37 or len(set(FINAL_ASSETS))!=37: raise RuntimeError("T0 final asset set differs")

GUARD_MECHANISM="q35-2b-h-iter-phase1-t0-preflight-r1-guard-v1"
GUARD_RUN_ID="h-iter-phase1-t0-preflight-r1-guard-proof-run2"
GUARD_PLAN_SCHEMA="prime-rl/latent-h-iter-phase1-t0-preflight-r1-guard-plan/v2"
GUARD_DIR=T0_DIR
FAILED_MANIFEST=f"{T0_DIR}/t0-preflight-run1-failed-evidence-manifest.json"
FAILED_ASSETS=[
 f"{T0_DIR}/t0-preflight-run1-failed.plan.json",
 f"{T0_DIR}/t0-preflight-run1-failed.plan.sha256",
 f"{T0_DIR}/t0-preflight-run1-failed.stdout",
 f"{T0_DIR}/t0-preflight-run1-failed.stderr",
 f"{T0_DIR}/t0-preflight-run1-failed.exit.txt",
 FAILED_MANIFEST,
]
GUARD_CODE_ASSETS=[
 "scripts/latent/run_h_iter_phase1_t0_preflight_r1_guard_proof_v1.py",
 "scripts/latent/run_h_iter_phase1_t0_preflight_r1_guard_proof_v1.sh",
 "tests/unit/latent/test_h_iter_phase1_t0_preflight_r1_guard_proof.py",
]
RUN1_EVIDENCE_MANIFEST=f"{T0_DIR}/t0-preflight-r1-guard-proof-run1-invalid-evidence-manifest.json"
RUN1_EVIDENCE_ASSETS=[
 RUN1_EVIDENCE_MANIFEST,
 f"{T0_DIR}/t0-preflight-r1-guard-proof-run1-invalid.T0-PREFLIGHT-R1-GUARD-FAILURE.json",
 f"{T0_DIR}/t0-preflight-r1-guard-proof-run1-invalid.exit.txt",
 f"{T0_DIR}/t0-preflight-r1-guard-proof-run1-invalid.launcher.log",
 f"{T0_DIR}/t0-preflight-r1-guard-proof-run1-invalid.plan.json",
 f"{T0_DIR}/t0-preflight-r1-guard-proof-run1-invalid.plan.sha256",
]
DIAGNOSTIC_MANIFEST=f"{T0_DIR}/t0-preflight-r1-guard-runtime-diagnostic-run1-evidence-manifest.json"
DIAGNOSTIC_ASSETS=[
 DIAGNOSTIC_MANIFEST,
 f"{T0_DIR}/t0-preflight-r1-guard-runtime-diagnostic-run1.command.txt",
 f"{T0_DIR}/t0-preflight-r1-guard-runtime-diagnostic-run1.exit.txt",
 f"{T0_DIR}/t0-preflight-r1-guard-runtime-diagnostic-run1.stderr",
 f"{T0_DIR}/t0-preflight-r1-guard-runtime-diagnostic-run1.stdout",
]
GUARD_PROOF_ASSETS=[*FINAL_ASSETS,*FAILED_ASSETS,*GUARD_CODE_ASSETS,*RUN1_EVIDENCE_ASSETS,*DIAGNOSTIC_ASSETS]
if len(GUARD_PROOF_ASSETS)!=57 or len(set(GUARD_PROOF_ASSETS))!=57: raise RuntimeError("T0 guard proof asset set differs")

FAILED_START_BINDING={
 "failed_execution_commit":"67b21d2ccd7cc3189154f67a80fe8db6abd3ec7b",
 "failed_mechanism_code_commit":"6d43c0175256f00c61bb08e26e61ff6112489911",
 "failed_tree_sha256":"7670bd01e51c16f27d5eaea6fbd37826d862eecd",
 "failed_plan_path":f"{T0_DIR}/t0-preflight-run1-failed.plan.json",
 "failed_plan_file_sha256":"3a872893adc44e770fcf8dd7784ac348107cc36b6b52f560dfe8ea467920ff85",
 "failed_plan_sha256":"8be654db553fc986cec36b6b6ca73516d18a6104fe9f86eb3e626a242352cf6a",
 "failed_plan_sidecar_path":f"{T0_DIR}/t0-preflight-run1-failed.plan.sha256",
 "failed_plan_sidecar_file_sha256":"024021a84f3f0caa15f31ea22668ab180eedcd86ed18c53b68c0c27f9cc0c87f",
 "evidence_commit":"e9193e94e58a1297ab1fa31fbe465848459b1470", "archive_freeze_commit":"cc46d9b0ac21dc0121480a49d77da1e466f400d5",
 "evidence_manifest_path":FAILED_MANIFEST,
 "evidence_manifest_file_sha256":"4a8cf5d418a67c05e4b999beed417dd087846fbd01fbfee5e8410410b60a92b5",
 "evidence_manifest_internal_sha256":"311d19834014c44f807777e9488a2cd9842a2cdc7b99c5456b7191c7ed6065e1",
 "stdout_path":f"{T0_DIR}/t0-preflight-run1-failed.stdout","stdout_file_sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","stdout_bytes":0,
 "stderr_path":f"{T0_DIR}/t0-preflight-run1-failed.stderr","stderr_file_sha256":"a73f3021da405f37db6bc0c4c2aed0d0f365515fbd20364f10360d1f2104ee08","stderr_bytes":1244,
 "exit_path":f"{T0_DIR}/t0-preflight-run1-failed.exit.txt","exit_file_sha256":"4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865","exit_bytes":2,
 "classification":"infrastructure_invalid",
}
FAILED_RUN1_ARCHIVE_BINDING={
 "archive_freeze_commit":"9cc45bd2a53a9add4e9ff5096d080093ad58a753","classification":"infrastructure_invalid","evidence_commit":"ce9a1ef5c5fe23b7ba5c56c4eae8c39a4e0afc01",
 "evidence_manifest_file_sha256":"ba4535b4e4bedaa8fb414cf39c63b149aa390009bb1ad13a10a503a07afb66ef","evidence_manifest_internal_sha256":"12cc0c159bc3bdc44a3117a946de6765a2c2fe27d31e4a368bd32982000a5d98","evidence_manifest_path":RUN1_EVIDENCE_MANIFEST,
 "failed_execution_commit":"e5337a086c62770c6386fa1ac5d77e41c2b00cfa","failed_mechanism_code_commit":"df503f8e8a84ca802e73b2f1606f2b7836f6181e","failed_plan_file_sha256":"ef5bcaa1dc2df46ebe0c39d255e4bc6b2fa236d851f3e15e34c30c8acf3baf4e","failed_plan_sha256":"6af359f9ad95c7bad9568a8071ccb574cd1da0d72b9b4de04c75834baa455b7b","failed_tree_sha256":"3808611ebc9314dce653a0ac35f9ceea1ba5be91","run_identity":"h-iter-phase1-t0-preflight-r1-guard-proof-run1","scientific_exposure":False,"terminal_failure_file_sha256":"7aa83d103838e7ccce55b9151b6cf1ffef1f597d0b8038cecf574f297202b836","terminal_failure_sha256":"dfdd277258bce8ad85feb87352a64e5b1b086c34089130238c973ebf9b15e1ab",
}
DIAGNOSTIC_ARCHIVE_BINDING={
 "archive_freeze_commit":"d755863d7ff6363e3e9c2ce7c6576625976d929c","diagnostic_identity":"h-iter-phase1-t0-preflight-r1-guard-runtime-diagnostic-run1","evidence_commit":"8c94a3d1068403a67283bc6d29490e117cb4fb9f","evidence_manifest_file_sha256":"c7c19b45a95a0e56d2d39af3912033ea128c2cb83c38c294d395b9c8444e0ccc","evidence_manifest_internal_sha256":"3f77523cc02f6a69a1c6a556da09efce51f6c7c2cc23e61162d4c4d39def3d06","evidence_manifest_path":DIAGNOSTIC_MANIFEST,"isolated_field":"sys_executable","observed_lexical":"/home/ubuntu/rlm/prime-rl/.venv/bin/python3","observed_resolved":"/home/ubuntu/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12","posthoc_transcript_reconstruction":True,"scientific_exposure":False,"target_attempt_identity":"h-iter-phase1-t0-preflight-r1-guard-proof-run1","transcript_call_id":"call_zhtdVrky5sq0V9mhrbHZdksD","transcript_output_timestamp":"2026-09-06T19:36:47.672Z",
}
GUARD_CASE_NAMES=["production_source_accepts","generate_call_rejected","save_pretrained_call_rejected","validation_bank_literal_rejected","heldout_bank_literal_rejected"]
GUARD_DECISION_DESIGN={"claim":"preflight_guard_robustness_proof_design_only","independent_scientific_replicate":False,"t0_preflight_authorized":False,"t0_full_authorized":False,"model_or_gpu_authorized":False,"scientific_exposure_authorized":False,"validation_or_heldout_opened":False,"training_authorized":False,"admission":False,"nomination":False,"promotion":False,"four_live_floor_unchanged":True}

def read_json(path:Path)->dict:
    return strict_loads(path.read_bytes())

def write_atomic(path:Path,payload:dict)->None:
    write_bytes_atomic(path,canonical_json(payload)+b"\n")

def write_bytes_atomic(path:Path,data:bytes)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        with os.fdopen(fd,"wb") as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp,path)
        dfd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
    except BaseException:
        try: tmp.unlink()
        except FileNotFoundError: pass
        raise

def materialize(repo:Path)->None:
    bank=read_json(repo/TRAIN_BANK_PATH)
    base=repo/MF0_DIR
    candidate=read_json(base/"candidate-module-contract.json")
    capture_source=read_json(base/"capture-contract.json")
    metric=read_json(base/"metric-gate-contract.json")
    schedule=read_json(base/"training-schedule.json")
    outputs={
      "t0-antecedent-evidence-manifest.json":build_antecedent_manifest(),
      "t0-capture-schedule.json":build_capture_schedule(bank),
      "t0-tamper-schedule.json":build_tamper_schedule(),
      "t0-contract.json":build_contract(candidate,capture_source,metric,schedule),
    }
    outputs["t0-memory-schedule.json"]=build_memory_schedule(outputs["t0-capture-schedule.json"],schedule)
    for name,value in outputs.items(): write_atomic(repo/T0_DIR/name,value)

def validate_guard_archives(repo:Path)->None:
    def file_sha(relative:str)->str:
        path=repo/relative
        if path.is_symlink() or not path.is_file(): raise RuntimeError("T0 guard archive path differs")
        return hashlib.sha256(path.read_bytes()).hexdigest()
    run1_rows=[
      (f"{T0_DIR}/t0-preflight-r1-guard-proof-run1-invalid.plan.json","ef5bcaa1dc2df46ebe0c39d255e4bc6b2fa236d851f3e15e34c30c8acf3baf4e",13797),
      (f"{T0_DIR}/t0-preflight-r1-guard-proof-run1-invalid.plan.sha256","17cdbf97efd6fffe391ec4b3f04005d8c9dae9b2be0443426f9103256a81318b",65),
      (f"{T0_DIR}/t0-preflight-r1-guard-proof-run1-invalid.T0-PREFLIGHT-R1-GUARD-FAILURE.json","7aa83d103838e7ccce55b9151b6cf1ffef1f597d0b8038cecf574f297202b836",4280),
      (f"{T0_DIR}/t0-preflight-r1-guard-proof-run1-invalid.launcher.log","e8e9020c12362ad635a9f24ad501e9193ddd47450abcc6a85811443cd8fd5357",1402),
      (f"{T0_DIR}/t0-preflight-r1-guard-proof-run1-invalid.exit.txt","53c234e5e8472b6ac51c1ae1cab3fe06fad053beb8ebfd8977b010655bfdd3c3",2),
    ]
    diagnostic_rows=[
      (f"{T0_DIR}/t0-preflight-r1-guard-runtime-diagnostic-run1.command.txt","444236dfe05ee73b81d366c9d14444a8a26b2605962b112202b7f6cd3f693083",1671),
      (f"{T0_DIR}/t0-preflight-r1-guard-runtime-diagnostic-run1.stdout","8901eaa3a02ff8bfd02330d4cdb45cf3e7ec655cd07eeec2058abc0778f1f3fe",1486),
      (f"{T0_DIR}/t0-preflight-r1-guard-runtime-diagnostic-run1.stderr","e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",0),
      (f"{T0_DIR}/t0-preflight-r1-guard-runtime-diagnostic-run1.exit.txt","9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",2),
    ]
    for relative,digest,size in [*run1_rows,*diagnostic_rows]:
        if file_sha(relative)!=digest or (repo/relative).stat().st_size!=size: raise RuntimeError("T0 guard archive asset differs")
    run1=read_json(repo/RUN1_EVIDENCE_MANIFEST); diagnostic=read_json(repo/DIAGNOSTIC_MANIFEST)
    if file_sha(RUN1_EVIDENCE_MANIFEST)!=FAILED_RUN1_ARCHIVE_BINDING["evidence_manifest_file_sha256"] or run1.get("manifest_sha256")!=FAILED_RUN1_ARCHIVE_BINDING["evidence_manifest_internal_sha256"] or run1["manifest_sha256"]!=sha256_bytes(canonical_json({k:v for k,v in run1.items() if k!="manifest_sha256"})): raise RuntimeError("T0 guard run1 manifest differs")
    if set(run1)!={"schema_version","status","mechanism","run_identity","execution_commit","mechanism_code_commit","tree_sha256","plan_file_sha256","plan_sha256","failure_evidence","claim_boundary","ordered_evidence_assets","manifest_sha256"} or run1["schema_version"]!="prime-rl/latent-h-iter-phase1-t0-preflight-r1-guard-failed-run-evidence-manifest/v1" or run1["status"]!="h_iter_phase1_t0_preflight_r1_guard_failed_run_archived" or run1["failure_evidence"]["terminal_failure_sha256"]!=FAILED_RUN1_ARCHIVE_BINDING["terminal_failure_sha256"] or run1["claim_boundary"]["scientific_exposure"] is not False: raise RuntimeError("T0 guard run1 manifest semantics differ")
    if file_sha(DIAGNOSTIC_MANIFEST)!=DIAGNOSTIC_ARCHIVE_BINDING["evidence_manifest_file_sha256"] or diagnostic.get("manifest_sha256")!=DIAGNOSTIC_ARCHIVE_BINDING["evidence_manifest_internal_sha256"] or diagnostic["manifest_sha256"]!=sha256_bytes(canonical_json({k:v for k,v in diagnostic.items() if k!="manifest_sha256"})): raise RuntimeError("T0 guard diagnostic manifest differs")
    if set(diagnostic)!={"schema_version","status","mechanism","diagnostic_identity","target_attempt_identity","host_paths","runtime_expected","runtime_observed","ordered_differences","safety","transcript_source","ordered_raw_assets","claim_boundary","manifest_sha256"} or diagnostic["schema_version"]!="prime-rl/latent-h-iter-phase1-t0-preflight-r1-guard-runtime-diagnostic-evidence-manifest/v1" or diagnostic["status"]!="h_iter_phase1_t0_preflight_r1_guard_runtime_diagnostic_archived" or diagnostic["ordered_differences"]!= [{"order":1,"field":"sys_executable","expected":DIAGNOSTIC_ARCHIVE_BINDING["observed_lexical"],"observed":DIAGNOSTIC_ARCHIVE_BINDING["observed_resolved"],"interpretation":"lexical_virtualenv_entrypoint_resolves_to_uv_managed_interpreter"}] or diagnostic["transcript_source"]["call_id"]!=DIAGNOSTIC_ARCHIVE_BINDING["transcript_call_id"] or diagnostic["claim_boundary"]["scientific_exposure"] is not False: raise RuntimeError("T0 guard diagnostic semantics differ")
    git=lambda *args:subprocess.check_output(["git",*args],cwd=repo,text=True).strip()
    e1=FAILED_RUN1_ARCHIVE_BINDING["evidence_commit"]; f1=FAILED_RUN1_ARCHIVE_BINDING["archive_freeze_commit"]; d1=DIAGNOSTIC_ARCHIVE_BINDING["evidence_commit"]; fd1=DIAGNOSTIC_ARCHIVE_BINDING["archive_freeze_commit"]
    if git("rev-parse",f"{e1}^")!=FAILED_RUN1_ARCHIVE_BINDING["failed_execution_commit"] or git("rev-parse",f"{f1}^")!=e1 or git("rev-parse",f"{f1}^{{tree}}")!=git("rev-parse",f"{e1}^{{tree}}") or git("diff-tree","--no-commit-id","--name-only","-r",f1): raise RuntimeError("T0 guard run1 archive ancestry differs")
    if git("rev-parse",f"{d1}^")!=f1 or git("rev-parse",f"{fd1}^")!=d1 or git("rev-parse",f"{fd1}^{{tree}}")!=git("rev-parse",f"{d1}^{{tree}}") or git("diff-tree","--no-commit-id","--name-only","-r",fd1): raise RuntimeError("T0 guard diagnostic archive ancestry differs")

def plan_value(repo:Path,commit:str,proof_input:bool=True)->dict:
    def file_sha(path:str)->str: return hashlib.sha256((repo/path).read_bytes()).hexdigest()
    candidate=file_sha(f"{MF0_DIR}/candidate-module-contract.json")
    capture=file_sha(f"{MF0_DIR}/capture-contract.json")
    metric=file_sha(f"{MF0_DIR}/metric-gate-contract.json")
    threshold=file_sha(f"{MF0_DIR}/threshold-builder-contract.json")
    partition=file_sha(f"{MF0_DIR}/train-partition.json")
    schedule=file_sha(f"{MF0_DIR}/training-schedule.json")
    antecedent_path=f"{T0_DIR}/t0-antecedent-evidence-manifest.json"
    antecedent=read_json(repo/antecedent_path)
    if not proof_input: validate_model_free_evidence_binding(repo,MODEL_FREE_EVIDENCE_BINDING)
    assets={path:file_sha(path) for path in sorted(PROOF_ASSETS if proof_input else FINAL_ASSETS)}
    resource={"minimum_gpu_free_gib":44,"maximum_allocated_or_reserved_gib":40,"minimum_ram_gib":64,"minimum_disk_gib":16,"maximum_output_bytes":33554432,"timing":{"outer":21600,"startup":600,"compute":18000,"audit":1200,"failure":1200,"terminal":300,"postexit":300,"alarm_safety_margin":1,"success_terminal_entry_max":19200,"compute_failure_terminal_entry_max":19200,"audit_failure_terminal_entry_max":20400,"prior_terminal_failure_entry_max":20700}}
    value={"schema_version":PLAN_SCHEMA,"status":"h_iter_phase1_t0_model_free_proof_preregistered" if proof_input else "h_iter_phase1_t0_preregistered","mechanism":MECHANISM,"run_identity":PROOF_RUN_ID if proof_input else RUN_ID,"implementation_commit":commit,"mechanism_code_commit":commit,"plan_sha256":"","asset_sha256":assets,
      "archive_bindings":{"mf0":MF0_BINDING,"cap0_r1":CAP0_BINDING,"antecedent_manifest_path":antecedent_path,"antecedent_manifest_file_sha256":file_sha(antecedent_path),"antecedent_manifest_internal_sha256":antecedent["manifest_sha256"]},
      "remote_paths":{"repo":"/home/ubuntu/rlm/worktrees/q35-2b-latent-workspace-v1","shared_project":"/home/ubuntu/rlm/prime-rl","shared_python":"/home/ubuntu/rlm/prime-rl/.venv/bin/python3","e33":E33_PATH,"h176":H176_PATH,"physical_gpu":"0","visible_device":"cuda:0","proof_output":PROOF_OUTPUT_ROOT,"t0_output":OUTPUT_ROOT},
      "runtime":{"python":"3.12.14","sys_executable":"/home/ubuntu/rlm/prime-rl/.venv/bin/python3","sys_prefix":"/home/ubuntu/rlm/prime-rl/.venv","transformers":"5.6.2","tokenizers":"0.22.2","torch_distribution":"2.11.0+cu128","torch_runtime":"2.11.0+cu128","flash_linear_attention":"0.5.2","gpu_model":"NVIDIA RTX A6000","shared_project_pyproject_sha256":"504907808f992f1e6883f54c2695a4814ae77d6b80814239cbfc98d81a543656","shared_project_uv_lock_sha256":"fca5fa6183345b5b68974078c38d58e0320f79eef13a695af11ceab12fdf36d5"},
      "model_contract":{"e33_path":E33_PATH,"e33_tree_sha256":E33_TREE_SHA256,"e33_state_sha256":E33_STATE_SHA256,"metadata_sha256":METADATA_SHA256,"dtype":"torch.bfloat16","attention":"eager","eval":True,"frozen":True,"h176_path":H176_PATH,"h176_tree_sha256":H176_TREE_SHA256,"h176_loaded":False},
      "data_contract":{"train_bank_path":TRAIN_BANK_PATH,"train_bank_file_sha256":TRAIN_BANK_FILE_SHA256,"train_bank_internal_sha256":TRAIN_BANK_INTERNAL_SHA256,"partition_path":f"{MF0_DIR}/train-partition.json","partition_file_sha256":partition,"schedule_path":f"{MF0_DIR}/training-schedule.json","schedule_file_sha256":schedule,"validation_open_allowed":False,"heldout_open_allowed":False},
      "capture_contract":{"source_contract_path":f"{MF0_DIR}/capture-contract.json","source_contract_file_sha256":capture,"capture_schedule_path":f"{T0_DIR}/t0-capture-schedule.json","capture_schedule_file_sha256":file_sha(f"{T0_DIR}/t0-capture-schedule.json"),"tokenizer_calls":96,"model_forwards":96,"sequences":2304,"cache_checks":194,"memory_only":True},
      "training_contract":{"candidate_contract_path":f"{MF0_DIR}/candidate-module-contract.json","candidate_contract_file_sha256":candidate,"arm_order":ARMS,"action_order":ACTIONS,"operation_count":385,"sidecar_forwards":385,"sidecar_backwards":325,"optimizer_steps":320,"cell_calls":773,"updates_per_arm":64},
      "metric_gate_contract":{"metric_contract_path":f"{MF0_DIR}/metric-gate-contract.json","metric_contract_file_sha256":metric,"threshold_contract_path":f"{MF0_DIR}/threshold-builder-contract.json","threshold_contract_file_sha256":threshold,"gate_count":9,"all_required":True},
      "terminal_contract":{"proof_schema":"prime-rl/latent-h-iter-phase1-t0-proof/v1","failure_schema":"prime-rl/latent-h-iter-phase1-t0-failure/v1","complete_statuses":["h_iter_phase1_t0_validation_contract_design_authorized","h_iter_phase1_train_calibration_stop"],"failure_statuses":["h_iter_phase1_t0_incomplete","h_iter_phase1_t0_capture_mechanism_rejected","h_iter_phase1_t0_exposure_boundary_rejected","infrastructure_invalid"],"proof_filename":"T0-PROOF.json","failure_filename":"T0-FAILURE.json","candidate_filenames":CANDIDATE_FILES,"derived_threshold_filename":"derived-validation-thresholds.json","atomic_publish":True,"reopen_validate":True,"dual_terminal_forbidden":True},
      "resource_bounds":resource,"full_freeze":{"execution_parent_is_evidence_closure":True,"mechanism_is_ancestor":True,"head_exact":True,"tree_exact":True,"clean_before_after":True,"assets_pre_post_exact":True,"protected_pre_post_exact":True,"output_excluded":True},
      "execution_authorization":{"model_free_proof_eligible_after_independent_review":True,"t0_preflight_eligible":False,"t0_full_authorized":False,"model_load_authorized":False,"gpu_authorized":False,"training_authorized":False} if proof_input else {"model_free_proof_eligible_after_independent_review":False,"t0_preflight_eligible_after_independent_review":True,"t0_full_authorized":False,"model_load_authorized":False,"gpu_authorized":False,"training_authorized":False},
      "decision_boundary":{"claim":"model_free_proof_design_only" if proof_input else "model_free_proof_validated_t0_preregistered","train_bank_schema_open_allowed":True,"train_scientific_model_exposure_allowed":False,"validation_or_heldout_opened":False,"model_or_gpu_authorized":False,"t0_training_authorized":False,"admission":False,"nomination":False,"promotion":False,"four_live_floor_unchanged":True},"model_free_proof_binding":None if proof_input else MODEL_FREE_EVIDENCE_BINDING}
    value["plan_sha256"]=sha256_bytes(canonical_json({k:v for k,v in value.items() if k!="plan_sha256"})); validate_plan(value,proof_input=proof_input,repo=None if proof_input else repo)
    return value

def guard_plan_value(repo:Path,commit:str)->dict:
    file_sha=lambda relative:hashlib.sha256((repo/relative).read_bytes()).hexdigest()
    assets={path:file_sha(path) for path in sorted(GUARD_PROOF_ASSETS)}
    value={
      "schema_version":GUARD_PLAN_SCHEMA,"status":"h_iter_phase1_t0_preflight_r1_guard_proof_preregistered","mechanism":GUARD_MECHANISM,"run_identity":GUARD_RUN_ID,
      "implementation_commit":commit,"mechanism_code_commit":commit,"plan_sha256":"","asset_sha256":assets,
      "remote_paths":{"repo":"/home/ubuntu/rlm/worktrees/q35-2b-latent-workspace-v1","shared_project":"/home/ubuntu/rlm/prime-rl","shared_python":"/home/ubuntu/rlm/prime-rl/.venv/bin/python3","proof_output":"/home/ubuntu/rlm/outputs/q35-2b-h-iter-phase1-t0-preflight-r1-guard-proof-run2","t0_output":OUTPUT_ROOT},
      "runtime":{"python":"3.12.14","sys_executable":"/home/ubuntu/rlm/prime-rl/.venv/bin/python3","sys_prefix":"/home/ubuntu/rlm/prime-rl/.venv","torch":"2.11.0+cu128","transformers":"5.6.2","tokenizers":"0.22.2","flash_linear_attention":"0.5.2","shared_project_pyproject_sha256":"504907808f992f1e6883f54c2695a4814ae77d6b80814239cbfc98d81a543656","shared_project_uv_lock_sha256":"fca5fa6183345b5b68974078c38d58e0320f79eef13a695af11ceab12fdf36d5","cuda_visible_devices":""},
      "failed_start_binding":FAILED_START_BINDING,
      "failed_run1_archive_binding":FAILED_RUN1_ARCHIVE_BINDING,"diagnostic_archive_binding":DIAGNOSTIC_ARCHIVE_BINDING,
      "source_contract":{"baseline_execution_commit":"67b21d2ccd7cc3189154f67a80fe8db6abd3ec7b","baseline_mechanism_code_commit":"6d43c0175256f00c61bb08e26e61ff6112489911","baseline_runner_path":"scripts/latent/run_h_iter_phase1_t0_v1.py","baseline_runner_file_sha256":"d6f8fcfb153e81c6aaf5faa42caf1700abba25a0594108ea8db4837034ca6bee","repaired_runner_path":"scripts/latent/run_h_iter_phase1_t0_v1.py","repaired_runner_file_sha256":file_sha("scripts/latent/run_h_iter_phase1_t0_v1.py"),"allowed_change":"add top-level validate_static_exposure_source and replace old guard-if with helper call","ast_canonicalizer":{"dump":"ast.dump","annotate_fields":True,"include_attributes":False},"normalization_rule":"remove helper and normalize old guard-if or repaired helper call to Expr(Constant STATIC_EXPOSURE_GUARD)","forbidden_call_attributes":["generate","save_pretrained"],"forbidden_filenames":["validation-bank.json","heldout-bank.json"],"ordered_case_names":GUARD_CASE_NAMES,"runtime_observer_runner_path":"scripts/latent/run_h_iter_phase1_t0_preflight_r1_guard_proof_v1.py","runtime_observer_baseline_commit":"e5337a086c62770c6386fa1ac5d77e41c2b00cfa","runtime_observer_baseline_file_sha256":hashlib.sha256(subprocess.check_output(["git","show","e5337a086c62770c6386fa1ac5d77e41c2b00cfa:scripts/latent/run_h_iter_phase1_t0_preflight_r1_guard_proof_v1.py"],cwd=repo)).hexdigest(),"runtime_observer_repaired_file_sha256":file_sha("scripts/latent/run_h_iter_phase1_t0_preflight_r1_guard_proof_v1.py"),"runtime_observer_semantic_delta":"observe sys.executable lexically with str(sys.executable) instead of resolving the virtualenv entrypoint"},
      "case_contract":{"ordered_case_names":GUARD_CASE_NAMES,"case_count":5,"production_pass_count":1,"mutation_reject_count":4,"required_qualifying_count":5},
      "terminal_contract":{"proof_schema":"prime-rl/latent-h-iter-phase1-t0-preflight-r1-guard-proof/v2","failure_schema":"prime-rl/latent-h-iter-phase1-t0-preflight-r1-guard-failure/v2","success_status":"h_iter_phase1_t0_preflight_r1_guard_mechanism_validated","proof_filename":"T0-PREFLIGHT-R1-GUARD-PROOF.json","failure_filename":"T0-PREFLIGHT-R1-GUARD-FAILURE.json","atomic_exclusive":True,"reopen_validate":True,"dual_terminal_forbidden":True},
      "resource_bounds":{"minimum_ram_gib":8,"minimum_disk_gib":8,"maximum_artifact_bytes":1048576,"timing":{"outer":720,"startup":60,"compute":300,"compute_alarm":299,"audit":120,"audit_alarm":119,"failure":90,"failure_alarm":89,"terminal":30,"terminal_alarm":29,"postexit":30,"reserve":90,"success_terminal_entry_max":480,"compute_failure_terminal_entry_max":450,"audit_failure_terminal_entry_max":570,"prior_terminal_failure_entry_max":600}},
      "full_freeze":{"head_exact":True,"tree_exact":True,"clean_before_after":True,"assets_pre_post_exact":True,"failed_start_exact":True,"output_excluded":True},
      "execution_authorization":{"guard_proof_eligible_after_independent_review":True,"t0_preflight_authorized":False,"t0_full_authorized":False,"model_load_authorized":False,"gpu_authorized":False,"training_authorized":False},
      "decision_boundary":GUARD_DECISION_DESIGN,
    }
    value["plan_sha256"]=sha256_bytes(canonical_json({k:v for k,v in value.items() if k!="plan_sha256"}))
    validate_guard_plan(value,repo=repo)
    return value

def validate_guard_plan(value:dict,repo:Path|None=None)->None:
    expected_keys={"schema_version","status","mechanism","run_identity","implementation_commit","mechanism_code_commit","plan_sha256","asset_sha256","remote_paths","runtime","failed_start_binding","failed_run1_archive_binding","diagnostic_archive_binding","source_contract","case_contract","terminal_contract","resource_bounds","full_freeze","execution_authorization","decision_boundary"}
    if set(value)!=expected_keys or value["schema_version"]!=GUARD_PLAN_SCHEMA or value["status"]!="h_iter_phase1_t0_preflight_r1_guard_proof_preregistered" or value["mechanism"]!=GUARD_MECHANISM or value["run_identity"]!=GUARD_RUN_ID: raise RuntimeError("T0 guard plan identity differs")
    if value["implementation_commit"]!=value["mechanism_code_commit"] or len(value["implementation_commit"])!=40: raise RuntimeError("T0 guard plan commit differs")
    if value["plan_sha256"]!=sha256_bytes(canonical_json({k:v for k,v in value.items() if k!="plan_sha256"})): raise RuntimeError("T0 guard plan self hash differs")
    if value["failed_start_binding"]!=FAILED_START_BINDING or value["failed_run1_archive_binding"]!=FAILED_RUN1_ARCHIVE_BINDING or value["diagnostic_archive_binding"]!=DIAGNOSTIC_ARCHIVE_BINDING or value["case_contract"]!={"ordered_case_names":GUARD_CASE_NAMES,"case_count":5,"production_pass_count":1,"mutation_reject_count":4,"required_qualifying_count":5}: raise RuntimeError("T0 guard plan frozen contract differs")
    if list(sorted(value["asset_sha256"]))!=sorted(GUARD_PROOF_ASSETS) or len(value["asset_sha256"])!=57: raise RuntimeError("T0 guard plan assets differ")
    if value["execution_authorization"]!={"guard_proof_eligible_after_independent_review":True,"t0_preflight_authorized":False,"t0_full_authorized":False,"model_load_authorized":False,"gpu_authorized":False,"training_authorized":False} or value["decision_boundary"]!=GUARD_DECISION_DESIGN: raise RuntimeError("T0 guard plan boundary differs")
    remote={"repo":"/home/ubuntu/rlm/worktrees/q35-2b-latent-workspace-v1","shared_project":"/home/ubuntu/rlm/prime-rl","shared_python":"/home/ubuntu/rlm/prime-rl/.venv/bin/python3","proof_output":"/home/ubuntu/rlm/outputs/q35-2b-h-iter-phase1-t0-preflight-r1-guard-proof-run2","t0_output":OUTPUT_ROOT}
    runtime={"python":"3.12.14","sys_executable":"/home/ubuntu/rlm/prime-rl/.venv/bin/python3","sys_prefix":"/home/ubuntu/rlm/prime-rl/.venv","torch":"2.11.0+cu128","transformers":"5.6.2","tokenizers":"0.22.2","flash_linear_attention":"0.5.2","shared_project_pyproject_sha256":"504907808f992f1e6883f54c2695a4814ae77d6b80814239cbfc98d81a543656","shared_project_uv_lock_sha256":"fca5fa6183345b5b68974078c38d58e0320f79eef13a695af11ceab12fdf36d5","cuda_visible_devices":""}
    terminal={"proof_schema":"prime-rl/latent-h-iter-phase1-t0-preflight-r1-guard-proof/v2","failure_schema":"prime-rl/latent-h-iter-phase1-t0-preflight-r1-guard-failure/v2","success_status":"h_iter_phase1_t0_preflight_r1_guard_mechanism_validated","proof_filename":"T0-PREFLIGHT-R1-GUARD-PROOF.json","failure_filename":"T0-PREFLIGHT-R1-GUARD-FAILURE.json","atomic_exclusive":True,"reopen_validate":True,"dual_terminal_forbidden":True}
    resources={"minimum_ram_gib":8,"minimum_disk_gib":8,"maximum_artifact_bytes":1048576,"timing":{"outer":720,"startup":60,"compute":300,"compute_alarm":299,"audit":120,"audit_alarm":119,"failure":90,"failure_alarm":89,"terminal":30,"terminal_alarm":29,"postexit":30,"reserve":90,"success_terminal_entry_max":480,"compute_failure_terminal_entry_max":450,"audit_failure_terminal_entry_max":570,"prior_terminal_failure_entry_max":600}}
    freeze={"head_exact":True,"tree_exact":True,"clean_before_after":True,"assets_pre_post_exact":True,"failed_start_exact":True,"output_excluded":True}
    source=value["source_contract"]
    source_keys={"baseline_execution_commit","baseline_mechanism_code_commit","baseline_runner_path","baseline_runner_file_sha256","repaired_runner_path","repaired_runner_file_sha256","allowed_change","ast_canonicalizer","normalization_rule","forbidden_call_attributes","forbidden_filenames","ordered_case_names","runtime_observer_runner_path","runtime_observer_baseline_commit","runtime_observer_baseline_file_sha256","runtime_observer_repaired_file_sha256","runtime_observer_semantic_delta"}
    if value["remote_paths"]!=remote or value["runtime"]!=runtime or value["terminal_contract"]!=terminal or value["resource_bounds"]!=resources or value["full_freeze"]!=freeze: raise RuntimeError("T0 guard plan nested contract differs")
    if set(source)!=source_keys or source["baseline_execution_commit"]!="67b21d2ccd7cc3189154f67a80fe8db6abd3ec7b" or source["baseline_mechanism_code_commit"]!="6d43c0175256f00c61bb08e26e61ff6112489911" or source["baseline_runner_path"]!="scripts/latent/run_h_iter_phase1_t0_v1.py" or source["baseline_runner_file_sha256"]!="d6f8fcfb153e81c6aaf5faa42caf1700abba25a0594108ea8db4837034ca6bee" or source["repaired_runner_path"]!="scripts/latent/run_h_iter_phase1_t0_v1.py" or source["forbidden_call_attributes"]!=["generate","save_pretrained"] or source["forbidden_filenames"]!=["validation-bank.json","heldout-bank.json"] or source["ordered_case_names"]!=GUARD_CASE_NAMES or source["runtime_observer_runner_path"]!="scripts/latent/run_h_iter_phase1_t0_preflight_r1_guard_proof_v1.py" or source["runtime_observer_baseline_commit"]!="e5337a086c62770c6386fa1ac5d77e41c2b00cfa" or source["runtime_observer_semantic_delta"]!="observe sys.executable lexically with str(sys.executable) instead of resolving the virtualenv entrypoint": raise RuntimeError("T0 guard plan source contract differs")
    if repo is not None:
        actual={path:hashlib.sha256((repo/path).read_bytes()).hexdigest() for path in sorted(GUARD_PROOF_ASSETS)}
        if actual!=value["asset_sha256"]: raise RuntimeError("T0 guard plan asset hashes differ")
        if source["repaired_runner_file_sha256"]!=actual["scripts/latent/run_h_iter_phase1_t0_v1.py"]: raise RuntimeError("T0 guard repaired runner hash differs")
        baseline_observer=hashlib.sha256(subprocess.check_output(["git","show",f"{source['runtime_observer_baseline_commit']}:{source['runtime_observer_runner_path']}"],cwd=repo)).hexdigest()
        if source["runtime_observer_baseline_file_sha256"]!=baseline_observer or source["runtime_observer_repaired_file_sha256"]!=actual[source["runtime_observer_runner_path"]]: raise RuntimeError("T0 guard runtime observer source differs")
        manifest=read_json(repo/FAILED_MANIFEST)
        if manifest.get("manifest_sha256")!=FAILED_START_BINDING["evidence_manifest_internal_sha256"] or hashlib.sha256((repo/FAILED_MANIFEST).read_bytes()).hexdigest()!=FAILED_START_BINDING["evidence_manifest_file_sha256"]: raise RuntimeError("T0 guard failed start manifest differs")
        git=lambda *args:subprocess.check_output(["git",*args],cwd=repo,text=True).strip()
        evidence=FAILED_START_BINDING["evidence_commit"]; archive=FAILED_START_BINDING["archive_freeze_commit"]
        if git("rev-parse",f"{evidence}^")!=FAILED_START_BINDING["failed_execution_commit"] or git("rev-parse",f"{archive}^")!=evidence or git("rev-parse",f"{archive}^{{tree}}")!=git("rev-parse",f"{evidence}^{{tree}}") or git("diff-tree","--no-commit-id","--name-only","-r",archive): raise RuntimeError("T0 guard failed archive ancestry differs")
        validate_guard_archives(repo)

def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--repo",type=Path,required=True); parser.add_argument("--materialize-only",action="store_true"); parser.add_argument("--final",action="store_true"); parser.add_argument("--guard-proof",action="store_true"); parser.add_argument("--mechanism-commit")
    args=parser.parse_args(); repo=args.repo.resolve(strict=True)
    if args.materialize_only:
        materialize(repo); return
    if not args.mechanism_commit: raise RuntimeError("--mechanism-commit required")
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()
    dirty=subprocess.check_output(["git","status","--porcelain","--untracked-files=all"],cwd=repo,text=True).strip()
    if head!=args.mechanism_commit or dirty: raise RuntimeError("T0 mechanism tree is not exact and clean")
    value=guard_plan_value(repo,args.mechanism_commit) if args.guard_proof else plan_value(repo,args.mechanism_commit,proof_input=not args.final); encoded=canonical_json(value)+b"\n"
    if args.guard_proof: plan=repo/T0_DIR/"t0-preflight-r1-guard-proof-run2-plan.json"; sidecar=repo/T0_DIR/"t0-preflight-r1-guard-proof-run2-plan.sha256"
    else: plan=repo/T0_DIR/("t0-plan.json" if args.final else "t0-model-free-proof-plan.json"); sidecar=repo/T0_DIR/("t0-plan.sha256" if args.final else "t0-model-free-proof-plan.sha256")
    write_atomic(plan,value); write_bytes_atomic(sidecar,((value["plan_sha256"] if args.guard_proof else hashlib.sha256(encoded).hexdigest())+"\n").encode())
    print(hashlib.sha256(encoded).hexdigest()); print(value["plan_sha256"])

if __name__=="__main__": main()
