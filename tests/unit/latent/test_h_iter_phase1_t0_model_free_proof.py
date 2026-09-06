from __future__ import annotations

import json
from pathlib import Path

import pytest

from prime_rl.latent.h_iter_phase1_t0 import (
    MECHANISM, MODEL_FREE_DECISION, MODEL_FREE_FAILURE_SCHEMA,
    MODEL_FREE_PROOF_KEYS, PROOF_MEMORY_LABELS, PROOF_RUN_ID,
    SYNTHETIC_SEED, SYNTHETIC_SHA256, T0ContractError,
    build_tamper_schedule, canonical_json, sha256_bytes,
    validate_model_free_failure,
)

ROOT=Path(__file__).resolve().parents[3]

def test_literal_proof_contract() -> None:
    assert SYNTHETIC_SHA256=="12d7df1f3bd89bdcf6511029ae3ecd31308598c63b3b1a9f5a2a2cbafe59de0e"
    assert SYNTHETIC_SEED==1357799137916525532
    assert len(PROOF_MEMORY_LABELS)==18
    tamper=build_tamper_schedule()
    assert tamper["tamper_count"]==98
    assert [row["index"] for row in tamper["rows"]]==list(range(98))

def test_proof_asset_count_literal() -> None:
    import sys
    sys.path.insert(0,str(ROOT/"scripts/latent"))
    from freeze_h_iter_phase1_t0_plan_v1 import PROOF_ASSETS
    assert len(PROOF_ASSETS)==len(set(PROOF_ASSETS))==31

def test_all_tamper_targets_are_reachable_and_non_noop() -> None:
    import sys
    sys.path.insert(0,str(ROOT/"scripts/latent"))
    from run_h_iter_phase1_t0_model_free_proof_v1 import tamper_results
    phase=ROOT/"experiments/qwen35-2b-latent-workspace-v1/h-iter-phase1-t0-train-calibration-v1"
    mf0=ROOT/"experiments/qwen35-2b-latent-workspace-v1/h-iter-phase1-train-calibration-v1"
    load=lambda p:json.loads(p.read_text())
    contract=load(phase/"t0-contract.json"); capture=load(phase/"t0-capture-schedule.json"); schedule=load(mf0/"training-schedule.json"); tampers=load(phase/"t0-tamper-schedule.json")
    assert tampers["rows"][53]["target"]=="/safety/forbidden_inputs_detected"
    assert tampers["rows"][53]["operand"]==["depth_or_K_scalar"]
    assert tampers["rows"][54]["target"]=="/safety/forbidden_inputs_detected"
    assert tampers["rows"][54]["operand"]==["global_pool"]
    plan={"archive_bindings":{"mf0":{},"cap0_r1":{}},"data_contract":{},"training_contract":{},"capture_contract":{},"metric_gate_contract":{},"remote_paths":{},"model_contract":{}}
    # Use a production-shaped plan fixture from the freezer, then exercise every literal mutation.
    from freeze_h_iter_phase1_t0_plan_v1 import plan_value
    plan=plan_value(ROOT,"0"*40)
    partition=load(mf0/"train-partition.json"); memory=load(phase/"t0-memory-schedule.json")
    results=tamper_results(plan,contract,capture,schedule,tampers,partition,memory,ROOT)
    assert len(results)==98 and all(row["rejected"] for row in results)

def _failure(status:str,error_type:str,safety:dict|None,audit_errors:list[str]) -> dict:
    value={key:None for key in MODEL_FREE_PROOF_KEYS if key!="proof_sha256"}
    rows=[{"index":0,"label":"ENTRY","rss_bytes":1}]
    value.update({"schema_version":MODEL_FREE_FAILURE_SCHEMA,"status":status,"mechanism":MECHANISM,"run_identity":PROOF_RUN_ID,"execution_commit":"0"*40,"mechanism_code_commit":None,"tree_sha256":None,"plan_file_sha256":"1"*64,"plan_sha256":None,"safety":safety,"memory":{"expected_labels":PROOF_MEMORY_LABELS,"rows":rows,"label_sha256":sha256_bytes(canonical_json(PROOF_MEMORY_LABELS)),"complete":False},"decision_boundary":MODEL_FREE_DECISION,"error_type":error_type,"error_message":"fixture","traceback":"fixture","stage":"runtime","execution_progress":{"stage":"runtime","memory_rows_completed":1},"audit_errors":audit_errors,"failure_sha256":""})
    value["failure_sha256"]=sha256_bytes(canonical_json({k:v for k,v in value.items() if k!="failure_sha256"}))
    return value

def test_model_free_failure_boundary_and_taxonomy() -> None:
    boundary_safety={"torch_imported":True,"cuda_initialized":True,"tokenizer_loaded":False,"model_loaded":False,"optimizer_constructed":False,"train_scientific_forwards":0,"validation_opens":0,"heldout_opens":0,"network_attempts":0,"output_namespace_fresh":True,"object_census_errors":0,"object_census_uninspectable":0}
    boundary=_failure("h_iter_phase1_t0_model_free_proof_boundary_rejected","T0ModelFreeProofBoundaryRejected",boundary_safety,[])
    validate_model_free_failure(boundary)
    hidden=dict(boundary); hidden["status"]="h_iter_phase1_t0_model_free_proof_incomplete"; hidden["error_type"]="T0ModelFreeProofError"; hidden["failure_sha256"]=sha256_bytes(canonical_json({k:v for k,v in hidden.items() if k!="failure_sha256"}))
    with pytest.raises(T0ContractError): validate_model_free_failure(hidden)
    infra=_failure("infrastructure_invalid","InfrastructureInvalid",None,["fixture"]); validate_model_free_failure(infra)
