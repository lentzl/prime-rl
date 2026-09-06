from __future__ import annotations

import json
import copy
from pathlib import Path

import pytest

from prime_rl.latent.h_iter_phase1_t0 import (
    MECHANISM, MODEL_FREE_DECISION, MODEL_FREE_FAILURE_SCHEMA,
    MODEL_FREE_PROOF_KEYS, PROOF_MEMORY_LABELS, PROOF_RUN_ID,
    SYNTHETIC_SEED, SYNTHETIC_SHA256, T0ContractError,
    T0_COMPLETE_COUNTS, T0_FAILURE_MEMORY_BOUNDS,
    build_tamper_schedule, canonical_json, capture_failure_evidence_row_count, sha256_bytes,
    validate_capture_failure_microstate, validate_failure_memory_stage, validate_model_free_failure, validate_t0_failure, validate_t0_proof,
    FAILURE_SCHEMA,
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

def _production_sources():
    import sys
    sys.path.insert(0,str(ROOT/"scripts/latent"))
    from run_h_iter_phase1_t0_model_free_proof_v1 import production_fixture
    phase=ROOT/"experiments/qwen35-2b-latent-workspace-v1/h-iter-phase1-t0-train-calibration-v1"
    mf0=ROOT/"experiments/qwen35-2b-latent-workspace-v1/h-iter-phase1-train-calibration-v1"
    load=lambda p:json.loads(p.read_text())
    partition=load(mf0/"train-partition.json"); capture=load(phase/"t0-capture-schedule.json"); schedule=load(mf0/"training-schedule.json"); memory=load(phase/"t0-memory-schedule.json"); tampers=load(phase/"t0-tamper-schedule.json")
    validation={"partition":partition,"capture_schedule":capture,"training_schedule":schedule,"memory_schedule":memory}
    return production_fixture,partition,capture,schedule,memory,tampers,validation

def test_negative_nll_and_frozen_schedule_forgery_rejected() -> None:
    pytest.importorskip("torch")
    fixture,partition,capture,schedule,memory,tampers,validation=_production_sources()
    proof=fixture(partition,capture,schedule,memory,tampers,[],True)
    validate_t0_proof(proof,**validation)
    forged=copy.deepcopy(proof); forged["metric_evidence"]["row_records"][0]["nll"]=-100.0
    forged["proof_sha256"]=sha256_bytes(canonical_json({k:v for k,v in forged.items() if k!="proof_sha256"}))
    with pytest.raises(T0ContractError,match="metric logits"): validate_t0_proof(forged,**validation)
    wrong_id=copy.deepcopy(proof); wrong_id["data_binding"]["fit_row_ids"][0]="hi_0000000000000000"; wrong_id["proof_sha256"]=sha256_bytes(canonical_json({k:v for k,v in wrong_id.items() if k!="proof_sha256"}))
    with pytest.raises(T0ContractError,match="data differs"): validate_t0_proof(wrong_id,**validation)
    wrong_capture=copy.deepcopy(proof); wrong_capture["capture_evidence"]["rows"][0]["row_id"]="hi_0000000000000000"; wrong_capture["capture_evidence"]["aggregate_sha256"]=sha256_bytes(canonical_json(wrong_capture["capture_evidence"]["rows"])); wrong_capture["proof_sha256"]=sha256_bytes(canonical_json({k:v for k,v in wrong_capture.items() if k!="proof_sha256"}))
    with pytest.raises(T0ContractError,match="capture row"): validate_t0_proof(wrong_capture,**validation)
    wrong_cache=copy.deepcopy(proof); wrong_cache["cache_guard"]["configuration_records"]=[]; wrong_cache["proof_sha256"]=sha256_bytes(canonical_json({k:v for k,v in wrong_cache.items() if k!="proof_sha256"}))
    with pytest.raises(T0ContractError,match="cache evidence"): validate_t0_proof(wrong_cache,**validation)

def test_impossible_failure_progress_and_memory_rejected() -> None:
    pytest.importorskip("torch")
    fixture,partition,capture,schedule,memory,tampers,validation=_production_sources()
    base=fixture(partition,capture,schedule,memory,tampers,[],True)
    failure={k:v for k,v in base.items() if k!="proof_sha256"}
    progress={"stage":"postflight_audit","capture_rows_completed":96,"tokenizer_calls_completed":96,"model_forwards_completed":96,"sequences_completed":2304,"cache_checks_completed":0,"candidates_initialized":5,"preconnect_arms_completed":5,"operations_completed":385,"current_operation_index":384,"current_phase":"POSTFIT","current_arm":"REC_K","current_epoch":None,"current_depth":4,"sidecar_forwards_completed":0,"sidecar_backwards_completed":0,"optimizer_steps_completed":0,"cell_calls_completed":0,"metric_row_records_completed":640,"aggregate_records_completed":135,"gates_evaluated":9,"candidate_files_present":[],"threshold_file_present":False}
    failure.update({"schema_version":FAILURE_SCHEMA,"status":"h_iter_phase1_t0_incomplete","error_type":"T0ContractError","error_message":"IMPOSSIBLE_FAILURE_ACCEPTED","traceback":"fixture","stage":"postflight_audit","execution_progress":progress,"audit_errors":[],"memory":{"schedule_file_sha256":sha256_bytes(canonical_json(memory)+b"\n"),"expected_count":534,"rows":[{"index":0,"label":"ARBITRARY","current_allocated_bytes":0,"current_reserved_bytes":0,"peak_allocated_bytes":0,"peak_reserved_bytes":0}],"label_sha256":sha256_bytes(canonical_json(["ARBITRARY"])),"complete":False},"failure_sha256":""})
    failure["failure_sha256"]=sha256_bytes(canonical_json({k:v for k,v in failure.items() if k!="failure_sha256"}))
    with pytest.raises(T0ContractError): validate_t0_failure(failure,**validation)

def _truthful_gate_failure(base:dict,memory:dict) -> dict:
    value=copy.deepcopy({key:item for key,item in base.items() if key!="proof_sha256"})
    progress={"stage":"gate_evaluation","capture_rows_completed":96,"tokenizer_calls_completed":96,"model_forwards_completed":96,"sequences_completed":2304,"cache_checks_completed":194,"candidates_initialized":5,"preconnect_arms_completed":5,"operations_completed":385,"current_operation_index":384,"current_phase":"POSTFIT","current_arm":"REC_K","current_epoch":None,"current_depth":4,"sidecar_forwards_completed":385,"sidecar_backwards_completed":325,"optimizer_steps_completed":320,"cell_calls_completed":773,"metric_row_records_completed":640,"aggregate_records_completed":135,"gates_evaluated":9,"candidate_files_present":[],"threshold_file_present":False}
    rows=value["memory"]["rows"][:530]
    value["memory"]={"schedule_file_sha256":sha256_bytes(canonical_json(memory)+b"\n"),"expected_count":534,"rows":rows,"label_sha256":sha256_bytes(canonical_json([row["label"] for row in rows])),"complete":False}
    value["asset_audit"].update({"post_entries":None,"post_sha256":None,"all_exact":None}); value["resources"]["timing"]=None; value["full_freeze"]=None; value["tamper_audit"]=None
    value["counts"].update({"memory_rows":530,"tamper_count":0,"candidate_files":0,"threshold_files":0})
    value.update({"schema_version":FAILURE_SCHEMA,"status":"h_iter_phase1_t0_incomplete","error_type":"T0ContractError","error_message":"truthful gate failure","traceback":"fixture","stage":"gate_evaluation","execution_progress":progress,"audit_errors":[],"failure_sha256":""})
    value["failure_sha256"]=sha256_bytes(canonical_json({key:item for key,item in value.items() if key!="failure_sha256"}))
    return value

def test_failure_counts_and_late_audits_are_deeply_bound() -> None:
    pytest.importorskip("torch")
    fixture,partition,capture,schedule,memory,tampers,validation=_production_sources()
    base=fixture(partition,capture,schedule,memory,tampers,[],True)
    failure=_truthful_gate_failure(base,memory); validate_t0_failure(failure,**validation)
    mutations=[]
    extra=copy.deepcopy(failure); extra["counts"]["fabricated_count"]=1; mutations.append(extra)
    full_at_zero=copy.deepcopy(failure); full_at_zero["counts"]["capture_rows"]=0; mutations.append(full_at_zero)
    full_at_one=copy.deepcopy(failure); full_at_one["counts"]["capture_rows"]=1; mutations.append(full_at_one)
    for field in ("runtime","asset_audit","antecedent_binding","data_binding","resources"):
        altered=copy.deepcopy(failure); altered[field]=None; mutations.append(altered)
    for altered in mutations:
        altered["failure_sha256"]=sha256_bytes(canonical_json({key:item for key,item in altered.items() if key!="failure_sha256"}))
        with pytest.raises(T0ContractError): validate_t0_failure(altered,**validation)

def test_complete_counts_cannot_be_claimed_at_zero_or_one_capture() -> None:
    pytest.importorskip("torch")
    fixture,partition,capture,schedule,memory,tampers,validation=_production_sources()
    base=fixture(partition,capture,schedule,memory,tampers,[],True)
    zero=copy.deepcopy({key:item for key,item in base.items() if key!="proof_sha256"})
    progress={"stage":"startup_pre_model","capture_rows_completed":0,"tokenizer_calls_completed":0,"model_forwards_completed":0,"sequences_completed":0,"cache_checks_completed":0,"candidates_initialized":0,"preconnect_arms_completed":0,"operations_completed":0,"current_operation_index":None,"current_phase":None,"current_arm":None,"current_epoch":None,"current_depth":None,"sidecar_forwards_completed":0,"sidecar_backwards_completed":0,"optimizer_steps_completed":0,"cell_calls_completed":0,"metric_row_records_completed":0,"aggregate_records_completed":0,"gates_evaluated":0,"candidate_files_present":[],"threshold_file_present":False}
    for field in ("runtime","asset_audit","antecedent_binding","data_binding","capture_evidence","candidate_initial_state","preconnect_evidence","metric_evidence","gate_evidence","derived_thresholds","protected_state","cache_guard","resources","memory","full_freeze","tamper_audit","decision_boundary"): zero[field]=None
    zero["candidate_disposition"]={"arm_order":["STATIC","FFN","FIXED_T4","RESET_K","REC_K"],"files_present":[],"threshold_file_present":False,"rows":[],"reusable":False}
    zero["counts"]={key:0 for key in base["counts"]}
    zero.update({"schema_version":FAILURE_SCHEMA,"status":"infrastructure_invalid","error_type":"InfrastructureInvalid","error_message":"startup fixture","traceback":"fixture","stage":"startup_pre_model","execution_progress":progress,"audit_errors":["startup fixture"],"failure_sha256":""}); zero["failure_sha256"]=sha256_bytes(canonical_json({key:item for key,item in zero.items() if key!="failure_sha256"}))
    validate_t0_failure(zero,**validation)
    forged=copy.deepcopy(zero); forged["counts"]={**T0_COMPLETE_COUNTS,"candidate_files":0,"threshold_files":0}; forged["failure_sha256"]=sha256_bytes(canonical_json({key:item for key,item in forged.items() if key!="failure_sha256"}))
    with pytest.raises(T0ContractError,match="counts/progress"): validate_t0_failure(forged,**validation)
    one=copy.deepcopy(zero); one_progress=copy.deepcopy(progress); one_progress.update({"stage":"capture","capture_rows_completed":1,"tokenizer_calls_completed":1,"model_forwards_completed":1,"sequences_completed":24,"cache_checks_completed":3}); one.update({"status":"h_iter_phase1_t0_exposure_boundary_rejected","error_type":"T0ExposureBoundaryRejected","error_message":"one capture fixture","stage":"capture","execution_progress":one_progress,"audit_errors":[]})
    for field in ("runtime","asset_audit","antecedent_binding","data_binding","resources"): one[field]=copy.deepcopy(base[field])
    one["asset_audit"].update({"post_entries":None,"post_sha256":None,"all_exact":None}); one["resources"]["timing"]=None
    capture_rows=base["capture_evidence"]["rows"][:1]; one["capture_evidence"]={"schedule_sha256":base["capture_evidence"]["schedule_sha256"],"rows":capture_rows,"counts":{"rows":1,"tokenizer_calls":1,"model_forwards":1,"sequences":24},"aggregate_sha256":sha256_bytes(canonical_json(capture_rows))}
    one["protected_state"]=copy.deepcopy(base["protected_state"]); one["protected_state"].update({"e33_grads_none":False,"e33_released":False})
    cache_rows=base["cache_guard"]["label_rows"][:3]; one["cache_guard"]={**base["cache_guard"],"label_rows":cache_rows,"label_sha256":sha256_bytes(canonical_json([row["label"] for row in cache_rows])),"actual_checks":3}
    memory_rows=base["memory"]["rows"][:10]; one["memory"]={**base["memory"],"rows":memory_rows,"label_sha256":sha256_bytes(canonical_json([row["label"] for row in memory_rows])),"complete":False}
    one["counts"].update({"capture_rows":1,"tokenizer_calls":1,"model_forwards":1,"sequences":24,"cache_checks":3,"memory_rows":10}); one["failure_sha256"]=sha256_bytes(canonical_json({key:item for key,item in one.items() if key!="failure_sha256"}))
    validate_t0_failure(one,**validation)
    forged_one=copy.deepcopy(one); forged_one["counts"]={**T0_COMPLETE_COUNTS,"candidate_files":0,"threshold_files":0}; forged_one["failure_sha256"]=sha256_bytes(canonical_json({key:item for key,item in forged_one.items() if key!="failure_sha256"}))
    with pytest.raises(T0ContractError,match="counts/progress"): validate_t0_failure(forged_one,**validation)

def test_failure_memory_stage_boundaries_and_future_evidence() -> None:
    pytest.importorskip("torch")
    expected={"startup_pre_model":(0,5),"model_load":(5,6),"capture":(6,201),"model_release":(201,203),"candidate_init":(203,204),"preconnect":(204,205),"precal":(205,206),"train":(206,527),"postcal":(527,528),"postfit":(528,529),"gate_evaluation":(529,531),"candidate_write":(530,531),"postflight_audit":(531,534),"terminal_publication":(534,534)}
    assert T0_FAILURE_MEMORY_BOUNDS==expected
    for stage,(low,high) in expected.items():
        validate_failure_memory_stage(stage,low); validate_failure_memory_stage(stage,high)
        if low:
            with pytest.raises(T0ContractError,match="memory stage boundary"): validate_failure_memory_stage(stage,low-1)
        with pytest.raises(T0ContractError,match="memory stage boundary"): validate_failure_memory_stage(stage,high+1)
    fixture,partition,capture,schedule,memory,tampers,validation=_production_sources()
    base=fixture(partition,capture,schedule,memory,tampers,[],True)
    progress={"stage":"startup_pre_model","capture_rows_completed":0,"tokenizer_calls_completed":0,"model_forwards_completed":0,"sequences_completed":0,"cache_checks_completed":0,"candidates_initialized":0,"preconnect_arms_completed":0,"operations_completed":0,"current_operation_index":None,"current_phase":None,"current_arm":None,"current_epoch":None,"current_depth":None,"sidecar_forwards_completed":0,"sidecar_backwards_completed":0,"optimizer_steps_completed":0,"cell_calls_completed":0,"metric_row_records_completed":0,"aggregate_records_completed":0,"gates_evaluated":0,"candidate_files_present":[],"threshold_file_present":False}
    zero=copy.deepcopy({key:item for key,item in base.items() if key!="proof_sha256"})
    for field in ("runtime","asset_audit","antecedent_binding","data_binding","capture_evidence","candidate_initial_state","preconnect_evidence","metric_evidence","gate_evidence","derived_thresholds","protected_state","cache_guard","resources","memory","full_freeze","tamper_audit","decision_boundary"): zero[field]=None
    zero["candidate_disposition"]={"arm_order":["STATIC","FFN","FIXED_T4","RESET_K","REC_K"],"files_present":[],"threshold_file_present":False,"rows":[],"reusable":False}
    zero["counts"]={key:0 for key in base["counts"]}
    zero.update({"schema_version":FAILURE_SCHEMA,"status":"infrastructure_invalid","error_type":"InfrastructureInvalid","error_message":"startup fixture","traceback":"fixture","stage":"startup_pre_model","execution_progress":progress,"audit_errors":["startup fixture"],"failure_sha256":""})
    def rejected(value:dict,pattern:str|None=None)->None:
        value["failure_sha256"]=sha256_bytes(canonical_json({key:item for key,item in value.items() if key!="failure_sha256"}))
        with pytest.raises(T0ContractError,match=pattern): validate_t0_failure(value,**validation)
    future_memory=copy.deepcopy(zero)
    for field in ("runtime","asset_audit","antecedent_binding","data_binding","resources"): future_memory[field]=copy.deepcopy(base[field])
    future_memory["asset_audit"].update({"post_entries":None,"post_sha256":None,"all_exact":None}); future_memory["resources"]["timing"]=None
    future_rows=base["memory"]["rows"][:8]
    future_memory["memory"]={**base["memory"],"rows":future_rows,"label_sha256":sha256_bytes(canonical_json([row["label"] for row in future_rows])),"complete":False}; future_memory["counts"]["memory_rows"]=8
    rejected(future_memory,"memory stage boundary")
    for field in ("candidate_initial_state","preconnect_evidence","capture_evidence","cache_guard"):
        future=copy.deepcopy(zero); future[field]=copy.deepcopy(base[field]); rejected(future)
    empty_capture=copy.deepcopy(zero); empty_capture["capture_evidence"]={"schedule_sha256":capture["schedule_sha256"],"rows":[],"counts":{"rows":0,"tokenizer_calls":0,"model_forwards":0,"sequences":0},"aggregate_sha256":sha256_bytes(canonical_json([]))}; rejected(empty_capture,"capture evidence")
    empty_cache=copy.deepcopy(zero); empty_cache["cache_guard"]={}; rejected(empty_cache,"cache evidence")

def test_cache_entry_configuration_drift_is_honest_and_stage_bound() -> None:
    pytest.importorskip("torch")
    fixture,partition,capture,schedule,memory,tampers,validation=_production_sources()
    base=fixture(partition,capture,schedule,memory,tampers,[],True)
    value=copy.deepcopy({key:item for key,item in base.items() if key!="proof_sha256"})
    progress={"stage":"capture","capture_rows_completed":0,"tokenizer_calls_completed":0,"model_forwards_completed":0,"sequences_completed":0,"cache_checks_completed":0,"candidates_initialized":0,"preconnect_arms_completed":0,"operations_completed":0,"current_operation_index":None,"current_phase":None,"current_arm":None,"current_epoch":None,"current_depth":None,"sidecar_forwards_completed":0,"sidecar_backwards_completed":0,"optimizer_steps_completed":0,"cell_calls_completed":0,"metric_row_records_completed":0,"aggregate_records_completed":0,"gates_evaluated":0,"candidate_files_present":[],"threshold_file_present":False}
    for field in ("capture_evidence","candidate_initial_state","preconnect_evidence","metric_evidence","gate_evidence","derived_thresholds","full_freeze","tamper_audit","decision_boundary"): value[field]=None
    value["asset_audit"].update({"post_entries":None,"post_sha256":None,"all_exact":None}); value["resources"]["timing"]=None; value["protected_state"]["e33_released"]=False
    records=copy.deepcopy(base["cache_guard"]["configuration_records"]); records[0]["value_during"]=True
    value["cache_guard"].update({"configuration_records":records,"label_rows":[],"label_sha256":sha256_bytes(canonical_json([])),"actual_checks":0,"dynamic_cache_negative_trips":0,"dynamic_cache_actual_trips":0,"returned_pkv_count":0,"configuration_drift_count":1,"restored":True})
    rows=base["memory"]["rows"][:6]; value["memory"]={**base["memory"],"rows":rows,"label_sha256":sha256_bytes(canonical_json([row["label"] for row in rows])),"complete":False}
    value["candidate_disposition"]={"arm_order":["STATIC","FFN","FIXED_T4","RESET_K","REC_K"],"files_present":[],"threshold_file_present":False,"rows":[],"reusable":False}; value["counts"]={key:0 for key in base["counts"]}; value["counts"]["memory_rows"]=6
    value.update({"schema_version":FAILURE_SCHEMA,"status":"h_iter_phase1_t0_capture_mechanism_rejected","error_type":"T0CaptureMechanismRejected","error_message":"CACHE_ENTRY configuration drift","traceback":"fixture","stage":"capture","execution_progress":progress,"audit_errors":[],"failure_sha256":""})
    def rehash(item:dict)->None: item["failure_sha256"]=sha256_bytes(canonical_json({key:field for key,field in item.items() if key!="failure_sha256"}))
    rehash(value); validate_t0_failure(value,**validation)
    masked=copy.deepcopy(value); masked.update({"status":"h_iter_phase1_t0_incomplete","error_type":"T0ContractError"}); rehash(masked)
    with pytest.raises(T0ContractError,match="cause exclusivity"): validate_t0_failure(masked,**validation)
    spurious=copy.deepcopy(value); spurious["cache_guard"]["configuration_records"][0]["value_during"]=False; rehash(spurious)
    with pytest.raises(T0ContractError,match="cache entry drift evidence"): validate_t0_failure(spurious,**validation)
    premature_label=copy.deepcopy(value); drift_rows=base["memory"]["rows"][:7]; premature_label["memory"].update({"rows":drift_rows,"label_sha256":sha256_bytes(canonical_json([row["label"] for row in drift_rows]))}); premature_label["counts"]["memory_rows"]=7; rehash(premature_label)
    with pytest.raises(T0ContractError,match="capture microstate"): validate_t0_failure(premature_label,**validation)

def test_cache_guard_entry_drift_restores_before_reraising(monkeypatch:pytest.MonkeyPatch) -> None:
    import sys
    sys.path.insert(0,str(ROOT/"scripts/latent"))
    import run_h_iter_phase1_t0_v1 as runner
    class StickyConfig:
        def __init__(self)->None: self.value=True
        @property
        def use_cache(self)->bool: return self.value
        @use_cache.setter
        def use_cache(self,value:bool)->None:
            if value: self.value=True
    class Cache: pass
    class DynamicCache(Cache): pass
    class CacheUtils: pass
    CacheUtils.Cache=Cache; CacheUtils.DynamicCache=DynamicCache
    class Model:
        config=StickyConfig(); generation_config=None
        def named_modules(self): return []
    runner.FAILURE_EVIDENCE["cache_guard"]=None; runner.PROGRESS["cache_checks_completed"]=0
    guard=runner.CacheGuard(Model(),CacheUtils,[])
    monkeypatch.setattr(guard,"evidence",lambda:{"configuration_drift_count":guard.drift,"dynamic_cache_negative_trips":guard.negative_trips,"restored":guard.restored})
    with pytest.raises(runner.T0CaptureMechanismRejected,match="configuration drift"): guard.__enter__()
    assert guard.labels==[] and guard.drift==1 and guard.negative_trips==0 and guard.trips==0 and guard.restored is True
    assert runner.PROGRESS["cache_checks_completed"]==0
    assert runner.FAILURE_EVIDENCE["cache_guard"]=={"configuration_drift_count":1,"dynamic_cache_negative_trips":0,"restored":True}

def test_mid_prefix_exit_drift_and_negative_control_no_trip() -> None:
    pytest.importorskip("torch")
    fixture,partition,capture,schedule,memory,tampers,validation=_production_sources(); base=fixture(partition,capture,schedule,memory,tampers,[],True)
    def cache_failure(name:str,captures:int,tokenizers:int,forwards:int,checks:int,memory_count:int,evidence_rows:int,drift:bool,status:str,error_type:str)->dict:
        value=copy.deepcopy({key:item for key,item in base.items() if key!="proof_sha256"})
        progress={"stage":"capture","capture_rows_completed":captures,"tokenizer_calls_completed":tokenizers,"model_forwards_completed":forwards,"sequences_completed":24*tokenizers,"cache_checks_completed":checks,"candidates_initialized":0,"preconnect_arms_completed":0,"operations_completed":0,"current_operation_index":None,"current_phase":None,"current_arm":None,"current_epoch":None,"current_depth":None,"sidecar_forwards_completed":0,"sidecar_backwards_completed":0,"optimizer_steps_completed":0,"cell_calls_completed":0,"metric_row_records_completed":0,"aggregate_records_completed":0,"gates_evaluated":0,"candidate_files_present":[],"threshold_file_present":False}
        for field in ("candidate_initial_state","preconnect_evidence","metric_evidence","gate_evidence","derived_thresholds","full_freeze","tamper_audit","decision_boundary"): value[field]=None
        value["asset_audit"].update({"post_entries":None,"post_sha256":None,"all_exact":None}); value["resources"]["timing"]=None; value["protected_state"]["e33_released"]=False
        rows=value["capture_evidence"]["rows"][:evidence_rows]
        value["capture_evidence"]=None if not rows else {"schedule_sha256":capture["schedule_sha256"],"rows":rows,"counts":{"rows":evidence_rows,"tokenizer_calls":tokenizers,"model_forwards":forwards,"sequences":24*tokenizers},"aggregate_sha256":sha256_bytes(canonical_json(rows))}
        config=copy.deepcopy(value["cache_guard"]["configuration_records"])
        if drift: config[0]["value_during"]=True
        labels=value["cache_guard"]["label_rows"][:checks]
        value["cache_guard"].update({"configuration_records":config,"label_rows":labels,"label_sha256":sha256_bytes(canonical_json([row["label"] for row in labels])),"actual_checks":checks,"dynamic_cache_negative_trips":0 if name=="NO_TRIP" else 1,"dynamic_cache_actual_trips":0,"returned_pkv_count":0,"configuration_drift_count":int(drift),"restored":True})
        memory_rows=value["memory"]["rows"][:memory_count]; value["memory"].update({"rows":memory_rows,"label_sha256":sha256_bytes(canonical_json([row["label"] for row in memory_rows])),"complete":False})
        value["candidate_disposition"]={"arm_order":["STATIC","FFN","FIXED_T4","RESET_K","REC_K"],"files_present":[],"threshold_file_present":False,"rows":[],"reusable":False}; value["counts"]={key:0 for key in base["counts"]}; value["counts"].update({"capture_rows":captures,"tokenizer_calls":tokenizers,"model_forwards":forwards,"sequences":24*tokenizers,"cache_checks":checks,"memory_rows":memory_count})
        value.update({"schema_version":FAILURE_SCHEMA,"status":status,"error_type":error_type,"error_message":name,"traceback":"fixture","stage":"capture","execution_progress":progress,"audit_errors":[name] if status=="infrastructure_invalid" else [],"failure_sha256":""}); value["failure_sha256"]=sha256_bytes(canonical_json({key:item for key,item in value.items() if key!="failure_sha256"})); return value
    variants=[
        cache_failure("PRE_DRIFT",1,2,1,3,11,1,True,"h_iter_phase1_t0_capture_mechanism_rejected","T0CaptureMechanismRejected"),
        cache_failure("POST_DRIFT",1,2,2,4,11,2,True,"h_iter_phase1_t0_capture_mechanism_rejected","T0CaptureMechanismRejected"),
        cache_failure("EXIT_DRIFT",96,96,96,193,201,96,True,"h_iter_phase1_t0_capture_mechanism_rejected","T0CaptureMechanismRejected"),
        cache_failure("NO_TRIP",0,0,0,1,6,0,False,"h_iter_phase1_t0_incomplete","T0ContractError"),
    ]
    for variant in variants: validate_t0_failure(variant,**validation)
    returned_variants=[
        cache_failure("RETURNED_PKV_FIRST",0,1,1,2,9,0,False,"h_iter_phase1_t0_capture_mechanism_rejected","T0CaptureMechanismRejected"),
        cache_failure("RETURNED_PKV_MIDDLE",1,2,2,4,11,1,False,"h_iter_phase1_t0_capture_mechanism_rejected","T0CaptureMechanismRejected"),
        cache_failure("RETURNED_PKV_LAST",95,96,96,192,199,95,False,"h_iter_phase1_t0_capture_mechanism_rejected","T0CaptureMechanismRejected"),
    ]
    for returned in returned_variants:
        returned["cache_guard"]["returned_pkv_count"]=1; returned["failure_sha256"]=sha256_bytes(canonical_json({key:item for key,item in returned.items() if key!="failure_sha256"})); validate_t0_failure(returned,**validation)
    allocation=cache_failure("CACHE_ALLOCATION",0,1,0,2,9,0,False,"h_iter_phase1_t0_capture_mechanism_rejected","T0CaptureMechanismRejected"); allocation["cache_guard"]["dynamic_cache_actual_trips"]=1; allocation["failure_sha256"]=sha256_bytes(canonical_json({key:item for key,item in allocation.items() if key!="failure_sha256"})); validate_t0_failure(allocation,**validation)
    nonfinite=cache_failure("NONFINITE",1,2,2,4,11,2,False,"h_iter_phase1_t0_capture_mechanism_rejected","T0CaptureMechanismRejected"); nonfinite["capture_evidence"]["rows"][-1]["finite"]=False; nonfinite["capture_evidence"]["counts"]["rows"]=1; nonfinite["capture_evidence"]["aggregate_sha256"]=sha256_bytes(canonical_json(nonfinite["capture_evidence"]["rows"])); nonfinite["failure_sha256"]=sha256_bytes(canonical_json({key:item for key,item in nonfinite.items() if key!="failure_sha256"})); validate_t0_failure(nonfinite,**validation)
    pre_post_finite=cache_failure("PRE_POST_FINITE",1,2,2,4,11,2,False,"infrastructure_invalid","InfrastructureInvalid"); validate_t0_failure(pre_post_finite,**validation)
    post_ledger=cache_failure("POST_LEDGER_PRE_COUNTER",1,2,2,5,12,2,False,"infrastructure_invalid","InfrastructureInvalid"); validate_t0_failure(post_ledger,**validation)
    causal_rejections=[]
    actual_overcount=copy.deepcopy(allocation); actual_overcount["cache_guard"]["dynamic_cache_actual_trips"]=2; causal_rejections.append(actual_overcount)
    returned_overcount=copy.deepcopy(returned_variants[0]); returned_overcount["cache_guard"]["returned_pkv_count"]=2; causal_rejections.append(returned_overcount)
    combined=copy.deepcopy(nonfinite); combined["cache_guard"]["returned_pkv_count"]=1; causal_rejections.append(combined)
    completed_false=copy.deepcopy(returned_variants[1]); completed_false["capture_evidence"]["rows"][0]["finite"]=False; completed_false["capture_evidence"]["aggregate_sha256"]=sha256_bytes(canonical_json(completed_false["capture_evidence"]["rows"])); causal_rejections.append(completed_false)
    nonmechanism_cause=copy.deepcopy(pre_post_finite); nonmechanism_cause["cache_guard"]["returned_pkv_count"]=1; causal_rejections.append(nonmechanism_cause)
    for malformed in causal_rejections:
        malformed["failure_sha256"]=sha256_bytes(canonical_json({key:item for key,item in malformed.items() if key!="failure_sha256"}))
        with pytest.raises(T0ContractError): validate_t0_failure(malformed,**validation)
    for variant in variants[:3]:
        masked=copy.deepcopy(variant); masked.update({"status":"h_iter_phase1_t0_incomplete","error_type":"T0ContractError"}); masked["failure_sha256"]=sha256_bytes(canonical_json({key:item for key,item in masked.items() if key!="failure_sha256"}))
        with pytest.raises(T0ContractError,match="cause exclusivity"): validate_t0_failure(masked,**validation)
        spurious=copy.deepcopy(variant); spurious["cache_guard"]["configuration_records"][0]["value_during"]=False; spurious["failure_sha256"]=sha256_bytes(canonical_json({key:item for key,item in spurious.items() if key!="failure_sha256"}))
        with pytest.raises(T0ContractError,match="cache entry drift evidence"): validate_t0_failure(spurious,**validation)
    false_reject=copy.deepcopy(variants[3]); false_reject.update({"status":"h_iter_phase1_t0_capture_mechanism_rejected","error_type":"T0CaptureMechanismRejected"}); false_reject["failure_sha256"]=sha256_bytes(canonical_json({key:item for key,item in false_reject.items() if key!="failure_sha256"}))
    with pytest.raises(T0ContractError,match="cause exclusivity"): validate_t0_failure(false_reject,**validation)

def test_capture_failure_microstate_transition_table() -> None:
    legal=[(0,0,0,0,6),(0,0,0,1,6),(0,0,0,1,7),(0,0,0,1,8),(0,1,0,1,8),(0,1,0,1,9),(0,1,0,2,9),(0,1,1,2,9),(0,1,1,3,9),(0,1,1,3,10),(1,1,1,3,10),(95,96,95,191,198),(95,96,95,191,199),(95,96,95,192,199),(95,96,96,192,199),(95,96,96,193,199),(95,96,96,193,200),(96,96,96,193,200),(96,96,96,193,201),(96,96,96,194,201)]
    for state in legal: validate_capture_failure_microstate("capture",*state)
    illegal=[(0,0,0,2,9),(0,1,0,3,9),(0,1,0,2,8),(0,1,1,1,9),(0,1,1,2,10),(0,1,1,3,11),(1,1,1,3,9),(96,96,96,194,200)]
    for state in illegal:
        with pytest.raises(T0ContractError,match="capture microstate"): validate_capture_failure_microstate("capture",*state)
    assert capture_failure_evidence_row_count([{"finite":True}],0,1)==1
    assert capture_failure_evidence_row_count([{"finite":False}],0,1)==0
    assert capture_failure_evidence_row_count([{"finite":True}],1,2)==1

def test_postflight_failure_requires_freeze_tamper_and_decision_evidence() -> None:
    pytest.importorskip("torch")
    fixture,partition,capture,schedule,memory,tampers,validation=_production_sources()
    base=fixture(partition,capture,schedule,memory,tampers,[],True)
    failure=_truthful_gate_failure(base,memory)
    failure.update({"status":"infrastructure_invalid","error_type":"InfrastructureInvalid","error_message":"postflight fixture","stage":"postflight_audit","audit_errors":["postflight fixture"]})
    failure["execution_progress"]["stage"]="postflight_audit"
    rows=base["memory"]["rows"][:531]; failure["memory"]={**failure["memory"],"rows":rows,"label_sha256":sha256_bytes(canonical_json([row["label"] for row in rows]))}; failure["counts"]["memory_rows"]=531
    failure["asset_audit"]=copy.deepcopy(base["asset_audit"]); failure["resources"]=copy.deepcopy(base["resources"]); failure["full_freeze"]=copy.deepcopy(base["full_freeze"]); failure["tamper_audit"]=copy.deepcopy(base["tamper_audit"])
    failure["counts"]["tamper_count"]=98; failure["failure_sha256"]=sha256_bytes(canonical_json({key:item for key,item in failure.items() if key!="failure_sha256"}))
    validate_t0_failure(failure,**validation)
    for field in ("runtime","resources","asset_audit","full_freeze","tamper_audit","decision_boundary"):
        altered=copy.deepcopy(failure); altered[field]=None; altered["failure_sha256"]=sha256_bytes(canonical_json({key:item for key,item in altered.items() if key!="failure_sha256"}))
        with pytest.raises(T0ContractError): validate_t0_failure(altered,**validation)

def test_candidate_artifact_unsafe_variants_rejected() -> None:
    torch=pytest.importorskip("torch")
    import sys
    sys.path.insert(0,str(ROOT/"scripts/latent"))
    from run_h_iter_phase1_t0_model_free_proof_v1 import candidate_roundtrip
    from prime_rl.latent.h_iter_phase1_t0 import INIT_SEED, candidate_class
    torch.manual_seed(INIT_SEED); state={name:tensor.detach().clone() for name,tensor in candidate_class(torch)().state_dict().items()}
    evidence=candidate_roundtrip(torch,state)
    assert evidence["all_safe"] is True
    assert evidence["unsafe_rejections"]==["wrong_name","wrong_shape","wrong_dtype","nonfinite","unsafe_format","symlink"]
