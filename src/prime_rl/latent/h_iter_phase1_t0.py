from __future__ import annotations

import hashlib
import json
import math
from typing import Any

MECHANISM = "q35-2b-h-iter-phase1-t0-train-calibration-v1"
RUN_ID = "h-iter-phase1-t0-train-calibration-run1"
PROOF_RUN_ID = "h-iter-phase1-t0-model-free-proof-run1"
ARTIFACT_DIR = "experiments/qwen35-2b-latent-workspace-v1/h-iter-phase1-t0-train-calibration-v1"
OUTPUT_ROOT = "/home/ubuntu/rlm/outputs/q35-2b-h-iter-phase1-t0-train-calibration-run1"
PROOF_OUTPUT_ROOT = "/home/ubuntu/rlm/outputs/q35-2b-h-iter-phase1-t0-model-free-proof-run1"
PLAN_SCHEMA = "prime-rl/latent-h-iter-phase1-t0-plan/v1"
CONTRACT_SCHEMA = "prime-rl/latent-h-iter-phase1-t0-contract/v1"
PROOF_SCHEMA = "prime-rl/latent-h-iter-phase1-t0-proof/v1"
FAILURE_SCHEMA = "prime-rl/latent-h-iter-phase1-t0-failure/v1"
MODEL_FREE_PROOF_SCHEMA = "prime-rl/latent-h-iter-phase1-t0-model-free-proof/v1"
MODEL_FREE_FAILURE_SCHEMA = "prime-rl/latent-h-iter-phase1-t0-model-free-failure/v1"
PREFLIGHT_SCHEMA = "prime-rl/latent-h-iter-phase1-t0-preflight/v1"
ARMS = ["STATIC", "FFN", "FIXED_T4", "RESET_K", "REC_K"]
ACTIONS = ["ACT_Z1", "ACT_K4", "ACT_M7", "ACT_Q9"]
CANDIDATE_FILES = [f"candidate-{arm}.safetensors" for arm in ARMS]
TRAIN_BANK_PATH = "experiments/qwen35-2b-latent-workspace-v1/h-iter-phase0-generator-locality-v1/train-bank.json"
TRAIN_BANK_FILE_SHA256 = "12f6f9a000c1fa13380b7d58d302ad9d2f75ebc5eeb1922d1a088f5fec4bdbfd"
TRAIN_BANK_INTERNAL_SHA256 = "1dd675f276cbe3164ed03901c6036761cee12f39c0571d3edc40d0d37fa4aca2"
MF0_DIR = "experiments/qwen35-2b-latent-workspace-v1/h-iter-phase1-train-calibration-v1"
CAP0_DIR = "experiments/qwen35-2b-latent-workspace-v1/h-iter-phase1-cap0-r1-v1"
PARTITION_PATH = f"{MF0_DIR}/train-partition.json"
SCHEDULE_PATH = f"{MF0_DIR}/training-schedule.json"
CANDIDATE_CONTRACT_PATH = f"{MF0_DIR}/candidate-module-contract.json"
CAPTURE_CONTRACT_PATH = f"{MF0_DIR}/capture-contract.json"
METRIC_CONTRACT_PATH = f"{MF0_DIR}/metric-gate-contract.json"
THRESHOLD_CONTRACT_PATH = f"{MF0_DIR}/threshold-builder-contract.json"
INIT_PAYLOAD = "q35-2b-h-iter-phase1-train-calibration-v1:init"
INIT_SHA256 = "f42df2afcbe75e6962c9abd4aac781c61f202f61274935ba1761f89f8492034d"
INIT_SEED = 17594986156060532329
SYNTHETIC_PAYLOAD = "q35-2b-h-iter-phase1-t0-model-free-proof-v1:synthetic"
SYNTHETIC_SHA256 = hashlib.sha256(SYNTHETIC_PAYLOAD.encode()).hexdigest()
SYNTHETIC_SEED = int(SYNTHETIC_SHA256[:16], 16)

E33_PATH = "/home/ubuntu/rlm/outputs/q35-2b-adaptive-cognition-sft-v1/c54-step8-action4-adaptive-nonroot-step2-v4/weights/step_2"
H176_PATH = "/home/ubuntu/rlm/outputs/q35-2b-document-child-sft-v1/h176child8-document-child-real12-step8-v2/weights/step_8"
E33_TREE_SHA256 = "e33bd4cdbfd92eb22844dbbde2764aa7fa00e1cd25ca7045f91ce22210499e47"
E33_STATE_SHA256 = "dd6a76377c6e43a28efe484927e0a8427026cc3517fac0aea5dd9d6972cc1bf9"
H176_TREE_SHA256 = "77980e247bbccd6463ddda02cd42d2c357e15f8ec1ad0ea84627e008a8674a1e"
METADATA_SHA256 = {"chat_template.jinja":"273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80","config.json":"22949388ed61c1100b20a3cae55bb22122554c74e06fc23f1be50cca1fec3b8c","generation_config.json":"93f19a5ed0fb9f9e8e65dafae7a9bc4c6a32b3e37f6278980d05d3f4ca29f17b","processor_config.json":"d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1","tokenizer.json":"06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523","tokenizer_config.json":"747ba36a06ba5428bb74e984d75136b37cf5dafe97b8dd315f701b361a9f417f"}

MF0_BINDING = {"archive_freeze_commit":"4087ecde6da743f1a248bf99493264ecac459c63","evidence_commit":"197fb0ba67273015c9db98b52f230c875c745ca9","manifest_path":f"{MF0_DIR}/mf0-prereg-run1-evidence-manifest.json","manifest_file_sha256":"79caa566a74bd73ef4b56002f67f9584c5ac76d2521ab96b13afa6ad07aa0140","manifest_internal_sha256":"c0a9034efe192a93efd3d755e0769e2dfadc2745b5772c8955fc43f319fa9758","proof_path":f"{MF0_DIR}/mf0-prereg-run1.MF0-PROOF.json","proof_file_sha256":"7b1f99f06adbc1282511a0e05306304e0b44ffc9c7411eaad8963543c67fa6fc","proof_internal_sha256":"7101a4f19783911567c9b301dfd416cdcaac7b763d1d80f46355821367898a06","launcher_log_path":f"{MF0_DIR}/mf0-prereg-run1.launcher.log","launcher_log_sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","exit_path":f"{MF0_DIR}/mf0-prereg-run1.exit.txt","exit_file_sha256":"9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa"}
CAP0_BINDING = {"archive_freeze_commit":"b358a4d38c3eed0bd266cdc9aceb00c215ba559f","evidence_commit":"7fd6b405356dbe7851eee04a1f0eb135b953316b","manifest_path":f"{CAP0_DIR}/cap0-r1-run1-evidence-manifest.json","manifest_file_sha256":"f60ca3e03e0bde9ad1818355feb362d729f2bfa51bb43f162cb0e4de08d3fd28","manifest_internal_sha256":"09d1ae6e14b95387398eb629cdb01b8ad705ef753f60d3707bf737cef323fae2","proof_path":f"{CAP0_DIR}/cap0-r1-run1.CAP0-R1-PROOF.json","proof_file_sha256":"c5716451c958d4382cdfd1853bc73db2d1d8b8c698d58668040d50ac9275afa9","proof_internal_sha256":"29651f11b9f719c7adc9435a139556e015f7cee79c2c260ee2e88ff225a6a7cc","launcher_log_path":f"{CAP0_DIR}/cap0-r1-run1.launcher.log","launcher_log_sha256":"e9963d000c16c399adf9b823a62e1951929391f6e6ee10aa768727c44c93a534","exit_path":f"{CAP0_DIR}/cap0-r1-run1.exit.txt","exit_file_sha256":"9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa"}

GATE_NAMES = ["rec_postcal_accuracy","rec_minus_reset_accuracy","rec_minus_max_static_ffn_accuracy","rec_each_depth","rec_each_action_recall","reset_nll_minus_rec","min_static_ffn_nll_minus_rec","rec_postcal_nll_factor","rec_postfit_accuracy"]
PROOF_MEMORY_LABELS = ["ENTRY","RUNTIME_VERIFIED","FULL_FREEZE_PREFLIGHT_VERIFIED","ASSETS_PREFLIGHT_VERIFIED","ANTECEDENTS_VALIDATED","TRAIN_SCHEMA_VALIDATED","CONTRACTS_VALIDATED","CPU_MODULES_VALIDATED","CONNECTIVITY_VALIDATED","GO_REPLAY_VALIDATED","STOP_REPLAY_VALIDATED","FAILURE_REPLAYS_VALIDATED","TAMPERS_VALIDATED","CANDIDATE_SAFETY_VALIDATED","TERMINAL_ROUNDTRIP_VALIDATED","SAFETY_AUDIT_COMPLETE","FULL_FREEZE_POSTFLIGHT_VERIFIED","TERMINAL_PREWRITE"]

class T0ContractError(RuntimeError): pass
class T0CaptureMechanismRejected(RuntimeError): pass
class T0ExposureBoundaryRejected(RuntimeError): pass
class T0ModelFreeProofError(RuntimeError): pass
class T0ModelFreeProofBoundaryRejected(RuntimeError): pass
class InfrastructureInvalid(RuntimeError): pass

def canonical_json(value: Any) -> bytes:
    try: return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
    except (TypeError,ValueError) as error: raise T0ContractError("T0 noncanonical value") from error

def strict_loads(data: bytes) -> Any:
    def pairs(items: list[tuple[str,Any]]) -> dict[str,Any]:
        out: dict[str,Any]={}
        for key,value in items:
            if key in out: raise T0ContractError("T0 duplicate JSON key")
            out[key]=value
        return out
    return json.loads(data,object_pairs_hook=pairs,parse_constant=lambda value: (_ for _ in ()).throw(T0ContractError("T0 nonfinite JSON")))

def sha256_bytes(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def finish(value: dict[str,Any], field: str) -> dict[str,Any]:
    value[field]=sha256_bytes(canonical_json({k:v for k,v in value.items() if k!=field})); return value

def build_antecedent_manifest() -> dict[str,Any]:
    rows=[]
    for order,(name,binding) in enumerate((("MF0",MF0_BINDING),("CAP0_R1",CAP0_BINDING)),1): rows.append({"order":order,"name":name,**binding})
    return finish({"schema_version":"prime-rl/latent-h-iter-phase1-t0-antecedent-evidence-manifest/v1","status":"h_iter_phase1_t0_antecedents_bound","mechanism":MECHANISM,"ordered_antecedents":rows,"manifest_sha256":""},"manifest_sha256")

def build_capture_schedule(bank: dict[str,Any]) -> dict[str,Any]:
    rows=[{"capture_index":i,"row_id":row["row_id"],"row_sha256":row["row_sha256"],"receiver_input_sha256":row["receiver_input_sha256"],"node_count":len(row["receiver_input"]["nodes"])} for i,row in enumerate(bank["rows"])]
    return finish({"schema_version":"prime-rl/latent-h-iter-phase1-t0-capture-schedule/v1","status":"h_iter_phase1_t0_capture_schedule_preregistered","mechanism":MECHANISM,"source_train_bank_file_sha256":TRAIN_BANK_FILE_SHA256,"source_train_bank_internal_sha256":TRAIN_BANK_INTERNAL_SHA256,"order_rule":"source train-bank rows array order","rows":rows,"counts":{"rows":96,"nodes":2304,"tokenizer_calls":96,"model_forwards":96,"sequences":2304},"schedule_sha256":""},"schedule_sha256")

def build_memory_schedule(capture: dict[str,Any], schedule: dict[str,Any]) -> dict[str,Any]:
    labels=["RUNTIME_VERIFIED","FULL_FREEZE_PREFLIGHT_VERIFIED","ANTECEDENTS_VALIDATED","TRAIN_INPUTS_VALIDATED","PROTECTED_PREFLIGHT_VERIFIED","MODEL_LOADED_FROZEN","CACHE_GUARD_ENTERED","CAPTURE_STORAGE_INITIALIZED"]
    for row in capture["rows"]:
        labels.extend([f"PRE_CAPTURE_{row['capture_index']:03d}_{row['row_id']}",f"POST_CAPTURE_{row['capture_index']:03d}_{row['row_id']}"])
    labels.extend(["CACHE_GUARD_EXITED","PROTECTED_POSTCAPTURE_VERIFIED","MODEL_RELEASED","CANDIDATES_INITIALIZED","PRECONNECT_COMPLETE","PRECAL_COMPLETE"])
    for row in schedule["batches"]["train"]: labels.append(f"POST_TRAIN_UPDATE_{row['update_index']:03d}_{row['arm']}_E{row['epoch']:02d}_D{row['depth']}")
    labels.extend(["TRAINING_COMPLETE","POSTCAL_COMPLETE","POSTFIT_COMPLETE","GATES_EVALUATED","CANDIDATE_DISPOSITION_COMPLETE","PROTECTED_POSTFLIGHT_VERIFIED","FULL_FREEZE_POSTFLIGHT_VERIFIED","TERMINAL_PREWRITE"])
    if len(labels)!=534: raise T0ContractError("T0 memory label count differs")
    return finish({"schema_version":"prime-rl/latent-h-iter-phase1-t0-memory-schedule/v1","status":"h_iter_phase1_t0_memory_schedule_preregistered","mechanism":MECHANISM,"rows":[{"index":i,"label":label} for i,label in enumerate(labels)],"memory_count":534,"schedule_sha256":""},"schedule_sha256")

_TAMPER_NAMES = "mf0_manifest_file,mf0_manifest_internal,mf0_proof_file,mf0_proof_internal,mf0_log,mf0_exit,cap0_manifest_file,cap0_manifest_internal,cap0_proof_file,cap0_proof_internal,cap0_log,cap0_exit,train_bank,train_partition,training_schedule,candidate_contract,capture_contract,metric_gate_contract,threshold_builder_contract,fit_cal_overlap,capture_row_order,capture_row_hash,action_order,arm_order,epoch_order,depth_order,batch_membership,capture_checkpoint_path,capture_state_tree_hash,capture_model_state_hash,capture_metadata_hash,tokenizer_geometry,tokenizer_truncation,capture_dtype,hidden_selection,use_cache,output_hidden_states,logits_to_keep,cache_class_closure,source_local_text_hash,dynamic_cache_negative_control,returned_pkv,capture_count,h176_loaded,init_seed,initial_state_parity,parameter_names,parameter_shapes,parameter_count,training_dtype,training_device,tf32,dropout,forbidden_depth_input,forbidden_global_pool,start_only_readout,static_semantics,ffn_semantics,fixed_t4_semantics,reset_k_semantics,rec_k_semantics,preconnect_row,preconnect_gradient_min,preconnect_state_unchanged,optimizer_type,optimizer_lr,optimizer_betas,optimizer_eps,optimizer_weight_decay,gradient_clip,zero_grad_semantics,updates_per_arm,shuffle_or_sampling,objective_loss,prepost_schedule,nll_formula,prediction_tie,aggregation_order,gate_postcal_accuracy,gate_reset_accuracy_advantage,gate_static_ffn_accuracy_advantage,gate_depth_floor,gate_action_floor,gate_reset_nll_advantage,gate_static_ffn_nll_advantage,gate_nll_improvement,gate_postfit_accuracy,stop_semantics,validation_firewall,heldout_firewall,generation_firewall,e33_protection,candidate_output_order,memory_schedule,resource_cap,terminal_status,self_hash,failure_taxonomy".split(",")
_TAMPER_TARGETS = [
"/archive_bindings/mf0/manifest_path","/archive_bindings/mf0/manifest_internal_sha256","/archive_bindings/mf0/proof_path","/archive_bindings/mf0/proof_internal_sha256","/archive_bindings/mf0/launcher_log_path","/archive_bindings/mf0/exit_path","/archive_bindings/cap0_r1/manifest_path","/archive_bindings/cap0_r1/manifest_internal_sha256","/archive_bindings/cap0_r1/proof_path","/archive_bindings/cap0_r1/proof_internal_sha256","/archive_bindings/cap0_r1/launcher_log_path","/archive_bindings/cap0_r1/exit_path","/data_contract/train_bank_path","/data_contract/partition_path","/data_contract/schedule_path","/training_contract/candidate_contract_path","/capture_contract/source_contract_path","/metric_gate_contract/metric_contract_path","/metric_gate_contract/threshold_contract_path","/data_binding/calibration_row_ids/0","/rows/0","/rows/0/row_sha256","/candidate_initial_state/action_order/0","/candidate_initial_state/arm_order/0","/batches/train/0/epoch","/depth_order/0","/batches/train/0/row_ids/0","/remote_paths/e33","/protected_state/e33_disk_tree_before","/protected_state/e33_state_after","/model_contract/metadata_sha256/tokenizer.json","/capture/tokenizer_max_length","/capture/tokenizer_truncation","/capture_evidence/rows/0/hidden_dtype","/capture/hidden_selection","/capture/use_cache","/capture/output_hidden_states","/capture/logits_to_keep","/cache_guard/class_records/0","/capture_evidence/rows/0/input_ids_sha256","/cache_guard/dynamic_cache_negative_trips","/cache_guard/returned_pkv_count","/counts/capture_rows","/protected_state/h176_loaded","/candidate_initial_state/seed_u64_be","/candidate_initial_state/rows/4/initial_state_sha256","/candidate_initial_state/rows/0/parameter_names/0","/candidate_initial_state/rows/0/parameter_shapes/0/shape/0","/candidate_initial_state/rows/0/parameter_count","/candidate_initial_state/rows/0/dtype","/candidate_initial_state/rows/0/device","/modules/tf32","/modules/dropout","/safety/forbidden_inputs_detected/0","/safety/forbidden_inputs_detected/0","/modules/readout","/modules/arm_semantics/STATIC","/modules/arm_semantics/FFN","/modules/arm_semantics/FIXED_T4","/modules/arm_semantics/RESET_K","/modules/arm_semantics/REC_K","/preconnect_evidence/row_id","/preconnect_evidence/rows/4/cell_grad_l2","/preconnect_evidence/rows/4/state_after_sha256","/training/optimizer/type","/training/optimizer/lr","/training/optimizer/betas/1","/training/optimizer/eps","/training/optimizer/weight_decay","/training/optimizer/gradient_clip_global_norm","/training/optimizer/zero_grad_set_to_none","/counts/optimizer_steps","/training/shuffle","/training/objective","/metric_evidence/row_records/0/phase","/metrics/nll_formula","/metrics/prediction_formula","/metrics/aggregation_order/0","/gate_evidence/ordered_rows/0/rhs","/gate_evidence/ordered_rows/1/rhs","/gate_evidence/ordered_rows/2/rhs","/gate_evidence/ordered_rows/3/rhs","/gate_evidence/ordered_rows/4/rhs","/gate_evidence/ordered_rows/5/rhs","/gate_evidence/ordered_rows/6/rhs","/gate_evidence/ordered_rows/7/rhs","/gate_evidence/ordered_rows/8/rhs","/decision_boundary/validation_contract_design_authorized","/safety/validation_opens","/safety/heldout_opens","/safety/generation_calls","/protected_state/e33_state_after","/candidate_files/0","/memory/rows/533","/resources/max_reserved_bytes","/status","/proof_sha256","/status"]

def build_tamper_schedule() -> dict[str,Any]:
    if len(_TAMPER_NAMES)!=98 or len(_TAMPER_TARGETS)!=98: raise T0ContractError("T0 tamper literal count differs")
    fixtures=["plan_copy"]*19+["terminal_copy","capture_schedule_copy","capture_schedule_copy","terminal_copy","terminal_copy","schedule_copy","schedule_copy","schedule_copy","plan_copy","terminal_copy","terminal_copy","plan_copy"]+["contract_copy","contract_copy","terminal_copy","contract_copy","contract_copy","contract_copy","contract_copy","terminal_copy","terminal_copy","terminal_copy","terminal_copy","terminal_copy","terminal_copy","terminal_copy","terminal_copy","terminal_copy","terminal_copy","terminal_copy"]+["contract_copy","contract_copy","terminal_copy","terminal_copy","contract_copy","contract_copy","contract_copy","contract_copy","contract_copy","terminal_copy","terminal_copy","terminal_copy"]+["contract_copy"]*10+["terminal_copy","contract_copy","contract_copy","contract_copy"]+["terminal_copy"]*10+["go_output_dir","terminal_copy","terminal_copy","terminal_copy","terminal_copy","failure_copy"]
    # Keep fixture assignment explicit at the three special late boundaries.
    fixtures=fixtures[:98]
    operators=["raw_append_hex","replace","raw_append_hex","replace","raw_append_hex","raw_replace_utf8","raw_append_hex","replace","raw_append_hex","replace","raw_append_hex","raw_replace_utf8",*["raw_append_hex"]*7,"replace","swap","replace","swap","swap","replace","swap","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","delete","replace","replace","replace","replace","replace","replace","replace","swap","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","swap","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","swap","delete","replace","replace","replace","replace"]
    operands=["00","0"*64,"00","0"*64,"00","1\n","00","0"*64,"00","0"*64,"00","1\n","00","00","00","00","00","00","00","hi_575cf5825abbc3a1","/rows/1","0"*64,"/candidate_initial_state/action_order/1","/candidate_initial_state/arm_order/1",1,"/depth_order/1","hi_b68d66d7d1372aa8",H176_PATH,"0"*64,"0"*64,"0"*64,127,True,"float32","hidden_states[-1][:,-2,:]",True,False,0,None,"0"*64,0,1,95,True,17594986156060532330,"0"*64,"/candidate_initial_state/rows/0/parameter_names/1",127,366339,"float16","cpu",True,True,"depth_or_K_scalar","global_pool","Linear(128,4,bias=True) on all-node pooled state","C(z,zeros)","C(z,adjacency_message)","persistent C exactly 3 times","persistent C exactly K times","K reset calls","hi_575cf5825abbc3a1",0.0,"0"*64,"SGD",0.002,0.999,1e-7,0.0,0.5,False,319,True,"sum 4-class cross entropy","POSTCAL","-log_softmax(logits.float(),dim=-1)[action_index]","argmax with highest-index tie","/metrics/aggregation_order/3",[21,32],[3,32],[1,32],[3,8],[2,8],0.04,0.01,0.8,[47,64],True,1,1,1,"0"*64,"/candidate_files/1",None,42949672961,"h_iter_phase1_t0_validation_authorized","0"*64,"infrastructure_invalid"]
    fixtures=[]
    for index in range(98):
        if index <= 18 or index in {27,30}: fixture="plan_copy"
        elif index in {20,21}: fixture="capture_schedule_copy"
        elif index in {24,25,26}: fixture="schedule_copy"
        elif index in {31,32,34,35,36,37,51,52,55,56,57,58,59,60,64,65,66,67,68,69,70,72,73,75,76,77}: fixture="contract_copy"
        elif index == 92: fixture="go_output_dir"
        elif index == 97: fixture="failure_copy"
        else: fixture="terminal_copy"
        fixtures.append(fixture)
    operators=["replace"]*98
    for index in {0,2,4,6,8,10,12,13,14,15,16,17,18}: operators[index]="raw_append_hex"
    for index in {5,11}: operators[index]="raw_replace_utf8"
    for index in {20,22,23,25,46,77,92}: operators[index]="swap"
    for index in {38,93}: operators[index]="delete"
    if not (len(fixtures)==len(operators)==len(operands)==98): raise T0ContractError("T0 tamper row construction differs")
    rows=[{"index":i,"name":_TAMPER_NAMES[i],"fixture":fixtures[i],"target":_TAMPER_TARGETS[i],"operator":operators[i],"operand":operands[i]} for i in range(98)]
    return finish({"schema_version":"prime-rl/latent-h-iter-phase1-t0-tamper-schedule/v1","status":"h_iter_phase1_t0_tamper_schedule_preregistered","mechanism":MECHANISM,"rows":rows,"tamper_count":98,"schedule_sha256":""},"schedule_sha256")

def build_contract(candidate: dict[str,Any], capture_source: dict[str,Any], metric: dict[str,Any], schedule: dict[str,Any]) -> dict[str,Any]:
    optimizer=candidate["determinism"]["optimizer"]
    return finish({"schema_version":CONTRACT_SCHEMA,"status":"h_iter_phase1_t0_contract_preregistered","mechanism":MECHANISM,
        "identities":{"run_identity":RUN_ID,"proof_run_identity":PROOF_RUN_ID,"experiment_dir":ARTIFACT_DIR,"output_root":OUTPUT_ROOT,"proof_output_root":PROOF_OUTPUT_ROOT},
        "antecedents":{"manifest_path":f"{ARTIFACT_DIR}/t0-antecedent-evidence-manifest.json","mf0_archive_freeze":MF0_BINDING["archive_freeze_commit"],"cap0_archive_freeze":CAP0_BINDING["archive_freeze_commit"],"require_exact_validation":True},
        "capture":{"contract_path":CAPTURE_CONTRACT_PATH,"schedule_path":f"{ARTIFACT_DIR}/t0-capture-schedule.json","graphs":96,"nodes_per_graph":24,"tokenizer_calls":96,"model_forwards":96,"sequences":2304,"cache_checks":194,"feature_shape":[24,2048],"feature_cpu_dtype":"torch.bfloat16","sidecar_dtype":"torch.float32","tokenizer_max_length":128,"tokenizer_truncation":False,"hidden_selection":"hidden_states[-1][:,-1,:]","use_cache":False,"output_hidden_states":True,"logits_to_keep":1},
        "modules":{"contract_path":CANDIDATE_CONTRACT_PATH,"arms":ARMS,"actions":ACTIONS,"input_dim":2048,"state_dim":128,"parameter_names_count":16,"parameters_per_arm":366340,"identical_initial_state":True,"tf32":False,"dropout":False,"readout":"Linear(128,4,bias=True) on indexed start vector only","arm_semantics":candidate["arm_semantics"]},
        "training":{"schedule_path":SCHEDULE_PATH,"operation_count":385,"phase_ranges":{"PRECONNECT":[0,4],"PRECAL":[5,24],"TRAIN":[25,344],"POSTCAL":[345,364],"POSTFIT":[365,384]},"optimizer":optimizer,"updates_per_arm":64,"total_updates":320,"total_forwards":385,"total_backwards":325,"total_cell_calls":773,"shuffle":False,"sampling":False,"objective":"mean 4-class cross entropy on final FP32 start-node logits"},
        "metrics":{"contract_path":METRIC_CONTRACT_PATH,"aggregation_order":metric["aggregation_order"],"nll_formula":"-torch.log_softmax(logits.detach().cpu().double(),dim=-1)[action_index]","prediction_formula":"torch.argmax with lowest-index tie","phase_order":["PRECAL","POSTCAL","POSTFIT"]},
        "gates":{"contract_path":METRIC_CONTRACT_PATH,"ordered_gate_names":GATE_NAMES,"all_required":True,"valid_miss_status":"h_iter_phase1_train_calibration_stop"},
        "outputs":{"proof_filename":"T0-PROOF.json","failure_filename":"T0-FAILURE.json","candidate_filenames":CANDIDATE_FILES,"derived_threshold_filename":"derived-validation-thresholds.json","stop_inventory":["T0-PROOF.json"],"go_inventory":[*CANDIDATE_FILES,"derived-validation-thresholds.json","T0-PROOF.json"]},
        "terminal":{"proof_schema":PROOF_SCHEMA,"failure_schema":FAILURE_SCHEMA,"complete_statuses":["h_iter_phase1_t0_validation_contract_design_authorized","h_iter_phase1_train_calibration_stop"],"failure_status_error_types":[{"status":"h_iter_phase1_t0_incomplete","error_type":"T0ContractError"},{"status":"h_iter_phase1_t0_capture_mechanism_rejected","error_type":"T0CaptureMechanismRejected"},{"status":"h_iter_phase1_t0_exposure_boundary_rejected","error_type":"T0ExposureBoundaryRejected"},{"status":"infrastructure_invalid","error_type":"InfrastructureInvalid"}],"atomic_publish":True,"reopen_byte_compare":True,"deep_validate":True,"dual_terminal_forbidden":True},
        "safety":{"train_only":True,"validation_opens":0,"heldout_opens":0,"h176_loads":0,"generation":0,"network_attempts":0,"e33_updates":0,"e33_backwards":0,"live_trajectories":0},
        "resources":{"memory_schedule_path":f"{ARTIFACT_DIR}/t0-memory-schedule.json","tamper_schedule_path":f"{ARTIFACT_DIR}/t0-tamper-schedule.json","minimum_gpu_free_gib":44,"maximum_allocated_or_reserved_gib":40,"minimum_ram_gib":64,"minimum_disk_gib":16,"maximum_output_bytes":33554432,"timing":{"outer":21600,"startup":600,"compute":18000,"audit":1200,"failure":1200,"terminal":300,"postexit":300,"alarm_safety_margin":1,"success_terminal_entry_max":19200,"compute_failure_terminal_entry_max":19200,"audit_failure_terminal_entry_max":20400,"prior_terminal_failure_entry_max":20700}},
        "proof":{"runner":"scripts/latent/run_h_iter_phase1_t0_model_free_proof_v1.py","launcher":"scripts/latent/run_h_iter_phase1_t0_model_free_proof_v1.sh","test":"tests/unit/latent/test_h_iter_phase1_t0_model_free_proof.py","run_identity":PROOF_RUN_ID,"output_root":PROOF_OUTPUT_ROOT,"terminal_filename":"T0-MODEL-FREE-PROOF.json","failure_filename":"T0-MODEL-FREE-FAILURE.json","model_free":True,"cuda_hidden":True,"required_tamper_count":98},"contract_sha256":""},"contract_sha256")

def validate_contract(value: dict[str,Any], candidate: dict[str,Any], capture_source: dict[str,Any], metric: dict[str,Any], schedule: dict[str,Any]) -> None:
    if value!=build_contract(candidate,capture_source,metric,schedule): raise T0ContractError("T0 contract differs")

def candidate_class(torch: Any) -> type:
    nn=torch.nn
    class Candidate(nn.Module):
        def __init__(self) -> None:
            super().__init__(); self.codec_ln=nn.LayerNorm(2048,eps=1e-5); self.codec_projection=nn.Linear(2048,128); self.self_norm=nn.LayerNorm(128,eps=1e-5); self.message_norm=nn.LayerNorm(128,eps=1e-5); self.cell_in=nn.Linear(256,256); self.cell_out=nn.Linear(256,128); self.post_norm=nn.LayerNorm(128,eps=1e-5); self.readout=nn.Linear(128,4)
        def cell(self,state:Any,message:Any)->Any: return self.post_norm(state+self.cell_out(torch.nn.functional.gelu(self.cell_in(torch.cat((self.self_norm(state),self.message_norm(message)),dim=-1)),approximate="none")))
        def forward(self,features:Any,successor:Any,arm:str,depth:int,start_indices:Any|None=None)->Any:
            z=torch.nn.functional.gelu(self.codec_projection(self.codec_ln(features)),approximate="none")
            if arm=="STATIC": state=z
            elif arm=="FFN": state=self.cell(z,torch.zeros_like(z))
            elif arm=="FIXED_T4":
                state=z
                for _ in range(4): state=self.cell(state,state.index_select(0,successor))
            elif arm=="RESET_K":
                state=z
                for _ in range(depth): state=self.cell(z,z.index_select(0,successor))
            elif arm=="REC_K":
                state=z
                for _ in range(depth): state=self.cell(state,state.index_select(0,successor))
            else: raise T0ContractError("T0 arm differs")
            if start_indices is None: start_indices=torch.tensor([0],dtype=torch.int64,device=state.device)
            return self.readout(state.index_select(0,start_indices))
    return Candidate

def tensor_raw_sha256(torch:Any,tensor:Any)->str: return sha256_bytes(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
def module_state_sha256(torch:Any,state:dict[str,Any])->str:
    digest=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        digest.update(canonical_json({"name":name,"dtype":str(tensor.dtype),"shape":list(tensor.shape),"sha256":tensor_raw_sha256(torch,tensor)})); digest.update(b"\n")
    return digest.hexdigest()

def validate_assets(bank:dict[str,Any],partition:dict[str,Any],schedule:dict[str,Any],capture:dict[str,Any],memory:dict[str,Any],tampers:dict[str,Any])->None:
    if bank.get("bank_sha256")!=TRAIN_BANK_INTERNAL_SHA256 or bank["bank_sha256"]!=sha256_bytes(canonical_json({k:v for k,v in bank.items() if k!="bank_sha256"})) or len(bank.get("rows",[]))!=96: raise T0ContractError("T0 train bank differs")
    for row in bank["rows"]:
        if row.get("split")!="train" or row.get("receiver_input_sha256")!=sha256_bytes(canonical_json(row["receiver_input"])) or row.get("row_sha256")!=sha256_bytes(canonical_json({k:v for k,v in row.items() if k!="row_sha256"})): raise T0ContractError("T0 train row differs")
    if partition.get("partition_sha256")!=sha256_bytes(canonical_json({k:v for k,v in partition.items() if k!="partition_sha256"})) or len(partition.get("fit_rows",[]))!=64 or len(partition.get("calibration_rows",[]))!=32: raise T0ContractError("T0 partition differs")
    refs=[{key:row[key] for key in ("depth","action_index","replicate","row_id","order_key_sha256","row_sha256","receiver_input_sha256")} for row in bank["rows"]]
    expected_fit=sorted((r for r in refs if r["replicate"] in range(4)),key=lambda r:(r["depth"],r["action_index"],r["replicate"],r["row_id"])); expected_cal=sorted((r for r in refs if r["replicate"] in {4,5}),key=lambda r:(r["depth"],r["action_index"],r["replicate"],r["row_id"]))
    if partition["fit_rows"]!=expected_fit or partition["calibration_rows"]!=expected_cal or set(r["row_id"] for r in expected_fit)&set(r["row_id"] for r in expected_cal): raise T0ContractError("T0 partition membership differs")
    if schedule.get("schedule_sha256")!=sha256_bytes(canonical_json({k:v for k,v in schedule.items() if k!="schedule_sha256"})): raise T0ContractError("T0 training schedule self hash differs")
    operations=sum((schedule["batches"][key] for key in ("preconnect","precal","train","postcal","postfit")),[])
    if [r["operation_index"] for r in operations]!=list(range(385)): raise T0ContractError("T0 operation schedule differs")
    if [r["update_index"] for r in schedule["batches"]["train"]]!=list(range(1,321)) or sum(len(r["row_ids"]) for r in schedule["batches"]["train"])!=5120: raise T0ContractError("T0 training updates differ")
    if capture!=build_capture_schedule(bank) or memory!=build_memory_schedule(capture,schedule) or tampers!=build_tamper_schedule(): raise T0ContractError("T0 materialized schedule differs")

def validate_archive_bytes(repo:Any,binding:dict[str,Any], *, kind:str)->None:
    from pathlib import Path
    root=Path(repo)
    expected={binding["manifest_path"]:binding["manifest_file_sha256"],binding["proof_path"]:binding["proof_file_sha256"],binding["launcher_log_path"]:binding["launcher_log_sha256"],binding["exit_path"]:binding["exit_file_sha256"]}
    for path,digest in expected.items():
        item=root/path
        if not item.is_file() or item.is_symlink() or sha256_bytes(item.read_bytes())!=digest: raise T0ContractError(f"T0 {kind} archive file differs")
    manifest=strict_loads((root/binding["manifest_path"]).read_bytes()); proof=strict_loads((root/binding["proof_path"]).read_bytes())
    if manifest.get("manifest_sha256")!=binding["manifest_internal_sha256"] or manifest["manifest_sha256"]!=sha256_bytes(canonical_json({k:v for k,v in manifest.items() if k!="manifest_sha256"})): raise T0ContractError(f"T0 {kind} manifest internal differs")
    if proof.get("proof_sha256")!=binding["proof_internal_sha256"] or proof["proof_sha256"]!=sha256_bytes(canonical_json({k:v for k,v in proof.items() if k!="proof_sha256"})): raise T0ContractError(f"T0 {kind} proof internal differs")
    if manifest.get("claim_boundary")!=proof.get("decision_boundary") or (root/binding["exit_path"]).read_bytes()!=b"0\n": raise T0ContractError(f"T0 {kind} claim boundary differs")

def finite_number(value:Any)->bool: return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)

MODEL_FREE_PROOF_KEYS={"schema_version","status","mechanism","run_identity","execution_commit","mechanism_code_commit","tree_sha256","plan_file_sha256","plan_sha256","runtime","asset_audit","antecedent_binding","data_binding","module_evidence","synthetic_evidence","terminal_replay_evidence","tamper_audit","safety","resources","memory","full_freeze","decision_boundary","proof_sha256"}
MODEL_FREE_DECISION={"claim":"t0_model_free_validator_and_execution_contract_only","independent_scientific_replicate":False,"train_bank_schema_opened":True,"train_scientific_model_exposure":False,"validation_or_heldout_opened":False,"model_or_gpu_authorized":False,"t0_training_authorized":False,"admission":False,"nomination":False,"promotion":False,"four_live_floor_unchanged":True}
PLAN_KEYS={"schema_version","status","mechanism","run_identity","implementation_commit","mechanism_code_commit","plan_sha256","asset_sha256","archive_bindings","remote_paths","runtime","model_contract","data_contract","capture_contract","training_contract","metric_gate_contract","terminal_contract","resource_bounds","full_freeze","execution_authorization","decision_boundary","model_free_proof_binding"}

def _keys(value:Any,expected:set[str],label:str)->None:
    if not isinstance(value,dict) or set(value)!=expected: raise T0ContractError(f"{label} keyset differs")

def validate_plan(value:dict[str,Any], *, proof_input:bool)->None:
    _keys(value,PLAN_KEYS,"T0 plan")
    if value["schema_version"]!=PLAN_SCHEMA or value["mechanism"]!=MECHANISM or value["implementation_commit"]!=value["mechanism_code_commit"]: raise T0ContractError("T0 plan identity differs")
    if value["plan_sha256"]!=sha256_bytes(canonical_json({k:v for k,v in value.items() if k!="plan_sha256"})): raise T0ContractError("T0 plan self hash differs")
    expected_status="h_iter_phase1_t0_model_free_proof_preregistered" if proof_input else "h_iter_phase1_t0_preregistered"; expected_run=PROOF_RUN_ID if proof_input else RUN_ID; expected_count=31 if proof_input else 38
    if value["status"]!=expected_status or value["run_identity"]!=expected_run or len(value["asset_sha256"])!=expected_count or list(value["asset_sha256"])!=sorted(value["asset_sha256"]): raise T0ContractError("T0 plan phase differs")
    if value["archive_bindings"].get("mf0")!=MF0_BINDING or value["archive_bindings"].get("cap0_r1")!=CAP0_BINDING: raise T0ContractError("T0 archive binding differs")
    if value["data_contract"].get("validation_open_allowed") is not False or value["data_contract"].get("heldout_open_allowed") is not False: raise T0ContractError("T0 data firewall differs")
    if value["training_contract"].get("arm_order")!=ARMS or value["training_contract"].get("action_order")!=ACTIONS or value["training_contract"].get("operation_count")!=385: raise T0ContractError("T0 training plan differs")
    if proof_input:
        if value["model_free_proof_binding"] is not None or value["execution_authorization"]!={"model_free_proof_eligible_after_independent_review":True,"t0_preflight_eligible":False,"t0_full_authorized":False,"model_load_authorized":False,"gpu_authorized":False,"training_authorized":False} or value["decision_boundary"].get("claim")!="model_free_proof_design_only": raise T0ContractError("T0 proof authorization differs")
    elif not isinstance(value["model_free_proof_binding"],dict): raise T0ContractError("T0 proof binding absent")

def validate_model_free_proof(value:dict[str,Any],plan:dict[str,Any],capture:dict[str,Any],schedule:dict[str,Any],tampers:dict[str,Any])->None:
    _keys(value,MODEL_FREE_PROOF_KEYS,"T0 model-free proof")
    if value["schema_version"]!=MODEL_FREE_PROOF_SCHEMA or value["status"]!="h_iter_phase1_t0_model_free_mechanism_validated" or value["mechanism"]!=MECHANISM or value["run_identity"]!=PROOF_RUN_ID: raise T0ContractError("T0 model-free proof identity differs")
    if value["mechanism_code_commit"]!=plan["mechanism_code_commit"] or value["plan_sha256"]!=plan["plan_sha256"]: raise T0ContractError("T0 model-free authority differs")
    if value["proof_sha256"]!=sha256_bytes(canonical_json({k:v for k,v in value.items() if k!="proof_sha256"})): raise T0ContractError("T0 model-free self hash differs")
    runtime=value["runtime"]; _keys(runtime,{"python","sys_executable","sys_prefix","torch","shared_pyproject_sha256","shared_uv_lock_sha256","cuda_visible_devices","cuda_initialized_before","cuda_initialized_after"},"T0 model-free runtime")
    if runtime["cuda_visible_devices"]!="" or runtime["cuda_initialized_before"] or runtime["cuda_initialized_after"]: raise T0ContractError("T0 model-free CUDA boundary differs")
    audit=value["asset_audit"]; _keys(audit,{"target_count","pre_entries","pre_sha256","post_entries","post_sha256","all_exact"},"T0 model-free asset audit")
    if audit["target_count"]!=31 or not audit["all_exact"] or audit["pre_entries"]!=audit["post_entries"] or audit["pre_sha256"]!=audit["post_sha256"]: raise T0ContractError("T0 model-free asset closure differs")
    binding=value["antecedent_binding"]; _keys(binding,{"manifest_file_sha256","manifest_internal_sha256","mf0_exact","cap0_r1_exact"},"T0 model-free antecedent")
    if not binding["mf0_exact"] or not binding["cap0_r1_exact"]: raise T0ContractError("T0 model-free antecedent differs")
    data=value["data_binding"]; _keys(data,{"train_bank_file_sha256","train_bank_internal_sha256","train_bank_open_count","train_rows_validated","capture_rows_validated","operations_validated","validation_open_count","heldout_open_count"},"T0 model-free data")
    if data!={"train_bank_file_sha256":TRAIN_BANK_FILE_SHA256,"train_bank_internal_sha256":TRAIN_BANK_INTERNAL_SHA256,"train_bank_open_count":1,"train_rows_validated":96,"capture_rows_validated":96,"operations_validated":385,"validation_open_count":0,"heldout_open_count":0}: raise T0ContractError("T0 model-free data boundary differs")
    module=value["module_evidence"]; _keys(module,{"arms","actions","parameter_names","parameter_name_count","parameters_per_arm","state_hashes_equal","devices","dtypes","finite"},"T0 model-free module")
    if module["arms"]!=ARMS or module["actions"]!=ACTIONS or module["parameter_name_count"]!=16 or module["parameters_per_arm"]!=366340 or not module["state_hashes_equal"] or module["devices"]!=["cpu"]*5 or module["dtypes"]!=["torch.float32"]*5 or not module["finite"]: raise T0ContractError("T0 model-free module differs")
    synthetic=value["synthetic_evidence"]; _keys(synthetic,{"seed_payload","seed_sha256","seed_u64_be","connectivity","go_fixture","stop_fixture","failure_fixtures","candidate_roundtrip"},"T0 model-free synthetic")
    if (synthetic["seed_payload"],synthetic["seed_sha256"],synthetic["seed_u64_be"])!=(SYNTHETIC_PAYLOAD,SYNTHETIC_SHA256,SYNTHETIC_SEED): raise T0ContractError("T0 model-free synthetic seed differs")
    if len(synthetic["connectivity"])!=5 or not all(row.get("qualifies") is True for row in synthetic["connectivity"]): raise T0ContractError("T0 model-free connectivity differs")
    if synthetic["go_fixture"].get("gate_pass_count")!=9 or synthetic["go_fixture"].get("output_count")!=7 or synthetic["stop_fixture"].get("gate_pass_count")!=8 or synthetic["stop_fixture"].get("output_count")!=1: raise T0ContractError("T0 model-free terminal fixtures differ")
    if [r.get("fixture") for r in synthetic["failure_fixtures"]]!=["F0","F1","F2","F3","F4","F5"] or not all(r.get("validated") is True and r.get("tamper_rejected") is True for r in synthetic["failure_fixtures"]): raise T0ContractError("T0 model-free failure fixtures differ")
    if not synthetic["candidate_roundtrip"].get("all_safe"): raise T0ContractError("T0 model-free candidate safety differs")
    replay=value["terminal_replay_evidence"]; _keys(replay,{"go_written_parsed_validated_reopened","stop_written_parsed_validated_reopened","failure_statuses_validated","late_failures_validated","mapping_insertion_order_invariant","dual_terminal_rejected","unsafe_outputs_rejected"},"T0 model-free replay")
    if not all(item is True for item in replay.values()): raise T0ContractError("T0 model-free terminal replay differs")
    ta=value["tamper_audit"]; _keys(ta,{"schedule_file_sha256","expected_count","results","rejected_count","all_rejected"},"T0 model-free tamper audit")
    if ta["expected_count"]!=98 or ta["rejected_count"]!=98 or not ta["all_rejected"] or len(ta["results"])!=98: raise T0ContractError("T0 model-free tamper count differs")
    for index,(result,row) in enumerate(zip(ta["results"],tampers["rows"],strict=True)):
        if result!={"index":index,"name":row["name"],"rejected":True,"observed_error_type":"T0ContractError"}: raise T0ContractError("T0 model-free tamper result differs")
    safety=value["safety"]; _keys(safety,{"torch_imported","cuda_initialized","tokenizer_loaded","model_loaded","optimizer_constructed","train_scientific_forwards","validation_opens","heldout_opens","network_attempts","output_namespace_fresh","object_census_errors","object_census_uninspectable"},"T0 model-free safety")
    if safety!={"torch_imported":True,"cuda_initialized":False,"tokenizer_loaded":False,"model_loaded":False,"optimizer_constructed":False,"train_scientific_forwards":0,"validation_opens":0,"heldout_opens":0,"network_attempts":0,"output_namespace_fresh":True,"object_census_errors":0,"object_census_uninspectable":0}: raise T0ContractError("T0 model-free safety differs")
    memory=value["memory"]; _keys(memory,{"expected_labels","rows","label_sha256","complete"},"T0 model-free memory")
    if memory["expected_labels"]!=PROOF_MEMORY_LABELS or len(memory["rows"])!=18 or [r for r in memory["rows"]]!=[{"index":i,"label":label,"rss_bytes":r["rss_bytes"]} for i,(label,r) in enumerate(zip(PROOF_MEMORY_LABELS,memory["rows"],strict=True))] or not memory["complete"]: raise T0ContractError("T0 model-free memory differs")
    if memory["label_sha256"]!=sha256_bytes(canonical_json(PROOF_MEMORY_LABELS)): raise T0ContractError("T0 model-free memory hash differs")
    if value["decision_boundary"]!=MODEL_FREE_DECISION: raise T0ContractError("T0 model-free decision boundary differs")
    if capture["counts"]["rows"]!=96 or len(schedule["batches"]["train"])!=320: raise T0ContractError("T0 model-free source schedule differs")

def validate_model_free_failure(value:dict[str,Any])->None:
    expected=(MODEL_FREE_PROOF_KEYS-{"proof_sha256"})|{"error_type","error_message","traceback","stage","execution_progress","audit_errors","failure_sha256"}
    _keys(value,expected,"T0 model-free failure")
    taxonomy={"h_iter_phase1_t0_model_free_proof_incomplete":"T0ModelFreeProofError","h_iter_phase1_t0_model_free_proof_boundary_rejected":"T0ModelFreeProofBoundaryRejected","infrastructure_invalid":"InfrastructureInvalid"}
    if value["schema_version"]!=MODEL_FREE_FAILURE_SCHEMA or value["mechanism"]!=MECHANISM or value["run_identity"]!=PROOF_RUN_ID or taxonomy.get(value["status"])!=value["error_type"]: raise T0ContractError("T0 model-free failure identity differs")
    if value["stage"] not in {"startup","runtime","assets","antecedents","data","module","connectivity","terminal_replay","tamper","candidate_safety","audit","terminal_publication"}: raise T0ContractError("T0 model-free failure stage differs")
    if value["failure_sha256"]!=sha256_bytes(canonical_json({k:v for k,v in value.items() if k!="failure_sha256"})): raise T0ContractError("T0 model-free failure self hash differs")

T0_PROOF_KEYS={"schema_version","status","mechanism","run_identity","execution_commit","mechanism_code_commit","plan_file_sha256","plan_sha256","runtime","asset_audit","antecedent_binding","data_binding","capture_evidence","candidate_initial_state","preconnect_evidence","metric_evidence","gate_evidence","derived_thresholds","candidate_disposition","protected_state","cache_guard","safety","counts","resources","memory","full_freeze","tamper_audit","decision_boundary","proof_sha256"}
T0_COMPLETE_COUNTS={"capture_rows":96,"tokenizer_calls":96,"model_forwards":96,"sequences":2304,"cache_checks":194,"memory_rows":534,"candidate_objects":5,"optimizer_objects":5,"sidecar_forwards":385,"sidecar_backwards":325,"optimizer_steps":320,"cell_calls":773,"precal_presentations":160,"train_presentations":5120,"postcal_presentations":160,"postfit_presentations":320,"tamper_count":98,"validation_opens":0,"heldout_opens":0,"h176_loads":0,"generation_calls":0,"network_attempts":0,"e33_backwards":0,"e33_updates":0,"checkpoints":0,"live_trajectories":0}

def validate_t0_proof(value:dict[str,Any], *, output_inventory:list[str]|None=None)->None:
    _keys(value,T0_PROOF_KEYS,"T0 proof")
    if value["schema_version"]!=PROOF_SCHEMA or value["mechanism"]!=MECHANISM or value["run_identity"]!=RUN_ID or value["status"] not in {"h_iter_phase1_t0_validation_contract_design_authorized","h_iter_phase1_train_calibration_stop"}: raise T0ContractError("T0 proof identity differs")
    if value["proof_sha256"]!=sha256_bytes(canonical_json({k:v for k,v in value.items() if k!="proof_sha256"})): raise T0ContractError("T0 proof self hash differs")
    counts=value["counts"]
    for key,expected in T0_COMPLETE_COUNTS.items():
        if counts.get(key)!=expected: raise T0ContractError(f"T0 count {key} differs")
    go=value["status"]=="h_iter_phase1_t0_validation_contract_design_authorized"
    expected_files=5 if go else 0; expected_threshold=1 if go else 0
    if counts.get("candidate_files")!=expected_files or counts.get("threshold_files")!=expected_threshold: raise T0ContractError("T0 output count differs")
    safety=value["safety"]
    if set(safety)!={"network_attempts","validation_opens","heldout_opens","h176_loads","generation_calls","e33_backwards","e33_optimizer_steps","e33_updates","live_trajectory_count","object_census_errors","object_census_uninspectable","forbidden_inputs_detected"} or safety["forbidden_inputs_detected"]!=[] or any(safety[k]!=0 for k in safety if k!="forbidden_inputs_detected"): raise T0ContractError("T0 safety differs")
    gates=value["gate_evidence"]
    if set(gates)!={"ordered_rows","pass_count","all_pass"} or len(gates["ordered_rows"])!=9 or [r.get("name") for r in gates["ordered_rows"]]!=GATE_NAMES: raise T0ContractError("T0 gates differ")
    expected_pass=9 if go else 8
    if gates["pass_count"]!=expected_pass or gates["all_pass"] is not go or sum(r.get("passed") is True for r in gates["ordered_rows"])!=expected_pass: raise T0ContractError("T0 gate result differs")
    metric=value["metric_evidence"]
    if set(metric)!={"row_records","aggregate_records","postcal_aggregate_sha256","counts"} or len(metric["row_records"])!=640 or len(metric["aggregate_records"])!=135 or metric["counts"]!={"row_records":640,"aggregate_records":135,"precal_presentations":160,"postcal_presentations":160,"postfit_presentations":320}: raise T0ContractError("T0 metric evidence differs")
    if metric["postcal_aggregate_sha256"]!=sha256_bytes(canonical_json([r for r in metric["aggregate_records"] if r["phase"]=="POSTCAL"])): raise T0ContractError("T0 POSTCAL aggregate hash differs")
    disposition=value["candidate_disposition"]
    if set(disposition)!={"arm_order","files_present","threshold_file_present","rows","reusable"} or disposition["arm_order"]!=ARMS or disposition["files_present"]!=(CANDIDATE_FILES if go else []) or disposition["threshold_file_present"] is not go or len(disposition["rows"])!=expected_files or disposition["reusable"] is not go: raise T0ContractError("T0 candidate disposition differs")
    memory=value["memory"]
    if set(memory)!={"schedule_file_sha256","expected_count","rows","label_sha256","complete"} or memory["expected_count"]!=534 or len(memory["rows"])!=534 or not memory["complete"]: raise T0ContractError("T0 memory evidence differs")
    if value["tamper_audit"].get("expected_count")!=98 or value["tamper_audit"].get("rejected_count")!=98 or value["tamper_audit"].get("all_rejected") is not True: raise T0ContractError("T0 tamper evidence differs")
    decision=value["decision_boundary"]
    claim="train_only_matched_sidecar_learning_screen_passed" if go else "train_only_matched_sidecar_learning_screen_stopped"
    if decision.get("claim")!=claim or decision.get("validation_contract_design_authorized") is not go or decision.get("validation_execution_authorized") is not False or decision.get("validation_or_heldout_opened") is not False: raise T0ContractError("T0 decision boundary differs")
    if output_inventory is not None:
        expected=["T0-PROOF.json"] if not go else sorted([*CANDIDATE_FILES,"derived-validation-thresholds.json","T0-PROOF.json"])
        if output_inventory!=expected: raise T0ContractError("T0 output inventory differs")

def validate_t0_failure(value:dict[str,Any])->None:
    statuses={"h_iter_phase1_t0_incomplete":"T0ContractError","h_iter_phase1_t0_capture_mechanism_rejected":"T0CaptureMechanismRejected","h_iter_phase1_t0_exposure_boundary_rejected":"T0ExposureBoundaryRejected","infrastructure_invalid":"InfrastructureInvalid"}
    expected=(T0_PROOF_KEYS-{"proof_sha256"})|{"error_type","error_message","traceback","stage","execution_progress","audit_errors","failure_sha256"}
    _keys(value,expected,"T0 failure")
    if value.get("schema_version")!=FAILURE_SCHEMA or value.get("mechanism")!=MECHANISM or value.get("run_identity")!=RUN_ID or statuses.get(value.get("status"))!=value.get("error_type"): raise T0ContractError("T0 failure taxonomy differs")
    if value.get("failure_sha256")!=sha256_bytes(canonical_json({k:v for k,v in value.items() if k!="failure_sha256"})): raise T0ContractError("T0 failure self hash differs")
    progress=value.get("execution_progress")
    progress_keys={"stage","capture_rows_completed","tokenizer_calls_completed","model_forwards_completed","sequences_completed","cache_checks_completed","candidates_initialized","preconnect_arms_completed","operations_completed","current_operation_index","current_phase","current_arm","current_epoch","current_depth","sidecar_forwards_completed","sidecar_backwards_completed","optimizer_steps_completed","cell_calls_completed","metric_row_records_completed","aggregate_records_completed","gates_evaluated","candidate_files_present","threshold_file_present"}
    if not isinstance(progress,dict) or set(progress)!=progress_keys or progress.get("stage") not in {"startup_pre_model","model_load","capture","model_release","candidate_init","preconnect","precal","train","postcal","postfit","gate_evaluation","candidate_write","postflight_audit","terminal_publication"}: raise T0ContractError("T0 failure progress differs")
    numeric=["capture_rows_completed","tokenizer_calls_completed","model_forwards_completed","sequences_completed","cache_checks_completed","candidates_initialized","preconnect_arms_completed","operations_completed","sidecar_forwards_completed","sidecar_backwards_completed","optimizer_steps_completed","cell_calls_completed","metric_row_records_completed","aggregate_records_completed","gates_evaluated"]
    if any(not isinstance(progress[k],int) or isinstance(progress[k],bool) or progress[k]<0 for k in numeric) or not isinstance(progress["candidate_files_present"],list) or not isinstance(progress["threshold_file_present"],bool): raise T0ContractError("T0 failure counter differs")
