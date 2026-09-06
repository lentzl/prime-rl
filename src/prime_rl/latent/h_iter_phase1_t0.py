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
PARAMETER_SHAPES = [
    {"name":"codec_ln.weight","shape":[2048],"dtype":"torch.float32"},
    {"name":"codec_ln.bias","shape":[2048],"dtype":"torch.float32"},
    {"name":"codec_projection.weight","shape":[128,2048],"dtype":"torch.float32"},
    {"name":"codec_projection.bias","shape":[128],"dtype":"torch.float32"},
    {"name":"self_norm.weight","shape":[128],"dtype":"torch.float32"},
    {"name":"self_norm.bias","shape":[128],"dtype":"torch.float32"},
    {"name":"message_norm.weight","shape":[128],"dtype":"torch.float32"},
    {"name":"message_norm.bias","shape":[128],"dtype":"torch.float32"},
    {"name":"cell_in.weight","shape":[256,256],"dtype":"torch.float32"},
    {"name":"cell_in.bias","shape":[256],"dtype":"torch.float32"},
    {"name":"cell_out.weight","shape":[128,256],"dtype":"torch.float32"},
    {"name":"cell_out.bias","shape":[128],"dtype":"torch.float32"},
    {"name":"post_norm.weight","shape":[128],"dtype":"torch.float32"},
    {"name":"post_norm.bias","shape":[128],"dtype":"torch.float32"},
    {"name":"readout.weight","shape":[4,128],"dtype":"torch.float32"},
    {"name":"readout.bias","shape":[4],"dtype":"torch.float32"},
]
PARAMETER_NAMES = [row["name"] for row in PARAMETER_SHAPES]
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

CACHE_CLASS_RECORDS = [
    {"distribution":"flash-linear-attention==0.5.2","fqcn":name,"module_path":"/home/ubuntu/rlm/prime-rl/.venv/lib/python3.12/site-packages/fla/models/utils.py","module_sha256":"3785d027727370b6eb8da96050109a249254c90caa12a3531a4034cc79f256a1"}
    for name in ("fla.models.utils.Cache","fla.models.utils.FLACache","fla.models.utils.LegacyFLACache")
] + [
    {"distribution":"transformers==5.6.2","fqcn":name,"module_path":"/home/ubuntu/rlm/prime-rl/.venv/lib/python3.12/site-packages/transformers/cache_utils.py","module_sha256":"a51d2cd525f4458941a20cb5b3b8a8d989d8ca5bcd451855a27bb41743fed586"}
    for name in ("transformers.cache_utils.Cache","transformers.cache_utils.DynamicCache","transformers.cache_utils.EncoderDecoderCache","transformers.cache_utils.QuantizedCache","transformers.cache_utils.StaticCache")
]
CACHE_CONFIGURATION_RECORDS = [
    {"source":"model.generation_config","value_before":True,"value_during":False,"value_after":True},
    {"source":"module:model.language_model.config","value_before":False,"value_during":False,"value_after":False},
]

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
"/archive_bindings/mf0/manifest_path","/archive_bindings/mf0/manifest_internal_sha256","/archive_bindings/mf0/proof_path","/archive_bindings/mf0/proof_internal_sha256","/archive_bindings/mf0/launcher_log_path","/archive_bindings/mf0/exit_path","/archive_bindings/cap0_r1/manifest_path","/archive_bindings/cap0_r1/manifest_internal_sha256","/archive_bindings/cap0_r1/proof_path","/archive_bindings/cap0_r1/proof_internal_sha256","/archive_bindings/cap0_r1/launcher_log_path","/archive_bindings/cap0_r1/exit_path","/data_contract/train_bank_path","/data_contract/partition_path","/data_contract/schedule_path","/training_contract/candidate_contract_path","/capture_contract/source_contract_path","/metric_gate_contract/metric_contract_path","/metric_gate_contract/threshold_contract_path","/data_binding/calibration_row_ids/0","/rows/0","/rows/0/row_sha256","/candidate_initial_state/action_order/0","/candidate_initial_state/arm_order/0","/batches/train/0/epoch","/depth_order/0","/batches/train/0/row_ids/0","/remote_paths/e33","/protected_state/e33_disk_tree_before","/protected_state/e33_state_after","/model_contract/metadata_sha256/tokenizer.json","/capture/tokenizer_max_length","/capture/tokenizer_truncation","/capture_evidence/rows/0/hidden_dtype","/capture/hidden_selection","/capture/use_cache","/capture/output_hidden_states","/capture/logits_to_keep","/cache_guard/class_records/0","/capture_evidence/rows/0/input_ids_sha256","/cache_guard/dynamic_cache_negative_trips","/cache_guard/returned_pkv_count","/counts/capture_rows","/protected_state/h176_loaded","/candidate_initial_state/seed_u64_be","/candidate_initial_state/rows/4/initial_state_sha256","/candidate_initial_state/rows/0/parameter_names/0","/candidate_initial_state/rows/0/parameter_shapes/0/shape/0","/candidate_initial_state/rows/0/parameter_count","/candidate_initial_state/rows/0/dtype","/candidate_initial_state/rows/0/device","/modules/tf32","/modules/dropout","/safety/forbidden_inputs_detected","/safety/forbidden_inputs_detected","/modules/readout","/modules/arm_semantics/STATIC","/modules/arm_semantics/FFN","/modules/arm_semantics/FIXED_T4","/modules/arm_semantics/RESET_K","/modules/arm_semantics/REC_K","/preconnect_evidence/row_id","/preconnect_evidence/rows/4/cell_grad_l2","/preconnect_evidence/rows/4/state_after_sha256","/training/optimizer/type","/training/optimizer/lr","/training/optimizer/betas/1","/training/optimizer/eps","/training/optimizer/weight_decay","/training/optimizer/gradient_clip_global_norm","/training/optimizer/zero_grad_set_to_none","/counts/optimizer_steps","/training/shuffle","/training/objective","/metric_evidence/row_records/0/phase","/metrics/nll_formula","/metrics/prediction_formula","/metrics/aggregation_order/0","/gate_evidence/ordered_rows/0/rhs","/gate_evidence/ordered_rows/1/rhs","/gate_evidence/ordered_rows/2/rhs","/gate_evidence/ordered_rows/3/rhs","/gate_evidence/ordered_rows/4/rhs","/gate_evidence/ordered_rows/5/rhs","/gate_evidence/ordered_rows/6/rhs","/gate_evidence/ordered_rows/7/rhs","/gate_evidence/ordered_rows/8/rhs","/decision_boundary/validation_contract_design_authorized","/safety/validation_opens","/safety/heldout_opens","/safety/generation_calls","/protected_state/e33_state_after","/candidate_files/0","/memory/rows/533","/resources/max_reserved_bytes","/status","/proof_sha256","/status"]

def build_tamper_schedule() -> dict[str,Any]:
    if len(_TAMPER_NAMES)!=98 or len(_TAMPER_TARGETS)!=98: raise T0ContractError("T0 tamper literal count differs")
    fixtures=["plan_copy"]*19+["terminal_copy","capture_schedule_copy","capture_schedule_copy","terminal_copy","terminal_copy","schedule_copy","schedule_copy","schedule_copy","plan_copy","terminal_copy","terminal_copy","plan_copy"]+["contract_copy","contract_copy","terminal_copy","contract_copy","contract_copy","contract_copy","contract_copy","terminal_copy","terminal_copy","terminal_copy","terminal_copy","terminal_copy","terminal_copy","terminal_copy","terminal_copy","terminal_copy","terminal_copy","terminal_copy"]+["contract_copy","contract_copy","terminal_copy","terminal_copy","contract_copy","contract_copy","contract_copy","contract_copy","contract_copy","terminal_copy","terminal_copy","terminal_copy"]+["contract_copy"]*10+["terminal_copy","contract_copy","contract_copy","contract_copy"]+["terminal_copy"]*10+["go_output_dir","terminal_copy","terminal_copy","terminal_copy","terminal_copy","failure_copy"]
    # Keep fixture assignment explicit at the three special late boundaries.
    fixtures=fixtures[:98]
    operators=["raw_append_hex","replace","raw_append_hex","replace","raw_append_hex","raw_replace_utf8","raw_append_hex","replace","raw_append_hex","replace","raw_append_hex","raw_replace_utf8",*["raw_append_hex"]*7,"replace","swap","replace","swap","swap","replace","swap","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","delete","replace","replace","replace","replace","replace","replace","replace","swap","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","swap","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","replace","swap","delete","replace","replace","replace","replace"]
    operands=["00","0"*64,"00","0"*64,"00","1\n","00","0"*64,"00","0"*64,"00","1\n","00","00","00","00","00","00","00","hi_575cf5825abbc3a1","/rows/1","0"*64,"/candidate_initial_state/action_order/1","/candidate_initial_state/arm_order/1",1,"/depth_order/1","hi_b68d66d7d1372aa8",H176_PATH,"0"*64,"0"*64,"0"*64,127,True,"float32","hidden_states[-1][:,-2,:]",True,False,0,None,"0"*64,0,1,95,True,17594986156060532330,"0"*64,"/candidate_initial_state/rows/0/parameter_names/1",127,366339,"float16","cpu",True,True,["depth_or_K_scalar"],["global_pool"],"Linear(128,4,bias=True) on all-node pooled state","C(z,zeros)","C(z,adjacency_message)","persistent C exactly 3 times","persistent C exactly K times","K reset calls","hi_575cf5825abbc3a1",0.0,"0"*64,"SGD",0.002,0.999,1e-7,0.0,0.5,False,319,True,"sum 4-class cross entropy","POSTCAL","-log_softmax(logits.float(),dim=-1)[action_index]","argmax with highest-index tie","/metrics/aggregation_order/3",[21,32],[3,32],[1,32],[3,8],[2,8],0.04,0.01,0.8,[47,64],True,1,1,1,"0"*64,"/candidate_files/1",None,42949672961,"h_iter_phase1_t0_validation_authorized","0"*64,"infrastructure_invalid"]
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

def validate_candidate_artifact(path:Any,row:dict[str,Any])->dict[str,Any]:
    from pathlib import Path
    path=Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size!=row.get("bytes") or sha256_bytes(path.read_bytes())!=row.get("file_sha256"): raise T0ContractError("T0 candidate file differs")
    try:
        import torch
        from safetensors.torch import load_file
        loaded=load_file(str(path),device="cpu")
    except Exception as error:
        raise T0ContractError("T0 candidate safe reload failed") from error
    if set(loaded)!=set(PARAMETER_NAMES) or any(list(loaded[item["name"]].shape)!=item["shape"] or str(loaded[item["name"]].dtype)!="torch.float32" or not bool(torch.isfinite(loaded[item["name"]]).all()) for item in PARAMETER_SHAPES) or module_state_sha256(torch,loaded)!=row.get("module_state_sha256"):
        raise T0ContractError("T0 candidate safe reload differs")
    return loaded

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
def digest_string(value:Any,length:int=64)->bool: return isinstance(value,str) and len(value)==length and all(c in "0123456789abcdef" for c in value)

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
    archives=value["archive_bindings"]
    if set(archives)!={"mf0","cap0_r1","antecedent_manifest_path","antecedent_manifest_file_sha256","antecedent_manifest_internal_sha256"} or archives["mf0"]!=MF0_BINDING or archives["cap0_r1"]!=CAP0_BINDING or archives["antecedent_manifest_path"]!=f"{ARTIFACT_DIR}/t0-antecedent-evidence-manifest.json" or not digest_string(archives["antecedent_manifest_file_sha256"]) or not digest_string(archives["antecedent_manifest_internal_sha256"]): raise T0ContractError("T0 archive binding differs")
    remote=value["remote_paths"]
    if remote!={"repo":"/home/ubuntu/rlm/worktrees/q35-2b-latent-workspace-v1","shared_project":"/home/ubuntu/rlm/prime-rl","shared_python":"/home/ubuntu/rlm/prime-rl/.venv/bin/python3","e33":E33_PATH,"h176":H176_PATH,"physical_gpu":"0","visible_device":"cuda:0","proof_output":PROOF_OUTPUT_ROOT,"t0_output":OUTPUT_ROOT}: raise T0ContractError("T0 remote paths differ")
    runtime=value["runtime"]
    if runtime!={"python":"3.12.14","sys_executable":"/home/ubuntu/rlm/prime-rl/.venv/bin/python3","sys_prefix":"/home/ubuntu/rlm/prime-rl/.venv","transformers":"5.6.2","tokenizers":"0.22.2","torch_distribution":"2.11.0+cu128","torch_runtime":"2.11.0+cu128","flash_linear_attention":"0.5.2","gpu_model":"NVIDIA RTX A6000","shared_project_pyproject_sha256":"504907808f992f1e6883f54c2695a4814ae77d6b80814239cbfc98d81a543656","shared_project_uv_lock_sha256":"fca5fa6183345b5b68974078c38d58e0320f79eef13a695af11ceab12fdf36d5"}: raise T0ContractError("T0 plan runtime differs")
    model=value["model_contract"]
    if model!={"e33_path":E33_PATH,"e33_tree_sha256":E33_TREE_SHA256,"e33_state_sha256":E33_STATE_SHA256,"metadata_sha256":METADATA_SHA256,"dtype":"torch.bfloat16","attention":"eager","eval":True,"frozen":True,"h176_path":H176_PATH,"h176_tree_sha256":H176_TREE_SHA256,"h176_loaded":False}: raise T0ContractError("T0 model plan differs")
    data=value["data_contract"]
    if set(data)!={"train_bank_path","train_bank_file_sha256","train_bank_internal_sha256","partition_path","partition_file_sha256","schedule_path","schedule_file_sha256","validation_open_allowed","heldout_open_allowed"} or data["train_bank_path"]!=TRAIN_BANK_PATH or data["train_bank_file_sha256"]!=TRAIN_BANK_FILE_SHA256 or data["train_bank_internal_sha256"]!=TRAIN_BANK_INTERNAL_SHA256 or data["partition_path"]!=PARTITION_PATH or data["schedule_path"]!=SCHEDULE_PATH or data["validation_open_allowed"] is not False or data["heldout_open_allowed"] is not False: raise T0ContractError("T0 data firewall differs")
    capture=value["capture_contract"]
    if capture.get("source_contract_path")!=CAPTURE_CONTRACT_PATH or capture.get("capture_schedule_path")!=f"{ARTIFACT_DIR}/t0-capture-schedule.json" or [capture.get(k) for k in ("tokenizer_calls","model_forwards","sequences","cache_checks","memory_only")]!=[96,96,2304,194,True]: raise T0ContractError("T0 capture plan differs")
    training=value["training_contract"]
    if training.get("candidate_contract_path")!=CANDIDATE_CONTRACT_PATH or training.get("arm_order")!=ARMS or training.get("action_order")!=ACTIONS or [training.get(k) for k in ("operation_count","sidecar_forwards","sidecar_backwards","optimizer_steps","cell_calls","updates_per_arm")]!=[385,385,325,320,773,64]: raise T0ContractError("T0 training plan differs")
    metric=value["metric_gate_contract"]
    if metric.get("metric_contract_path")!=METRIC_CONTRACT_PATH or metric.get("threshold_contract_path")!=THRESHOLD_CONTRACT_PATH or metric.get("gate_count")!=9 or metric.get("all_required") is not True: raise T0ContractError("T0 metric plan differs")
    if proof_input:
        if value["model_free_proof_binding"] is not None or value["execution_authorization"]!={"model_free_proof_eligible_after_independent_review":True,"t0_preflight_eligible":False,"t0_full_authorized":False,"model_load_authorized":False,"gpu_authorized":False,"training_authorized":False} or value["decision_boundary"]!={"claim":"model_free_proof_design_only","train_bank_schema_open_allowed":True,"train_scientific_model_exposure_allowed":False,"validation_or_heldout_opened":False,"model_or_gpu_authorized":False,"t0_training_authorized":False,"admission":False,"nomination":False,"promotion":False,"four_live_floor_unchanged":True}: raise T0ContractError("T0 proof authorization differs")
    else:
        if value["execution_authorization"]!={"model_free_proof_eligible_after_independent_review":False,"t0_preflight_eligible_after_independent_review":True,"t0_full_authorized":False,"model_load_authorized":False,"gpu_authorized":False,"training_authorized":False} or value["decision_boundary"]!={"claim":"model_free_proof_validated_t0_preregistered","train_bank_schema_open_allowed":True,"train_scientific_model_exposure_allowed":False,"validation_or_heldout_opened":False,"model_or_gpu_authorized":False,"t0_training_authorized":False,"admission":False,"nomination":False,"promotion":False,"four_live_floor_unchanged":True}: raise T0ContractError("T0 final authorization differs")
        binding=value["model_free_proof_binding"]
        if not isinstance(binding,dict) or set(binding)!={"proof_execution_commit","proof_plan_path","proof_plan_file_sha256","proof_plan_sha256","evidence_manifest_path","evidence_manifest_file_sha256","evidence_manifest_internal_sha256","proof_path","proof_file_sha256","proof_internal_sha256","stdout_log_path","stdout_log_sha256","stderr_log_path","stderr_log_sha256","exit_path","exit_file_sha256","validator_replay_passed"} or binding["validator_replay_passed"] is not True: raise T0ContractError("T0 proof binding absent")

def validate_model_free_proof(value:dict[str,Any],plan:dict[str,Any],capture:dict[str,Any],schedule:dict[str,Any],tampers:dict[str,Any])->None:
    _keys(value,MODEL_FREE_PROOF_KEYS,"T0 model-free proof")
    if value["schema_version"]!=MODEL_FREE_PROOF_SCHEMA or value["status"]!="h_iter_phase1_t0_model_free_mechanism_validated" or value["mechanism"]!=MECHANISM or value["run_identity"]!=PROOF_RUN_ID: raise T0ContractError("T0 model-free proof identity differs")
    if value["mechanism_code_commit"]!=plan["mechanism_code_commit"] or value["plan_sha256"]!=plan["plan_sha256"]: raise T0ContractError("T0 model-free authority differs")
    if value["proof_sha256"]!=sha256_bytes(canonical_json({k:v for k,v in value.items() if k!="proof_sha256"})): raise T0ContractError("T0 model-free self hash differs")
    runtime=value["runtime"]; _keys(runtime,{"python","sys_executable","sys_prefix","torch","shared_pyproject_sha256","shared_uv_lock_sha256","cuda_visible_devices","cuda_initialized_before","cuda_initialized_after"},"T0 model-free runtime")
    if runtime!={"python":"3.12.14","sys_executable":"/home/ubuntu/rlm/prime-rl/.venv/bin/python3","sys_prefix":"/home/ubuntu/rlm/prime-rl/.venv","torch":"2.11.0+cu128","shared_pyproject_sha256":"504907808f992f1e6883f54c2695a4814ae77d6b80814239cbfc98d81a543656","shared_uv_lock_sha256":"fca5fa6183345b5b68974078c38d58e0320f79eef13a695af11ceab12fdf36d5","cuda_visible_devices":"","cuda_initialized_before":False,"cuda_initialized_after":False}: raise T0ContractError("T0 model-free CUDA boundary differs")
    audit=value["asset_audit"]; _keys(audit,{"target_count","pre_entries","pre_sha256","post_entries","post_sha256","all_exact"},"T0 model-free asset audit")
    if audit["target_count"]!=31 or len(audit["pre_entries"])!=31 or not audit["all_exact"] or audit["pre_entries"]!=audit["post_entries"] or audit["pre_sha256"]!=audit["post_sha256"] or audit["pre_sha256"]!=sha256_bytes(canonical_json(audit["pre_entries"])): raise T0ContractError("T0 model-free asset closure differs")
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
    if not synthetic["candidate_roundtrip"].get("all_safe") or synthetic["candidate_roundtrip"].get("unsafe_rejections")!=["wrong_name","wrong_shape","wrong_dtype","nonfinite","unsafe_format","symlink"]: raise T0ContractError("T0 model-free candidate safety differs")
    replay=value["terminal_replay_evidence"]; _keys(replay,{"go_written_parsed_validated_reopened","stop_written_parsed_validated_reopened","failure_statuses_validated","late_failures_validated","mapping_insertion_order_invariant","dual_terminal_rejected","unsafe_outputs_rejected"},"T0 model-free replay")
    if not all(item is True for item in replay.values()): raise T0ContractError("T0 model-free terminal replay differs")
    ta=value["tamper_audit"]; _keys(ta,{"schedule_file_sha256","expected_count","results","rejected_count","all_rejected"},"T0 model-free tamper audit")
    if ta["expected_count"]!=98 or ta["rejected_count"]!=98 or not ta["all_rejected"] or len(ta["results"])!=98: raise T0ContractError("T0 model-free tamper count differs")
    for index,(result,row) in enumerate(zip(ta["results"],tampers["rows"])):
        if result!={"index":index,"name":row["name"],"rejected":True,"observed_error_type":"T0ContractError"}: raise T0ContractError("T0 model-free tamper result differs")
    safety=value["safety"]; _keys(safety,{"torch_imported","cuda_initialized","tokenizer_loaded","model_loaded","optimizer_constructed","train_scientific_forwards","validation_opens","heldout_opens","network_attempts","output_namespace_fresh","object_census_errors","object_census_uninspectable"},"T0 model-free safety")
    if safety!={"torch_imported":True,"cuda_initialized":False,"tokenizer_loaded":False,"model_loaded":False,"optimizer_constructed":False,"train_scientific_forwards":0,"validation_opens":0,"heldout_opens":0,"network_attempts":0,"output_namespace_fresh":True,"object_census_errors":0,"object_census_uninspectable":0}: raise T0ContractError("T0 model-free safety differs")
    memory=value["memory"]; _keys(memory,{"expected_labels","rows","label_sha256","complete"},"T0 model-free memory")
    if memory["expected_labels"]!=PROOF_MEMORY_LABELS or len(memory["rows"])!=18 or [r for r in memory["rows"]]!=[{"index":i,"label":label,"rss_bytes":r["rss_bytes"]} for i,(label,r) in enumerate(zip(PROOF_MEMORY_LABELS,memory["rows"]))] or not memory["complete"]: raise T0ContractError("T0 model-free memory differs")
    if memory["label_sha256"]!=sha256_bytes(canonical_json(PROOF_MEMORY_LABELS)): raise T0ContractError("T0 model-free memory hash differs")
    resources=value["resources"]; _keys(resources,{"minimum_ram_gib","minimum_disk_gib","maximum_artifact_bytes","timing","observed_rss_peak_bytes","artifact_bytes"},"T0 model-free resources")
    timing={"outer":1800,"startup":120,"compute":1050,"compute_alarm":1049,"audit":240,"audit_alarm":239,"failure":180,"failure_alarm":179,"terminal":60,"terminal_alarm":59,"postexit":60,"success_entry":1290,"compute_failure_entry":1230,"audit_failure_entry":1470,"prior_terminal_failure_entry":1530,"worst_external":1770,"reserve":30}
    if resources["minimum_ram_gib"]!=8 or resources["minimum_disk_gib"]!=8 or resources["maximum_artifact_bytes"]!=16777216 or resources["timing"]!=timing or not isinstance(resources["observed_rss_peak_bytes"],int) or resources["observed_rss_peak_bytes"]<0 or not isinstance(resources["artifact_bytes"],int) or not 0<=resources["artifact_bytes"]<=16777216: raise T0ContractError("T0 model-free resources differ")
    freeze=value["full_freeze"]; _keys(freeze,{"head_before","head_after","tree_before","tree_after","clean_before","clean_after","assets_pre_sha256","assets_post_sha256","complete"},"T0 model-free freeze")
    if freeze["head_before"]!=value["execution_commit"] or freeze["head_before"]!=freeze["head_after"] or freeze["tree_before"]!=value["tree_sha256"] or freeze["tree_before"]!=freeze["tree_after"] or freeze["clean_before"] is not True or freeze["clean_after"] is not True or freeze["assets_pre_sha256"]!=audit["pre_sha256"] or freeze["assets_post_sha256"]!=audit["post_sha256"] or freeze["complete"] is not True: raise T0ContractError("T0 model-free freeze differs")
    if value["decision_boundary"]!=MODEL_FREE_DECISION: raise T0ContractError("T0 model-free decision boundary differs")
    if capture["counts"]["rows"]!=96 or len(schedule["batches"]["train"])!=320: raise T0ContractError("T0 model-free source schedule differs")

def validate_model_free_failure(value:dict[str,Any])->None:
    expected=(MODEL_FREE_PROOF_KEYS-{"proof_sha256"})|{"error_type","error_message","traceback","stage","execution_progress","audit_errors","failure_sha256"}
    _keys(value,expected,"T0 model-free failure")
    taxonomy={"h_iter_phase1_t0_model_free_proof_incomplete":"T0ModelFreeProofError","h_iter_phase1_t0_model_free_proof_boundary_rejected":"T0ModelFreeProofBoundaryRejected","infrastructure_invalid":"InfrastructureInvalid"}
    if value["schema_version"]!=MODEL_FREE_FAILURE_SCHEMA or value["mechanism"]!=MECHANISM or value["run_identity"]!=PROOF_RUN_ID or taxonomy.get(value["status"])!=value["error_type"]: raise T0ContractError("T0 model-free failure identity differs")
    stages=["startup","runtime","assets","antecedents","data","module","connectivity","terminal_replay","tamper","candidate_safety","audit","terminal_publication"]
    if value["stage"] not in stages: raise T0ContractError("T0 model-free failure stage differs")
    progress=value["execution_progress"]
    if not isinstance(progress,dict) or set(progress)!={"stage","memory_rows_completed"} or progress["stage"]!=value["stage"] or not isinstance(progress["memory_rows_completed"],int) or isinstance(progress["memory_rows_completed"],bool) or not 0<=progress["memory_rows_completed"]<=18: raise T0ContractError("T0 model-free failure progress differs")
    memory=value["memory"]
    if not isinstance(memory,dict) or set(memory)!={"expected_labels","rows","label_sha256","complete"} or memory["expected_labels"]!=PROOF_MEMORY_LABELS or memory["label_sha256"]!=sha256_bytes(canonical_json(PROOF_MEMORY_LABELS)) or not isinstance(memory["rows"],list) or len(memory["rows"])!=progress["memory_rows_completed"] or memory["complete"] is not (len(memory["rows"])==18): raise T0ContractError("T0 model-free failure memory differs")
    for index,row in enumerate(memory["rows"]):
        if row.get("index")!=index or row.get("label")!=PROOF_MEMORY_LABELS[index] or set(row)!={"index","label","rss_bytes"} or not isinstance(row["rss_bytes"],int) or row["rss_bytes"]<0: raise T0ContractError("T0 model-free failure memory prefix differs")
    rank=stages.index(value["stage"])
    for minimum,field in ((3,"asset_audit"),(4,"antecedent_binding"),(5,"data_binding"),(6,"module_evidence"),(8,"synthetic_evidence"),(8,"terminal_replay_evidence"),(9,"tamper_audit")):
        if rank>=minimum and value[field] is None: raise T0ContractError(f"T0 model-free failure {field} missing")
    safety=value["safety"]
    boundary_positive=False; census_degraded=False
    if safety is not None:
        _keys(safety,{"torch_imported","cuda_initialized","tokenizer_loaded","model_loaded","optimizer_constructed","train_scientific_forwards","validation_opens","heldout_opens","network_attempts","output_namespace_fresh","object_census_errors","object_census_uninspectable"},"T0 model-free failure safety")
        counters=[safety[k] for k in ("train_scientific_forwards","validation_opens","heldout_opens","network_attempts","object_census_errors","object_census_uninspectable")]
        if any(not isinstance(item,int) or isinstance(item,bool) or item<0 for item in counters): raise T0ContractError("T0 model-free failure safety count differs")
        boundary_positive=bool(safety["cuda_initialized"] or safety["tokenizer_loaded"] or safety["model_loaded"] or safety["train_scientific_forwards"] or safety["validation_opens"] or safety["heldout_opens"] or safety["network_attempts"])
        census_degraded=bool(safety["object_census_errors"] or safety["object_census_uninspectable"])
    if boundary_positive:
        if value["status"]!="h_iter_phase1_t0_model_free_proof_boundary_rejected": raise T0ContractError("T0 model-free positive boundary evidence masked")
    elif value["status"]=="h_iter_phase1_t0_model_free_proof_boundary_rejected": raise T0ContractError("T0 model-free boundary rejection lacks evidence")
    if census_degraded and value["status"]!="infrastructure_invalid": raise T0ContractError("T0 model-free census degradation masked")
    if value["status"]=="infrastructure_invalid" and (not isinstance(value["audit_errors"],list) or not value["audit_errors"]): raise T0ContractError("T0 model-free infrastructure evidence missing")
    if value["status"]!="infrastructure_invalid" and value["audit_errors"]!=[]: raise T0ContractError("T0 model-free non-infrastructure audit errors differ")
    if value["failure_sha256"]!=sha256_bytes(canonical_json({k:v for k,v in value.items() if k!="failure_sha256"})): raise T0ContractError("T0 model-free failure self hash differs")

T0_PROOF_KEYS={"schema_version","status","mechanism","run_identity","execution_commit","mechanism_code_commit","plan_file_sha256","plan_sha256","runtime","asset_audit","antecedent_binding","data_binding","capture_evidence","candidate_initial_state","preconnect_evidence","metric_evidence","gate_evidence","derived_thresholds","candidate_disposition","protected_state","cache_guard","safety","counts","resources","memory","full_freeze","tamper_audit","decision_boundary","proof_sha256"}
T0_COMPLETE_COUNTS={"capture_rows":96,"tokenizer_calls":96,"model_forwards":96,"sequences":2304,"cache_checks":194,"memory_rows":534,"candidate_objects":5,"optimizer_objects":5,"sidecar_forwards":385,"sidecar_backwards":325,"optimizer_steps":320,"cell_calls":773,"precal_presentations":160,"train_presentations":5120,"postcal_presentations":160,"postfit_presentations":320,"tamper_count":98,"validation_opens":0,"heldout_opens":0,"h176_loads":0,"generation_calls":0,"network_attempts":0,"e33_backwards":0,"e33_updates":0,"checkpoints":0,"live_trajectories":0}
T0_COUNT_KEYS=set(T0_COMPLETE_COUNTS)|{"candidate_files","threshold_files"}
T0_RUNTIME={"python":"3.12.14","sys_executable":"/home/ubuntu/rlm/prime-rl/.venv/bin/python3","sys_prefix":"/home/ubuntu/rlm/prime-rl/.venv","transformers":"5.6.2","tokenizers":"0.22.2","torch_distribution":"2.11.0+cu128","torch_runtime":"2.11.0+cu128","flash_linear_attention":"0.5.2","gpu_name":"NVIDIA RTX A6000","physical_gpu":"0","visible_device":"cuda:0","shared_project_pyproject_sha256":"504907808f992f1e6883f54c2695a4814ae77d6b80814239cbfc98d81a543656","shared_project_uv_lock_sha256":"fca5fa6183345b5b68974078c38d58e0320f79eef13a695af11ceab12fdf36d5"}
T0_ASSET_ROW_KEYS={"path","sha256","bytes"}
T0_RESOURCE_KEYS={"gpu_name","physical_gpu","visible_device","free_gpu_bytes_pre","available_ram_bytes_pre","free_disk_bytes_pre","max_allocated_bytes","max_reserved_bytes","artifact_bytes","timing"}
T0_TIMING_KEYS={"outer_seconds","startup_seconds","compute_seconds","audit_seconds","failure_seconds","terminal_seconds","postexit_seconds","alarm_safety_margin_seconds","compute_enter_ns","compute_exit_ns","compute_duration_ns","audit_enter_ns","audit_exit_ns","audit_duration_ns","failure_enter_ns","failure_exit_ns","failure_duration_ns","terminal_enter_ns","prepublication_elapsed_ns"}
T0_DECISION_BASE={"candidate_modules_updated":True,"protected_models_updated":False,"validation_execution_authorized":False,"validation_or_heldout_opened":False,"live_trajectory_count":0,"admission":False,"nomination":False,"promotion":False,"four_live_floor_unchanged":True}
T0_FAILURE_MEMORY_BOUNDS={
    "startup_pre_model":(0,5),"model_load":(5,6),"capture":(6,201),"model_release":(201,203),
    "candidate_init":(203,204),"preconnect":(204,205),"precal":(205,206),"train":(206,527),
    "postcal":(527,528),"postfit":(528,529),"gate_evaluation":(529,531),"candidate_write":(530,531),
    "postflight_audit":(531,534),"terminal_publication":(534,534),
}

def validate_failure_memory_stage(stage:str,row_count:int)->None:
    if stage not in T0_FAILURE_MEMORY_BOUNDS or not isinstance(row_count,int) or isinstance(row_count,bool): raise T0ContractError("T0 failure memory stage schema differs")
    low,high=T0_FAILURE_MEMORY_BOUNDS[stage]
    if not low<=row_count<=high: raise T0ContractError("T0 failure memory stage boundary differs")

def validate_capture_failure_microstate(stage:str,captures:int,tokenizers:int,forwards:int,cache_checks:int,memory_rows:int)->None:
    state=(captures,tokenizers,forwards,cache_checks,memory_rows)
    if stage in {"startup_pre_model","model_load"}:
        if any(state[:4]): raise T0ContractError("T0 failure pre-capture microstate differs")
        return
    if stage!="capture":
        if state[:4]!=(96,96,96,194): raise T0ContractError("T0 failure completed capture microstate differs")
        return
    # This is the literal observable state machine in run_h_iter_phase1_t0_v1.py:
    # tokenizer counter -> PRE ledger -> CACHE_PRE -> model/forward counter ->
    # CACHE_POST -> POST ledger -> capture counter.  Enumerating every boundary
    # prevents neighboring, mutually impossible counter combinations from being
    # accepted by independent terminal replay.
    legal_states={(0,0,0,0,6),(0,0,0,1,6),(0,0,0,1,7),(0,0,0,1,8)}
    for completed in range(96):
        checks=1+2*completed; memory=8+2*completed
        if completed: legal_states.add((completed,completed,completed,checks,memory))
        legal_states.update({
            (completed,completed+1,completed,checks,memory),
            (completed,completed+1,completed,checks,memory+1),
            (completed,completed+1,completed,checks+1,memory+1),
            (completed,completed+1,completed+1,checks+1,memory+1),
            (completed,completed+1,completed+1,checks+2,memory+1),
            (completed,completed+1,completed+1,checks+2,memory+2),
        })
    legal_states.update({
        (96,96,96,193,200),             # final capture counter advanced
        (96,96,96,193,201),             # CACHE_GUARD_EXITED ledger, before EXIT
        (96,96,96,194,201),             # CACHE_EXIT completed
    })
    if state not in legal_states: raise T0ContractError("T0 failure capture microstate differs")

def capture_failure_evidence_row_count(rows:list[dict[str,Any]],captures:int,forwards:int)->int:
    if len(rows)==captures+1 and forwards==captures+1 and rows[-1].get("finite") is True: return len(rows)
    return captures

def validate_t0_proof(value:dict[str,Any], *, partition:dict[str,Any]|None=None, capture_schedule:dict[str,Any]|None=None, training_schedule:dict[str,Any]|None=None, memory_schedule:dict[str,Any]|None=None, output_inventory:list[str]|None=None, output_dir:Any|None=None)->None:
    if partition is None or capture_schedule is None or training_schedule is None or memory_schedule is None: raise T0ContractError("T0 proof frozen validation sources missing")
    _keys(value,T0_PROOF_KEYS,"T0 proof")
    if value["schema_version"]!=PROOF_SCHEMA or value["mechanism"]!=MECHANISM or value["run_identity"]!=RUN_ID or value["status"] not in {"h_iter_phase1_t0_validation_contract_design_authorized","h_iter_phase1_train_calibration_stop"}: raise T0ContractError("T0 proof identity differs")
    if value["proof_sha256"]!=sha256_bytes(canonical_json({k:v for k,v in value.items() if k!="proof_sha256"})): raise T0ContractError("T0 proof self hash differs")
    counts=value["counts"]
    for key,expected in T0_COMPLETE_COUNTS.items():
        if counts.get(key)!=expected: raise T0ContractError(f"T0 count {key} differs")
    go=value["status"]=="h_iter_phase1_t0_validation_contract_design_authorized"
    expected_files=5 if go else 0; expected_threshold=1 if go else 0
    if counts.get("candidate_files")!=expected_files or counts.get("threshold_files")!=expected_threshold: raise T0ContractError("T0 output count differs")
    runtime=value["runtime"]
    _keys(runtime,{"python","sys_executable","sys_prefix","transformers","tokenizers","torch_distribution","torch_runtime","flash_linear_attention","gpu_name","physical_gpu","visible_device","shared_project_pyproject_sha256","shared_project_uv_lock_sha256"},"T0 runtime")
    if runtime!={"python":"3.12.14","sys_executable":"/home/ubuntu/rlm/prime-rl/.venv/bin/python3","sys_prefix":"/home/ubuntu/rlm/prime-rl/.venv","transformers":"5.6.2","tokenizers":"0.22.2","torch_distribution":"2.11.0+cu128","torch_runtime":"2.11.0+cu128","flash_linear_attention":"0.5.2","gpu_name":"NVIDIA RTX A6000","physical_gpu":"0","visible_device":"cuda:0","shared_project_pyproject_sha256":"504907808f992f1e6883f54c2695a4814ae77d6b80814239cbfc98d81a543656","shared_project_uv_lock_sha256":"fca5fa6183345b5b68974078c38d58e0320f79eef13a695af11ceab12fdf36d5"}: raise T0ContractError("T0 runtime differs")
    audit=value["asset_audit"]; _keys(audit,{"target_count","pre_entries","pre_sha256","post_entries","post_sha256","all_exact"},"T0 asset audit")
    if audit["target_count"]!=38 or len(audit["pre_entries"])!=38 or audit["pre_entries"]!=audit["post_entries"] or audit["pre_sha256"]!=audit["post_sha256"] or audit["pre_sha256"]!=sha256_bytes(canonical_json(audit["pre_entries"])) or audit["all_exact"] is not True: raise T0ContractError("T0 asset audit differs")
    if any(set(row)!={"path","sha256","bytes"} or not isinstance(row["path"],str) or not digest_string(row["sha256"]) or not isinstance(row["bytes"],int) or isinstance(row["bytes"],bool) or row["bytes"]<0 for row in audit["pre_entries"]): raise T0ContractError("T0 asset row differs")
    antecedent=value["antecedent_binding"]; _keys(antecedent,{"manifest_path","manifest_file_sha256","manifest_internal_sha256","mf0_archive_exact","cap0_r1_archive_exact"},"T0 antecedent")
    if antecedent["mf0_archive_exact"] is not True or antecedent["cap0_r1_archive_exact"] is not True: raise T0ContractError("T0 antecedent differs")
    data=value["data_binding"]; _keys(data,{"train_bank_path","train_bank_file_sha256","train_bank_internal_sha256","source_rows","fit_rows","calibration_rows","fit_row_ids","calibration_row_ids","fit_calibration_intersection","complete_train_union","validation_open_count","heldout_open_count"},"T0 data")
    fit_ids=data["fit_row_ids"]; cal_ids=data["calibration_row_ids"]
    if data["train_bank_path"]!=TRAIN_BANK_PATH or data["train_bank_file_sha256"]!=TRAIN_BANK_FILE_SHA256 or data["train_bank_internal_sha256"]!=TRAIN_BANK_INTERNAL_SHA256 or (data["source_rows"],data["fit_rows"],data["calibration_rows"])!=(96,64,32) or fit_ids!=[r["row_id"] for r in partition["fit_rows"]] or cal_ids!=[r["row_id"] for r in partition["calibration_rows"]] or data["fit_calibration_intersection"]!=[] or len(set(fit_ids)|set(cal_ids))!=96 or data["complete_train_union"] is not True or data["validation_open_count"] or data["heldout_open_count"]: raise T0ContractError("T0 data differs")
    capture=value["capture_evidence"]; _keys(capture,{"schedule_sha256","rows","counts","aggregate_sha256"},"T0 capture")
    row_keys={"capture_index","row_id","row_sha256","receiver_input_sha256","node_count","token_count_min","token_count_max","input_ids_sha256","attention_mask_sha256","hidden_shape","hidden_dtype","hidden_sha256","finite"}
    if capture["schedule_sha256"]!=capture_schedule["schedule_sha256"] or len(capture["rows"])!=96 or capture["counts"]!={"rows":96,"tokenizer_calls":96,"model_forwards":96,"sequences":2304} or capture["aggregate_sha256"]!=sha256_bytes(canonical_json(capture["rows"])): raise T0ContractError("T0 capture count differs")
    for index,(row,source_row) in enumerate(zip(capture["rows"],capture_schedule["rows"])):
        if set(row)!=row_keys or {k:row[k] for k in ("capture_index","row_id","row_sha256","receiver_input_sha256","node_count")}!=source_row or row["capture_index"]!=index or not 1<=row["token_count_min"]<=row["token_count_max"]<=128 or row["hidden_shape"]!=[24,2048] or row["hidden_dtype"]!="torch.bfloat16" or row["finite"] is not True or any(not digest_string(row[key]) for key in ("input_ids_sha256","attention_mask_sha256","hidden_sha256")): raise T0ContractError("T0 capture row differs")
    initial=value["candidate_initial_state"]; _keys(initial,{"seed_payload","seed_payload_sha256","seed_u64_be","arm_order","action_order","rows","all_equal"},"T0 candidate initial")
    initial_row_keys={"arm","parameter_names","parameter_shapes","parameter_name_count","parameter_count","device","dtype","initial_state_sha256","finite"}
    if (initial["seed_payload"],initial["seed_payload_sha256"],initial["seed_u64_be"])!=(INIT_PAYLOAD,INIT_SHA256,INIT_SEED) or initial["arm_order"]!=ARMS or initial["action_order"]!=ACTIONS or initial["all_equal"] is not True or [r.get("arm") for r in initial["rows"]]!=ARMS: raise T0ContractError("T0 candidate initial differs")
    hashes=set()
    for row in initial["rows"]:
        if set(row)!=initial_row_keys or row["parameter_name_count"]!=16 or row["parameter_count"]!=366340 or row["device"]!="cuda:0" or row["dtype"]!="torch.float32" or row["finite"] is not True or row["parameter_names"]!=PARAMETER_NAMES or row["parameter_shapes"]!=PARAMETER_SHAPES or not digest_string(row["initial_state_sha256"]): raise T0ContractError("T0 candidate row differs")
        hashes.add(row["initial_state_sha256"])
    if len(hashes)!=1: raise T0ContractError("T0 candidate parity differs")
    pre=value["preconnect_evidence"]; _keys(pre,{"row_id","depth","action_index","rows","counts","all_qualify"},"T0 preconnect")
    if (pre["row_id"],pre["depth"],pre["action_index"])!=("hi_be342227610f62e0",4,0) or pre["counts"]!={"forwards":5,"backwards":5,"optimizer_steps":0} or [r.get("arm") for r in pre["rows"]]!=ARMS or pre["all_qualify"] is not True: raise T0ContractError("T0 preconnect differs")
    pre_keys={"arm","operation_index","loss","codec_grad_l2","cell_grad_l2","readout_grad_l2","state_before_sha256","state_after_sha256","gradients_cleared","finite","qualifies"}
    for index,row in enumerate(pre["rows"]):
        cell=row.get("cell_grad_l2")
        if set(row)!=pre_keys or row["operation_index"]!=index or not finite_number(row["loss"]) or not finite_number(row["codec_grad_l2"]) or row["codec_grad_l2"]<1e-8 or not finite_number(row["readout_grad_l2"]) or row["readout_grad_l2"]<1e-8 or (cell is not None if index==0 else not finite_number(cell) or cell<1e-8) or row["state_before_sha256"]!=row["state_after_sha256"] or row["gradients_cleared"] is not True or row["finite"] is not True or row["qualifies"] is not True: raise T0ContractError("T0 preconnect row differs")
    safety=value["safety"]
    if set(safety)!={"network_attempts","validation_opens","heldout_opens","h176_loads","generation_calls","e33_backwards","e33_optimizer_steps","e33_updates","live_trajectory_count","object_census_errors","object_census_uninspectable","forbidden_inputs_detected"} or safety["forbidden_inputs_detected"]!=[] or any(safety[k]!=0 for k in safety if k!="forbidden_inputs_detected"): raise T0ContractError("T0 safety differs")
    gates=value["gate_evidence"]
    if set(gates)!={"ordered_rows","pass_count","all_pass"} or len(gates["ordered_rows"])!=9 or [r.get("name") for r in gates["ordered_rows"]]!=GATE_NAMES: raise T0ContractError("T0 gates differ")
    expected_pass=9 if go else 8
    if gates["pass_count"]!=expected_pass or gates["all_pass"] is not go or sum(r.get("passed") is True for r in gates["ordered_rows"])!=expected_pass: raise T0ContractError("T0 gate result differs")
    metric=value["metric_evidence"]
    if set(metric)!={"row_records","aggregate_records","postcal_aggregate_sha256","counts"} or len(metric["row_records"])!=640 or len(metric["aggregate_records"])!=135 or metric["counts"]!={"row_records":640,"aggregate_records":135,"precal_presentations":160,"postcal_presentations":160,"postfit_presentations":320}: raise T0ContractError("T0 metric evidence differs")
    if metric["postcal_aggregate_sha256"]!=sha256_bytes(canonical_json([r for r in metric["aggregate_records"] if r["phase"]=="POSTCAL"])): raise T0ContractError("T0 POSTCAL aggregate hash differs")
    metric_row_keys={"phase","operation_index","arm","depth","row_id","action_index","replicate","logits","logits_sha256","prediction","nll"}; aggregate_row_keys={"phase","arm","scope","scope_value","count","correct","accuracy_numerator","accuracy_denominator","mean_nll"}
    partition_by_id={r["row_id"]:r for r in [*partition["fit_rows"],*partition["calibration_rows"]]}
    expected_metric_identity=[]
    for phase_name,batch_name in (("PRECAL","precal"),("POSTCAL","postcal"),("POSTFIT","postfit")):
        for operation in training_schedule["batches"][batch_name]:
            for row_id in operation["row_ids"]:
                source=partition_by_id[row_id]
                expected_metric_identity.append({"phase":phase_name,"operation_index":operation["operation_index"],"arm":operation["arm"],"depth":operation["depth"],"row_id":row_id,"action_index":source["action_index"],"replicate":source["replicate"]})
    if len(expected_metric_identity)!=640 or len(metric["row_records"])!=640: raise T0ContractError("T0 metric schedule differs")
    try: import torch
    except Exception as error: raise T0ContractError("T0 metric validator requires frozen CPU float64 runtime") from error
    for record,identity in zip(metric["row_records"],expected_metric_identity):
        if set(record)!=metric_row_keys or {k:record[k] for k in identity}!=identity or not isinstance(record["logits"],list) or len(record["logits"])!=4 or any(not finite_number(x) for x in record["logits"]): raise T0ContractError("T0 metric row differs")
        vector=torch.tensor(record["logits"],dtype=torch.float64,device="cpu")
        expected_hash=sha256_bytes(canonical_json({"dtype":"torch.float64","shape":[4]})+b"\n"+vector.contiguous().view(torch.uint8).numpy().tobytes())
        expected_nll=float(-torch.log_softmax(vector,dim=-1)[record["action_index"]]); expected_prediction=int(torch.argmax(vector))
        if record["logits_sha256"]!=expected_hash or record["nll"]!=expected_nll or record["prediction"]!=expected_prediction: raise T0ContractError("T0 metric logits differ")
    if any(set(r)!=aggregate_row_keys or r["accuracy_numerator"]!=r["correct"] or r["accuracy_denominator"]!=r["count"] or not finite_number(r["mean_nll"]) for r in metric["aggregate_records"]): raise T0ContractError("T0 aggregate row differs")
    recomputed=[]
    for phase in ("PRECAL","POSTCAL","POSTFIT"):
        for arm in ARMS:
            subset=[r for r in metric["row_records"] if r["phase"]==phase and r["arm"]==arm]
            scopes=[("overall",None,subset),*[("depth",d,[r for r in subset if r["depth"]==d]) for d in range(1,5)],*[("action",a,[r for r in subset if r["action_index"]==a]) for a in range(4)]]
            for scope,scope_value,rows in scopes:
                correct=sum(r["prediction"]==r["action_index"] for r in rows); recomputed.append({"phase":phase,"arm":arm,"scope":scope,"scope_value":scope_value,"count":len(rows),"correct":correct,"accuracy_numerator":correct,"accuracy_denominator":len(rows),"mean_nll":math.fsum(r["nll"] for r in rows)/len(rows)})
    if metric["aggregate_records"]!=recomputed: raise T0ContractError("T0 aggregates do not recompute")
    get=lambda p,a,s="overall",v=None:next(r for r in recomputed if (r["phase"],r["arm"],r["scope"],r["scope_value"])==(p,a,s,v))
    rec=get("POSTCAL","REC_K"); reset=get("POSTCAL","RESET_K"); static=get("POSTCAL","STATIC"); ffn=get("POSTCAL","FFN"); precal=get("PRECAL","REC_K"); fit=get("POSTFIT","REC_K")
    lhs=[[rec["correct"],32],[rec["correct"]-reset["correct"],32],[rec["correct"]-max(static["correct"],ffn["correct"]),32],[[get("POSTCAL","REC_K","depth",d)["correct"],8] for d in range(1,5)],[[get("POSTCAL","REC_K","action",a)["correct"],8] for a in range(4)],reset["mean_nll"]-rec["mean_nll"],min(static["mean_nll"],ffn["mean_nll"])-rec["mean_nll"],rec["mean_nll"]/precal["mean_nll"],[fit["correct"],64]]
    rhs=[[20,32],[4,32],[2,32],[4,8],[3,8],.05,.02,.75,[48,64]]; operators=[">=",">=",">=","all>=","all>=",">=",">=","<=",">="]
    passes=[lhs[0][0]>=20,lhs[1][0]>=4,lhs[2][0]>=2,all(x[0]>=4 for x in lhs[3]),all(x[0]>=3 for x in lhs[4]),lhs[5]>=.05,lhs[6]>=.02,lhs[7]<=.75,lhs[8][0]>=48]
    expected_gate_rows=[{"index":i,"name":GATE_NAMES[i],"lhs":lhs[i],"operator":operators[i],"rhs":rhs[i],"passed":passes[i]} for i in range(9)]
    if gates["ordered_rows"]!=expected_gate_rows: raise T0ContractError("T0 gates do not recompute")
    disposition=value["candidate_disposition"]
    if set(disposition)!={"arm_order","files_present","threshold_file_present","rows","reusable"} or disposition["arm_order"]!=ARMS or disposition["files_present"]!=(CANDIDATE_FILES if go else []) or disposition["threshold_file_present"] is not go or len(disposition["rows"])!=expected_files or disposition["reusable"] is not go: raise T0ContractError("T0 candidate disposition differs")
    candidate_row_keys={"arm","path","file_sha256","bytes","module_state_sha256","parameter_names","parameter_name_count","parameter_count","dtype","device","finite","safe_reload_passed"}
    for arm,row in zip(ARMS,disposition["rows"]):
        if set(row)!=candidate_row_keys or row["arm"]!=arm or row["path"]!=f"candidate-{arm}.safetensors" or row["parameter_names"]!=PARAMETER_NAMES or row["parameter_name_count"]!=16 or row["parameter_count"]!=366340 or row["dtype"]!="torch.float32" or row["device"]!="cpu" or row["finite"] is not True or row["safe_reload_passed"] is not True or not digest_string(row["file_sha256"]) or not digest_string(row["module_state_sha256"]) or not isinstance(row["bytes"],int) or row["bytes"]<=0: raise T0ContractError("T0 candidate file evidence differs")
    if output_dir is not None and go:
        from pathlib import Path
        root=Path(output_dir)
        for row in disposition["rows"]:
            validate_candidate_artifact(root/row["path"],row)
    derived=value["derived_thresholds"]
    if not go:
        if derived is not None: raise T0ContractError("T0 STOP thresholds present")
    else:
        _keys(derived,{"path","file_sha256","bytes","payload"},"T0 derived thresholds")
        payload=derived["payload"]
        if derived["path"]!="derived-validation-thresholds.json" or payload.get("schema_version")!="prime-rl/latent-h-iter-phase1-t0-derived-validation-thresholds/v1" or payload.get("status")!="h_iter_phase1_t0_validation_thresholds_derived" or payload.get("threshold_sha256")!=sha256_bytes(canonical_json({k:v for k,v in payload.items() if k!="threshold_sha256"})): raise T0ContractError("T0 derived thresholds differ")
        inputs={"AR":rec["correct"]/32,"min_depth_AR":min(get("POSTCAL","REC_K","depth",d)["correct"]/8 for d in range(1,5)),"A_RESET":reset["correct"]/32,"N_RESET":reset["mean_nll"],"NR":rec["mean_nll"]}; ar=inputs["AR"]; min_depth=inputs["min_depth_AR"]; a_reset=inputs["A_RESET"]; n_reset=inputs["N_RESET"]; nr=inputs["NR"]
        formulas={"fixed_rec_nll_min":".01","per_depth_min":"min(18/24,max(10/24,floor_24(min_depth_AR-1/8)))","rec_accuracy_min":"min(36/48,max(24/48,floor_48(AR-4/32)))","rec_fixed_accuracy_min":"2/48","rec_reset_accuracy_min":"min(8/48,max(2/48,floor_48(.5*(AR-A_RESET))))","reset_rec_nll_min":"min(.15,max(.02,floor_1e6(.5*(N_RESET-NR))))"}
        thresholds={"rec_accuracy_min":[min(36,max(24,math.floor(48*(ar-4/32)))),48],"per_depth_min":[min(18,max(10,math.floor(24*(min_depth-1/8)))),24],"rec_reset_accuracy_min":[min(8,max(2,math.floor(48*.5*(ar-a_reset)))),48],"reset_rec_nll_min":min(.15,max(.02,math.floor(1_000_000*.5*(n_reset-nr))/1_000_000)),"rec_fixed_accuracy_min":[2,48],"fixed_rec_nll_min":.01}
        if payload.get("mechanism")!=MECHANISM or payload.get("run_identity")!=RUN_ID or payload.get("postcal_aggregate_sha256")!=metric["postcal_aggregate_sha256"] or payload.get("inputs")!=inputs or payload.get("formulas")!=formulas or payload.get("thresholds")!=thresholds or derived["file_sha256"]!=sha256_bytes(canonical_json(payload)+b"\n") or derived["bytes"]!=len(canonical_json(payload)+b"\n"): raise T0ContractError("T0 threshold formula differs")
        if output_dir is not None:
            from pathlib import Path
            path=Path(output_dir)/derived["path"]
            if not path.is_file() or path.is_symlink() or path.stat().st_size!=derived["bytes"] or sha256_bytes(path.read_bytes())!=derived["file_sha256"] or strict_loads(path.read_bytes())!=payload: raise T0ContractError("T0 threshold file differs")
    protected=value["protected_state"]; _keys(protected,{"e33_disk_tree_before","e33_disk_tree_after","e33_state_before","e33_state_after","e33_metadata_before","e33_metadata_after","e33_grads_none","h176_tree_sha256","h176_loaded","e33_released"},"T0 protected")
    if protected!={"e33_disk_tree_before":E33_TREE_SHA256,"e33_disk_tree_after":E33_TREE_SHA256,"e33_state_before":E33_STATE_SHA256,"e33_state_after":E33_STATE_SHA256,"e33_metadata_before":METADATA_SHA256,"e33_metadata_after":METADATA_SHA256,"e33_grads_none":True,"h176_tree_sha256":H176_TREE_SHA256,"h176_loaded":False,"e33_released":True}: raise T0ContractError("T0 protected state differs")
    cache=value["cache_guard"]; _keys(cache,{"class_records","configuration_records","label_rows","label_sha256","expected_checks","actual_checks","dynamic_cache_negative_trips","dynamic_cache_actual_trips","returned_pkv_count","configuration_drift_count","restored"},"T0 cache")
    cache_labels=["CACHE_ENTRY",*[x for i in range(96) for x in (f"CACHE_PRE_{i:03d}",f"CACHE_POST_{i:03d}")],"CACHE_EXIT"]
    if cache["class_records"]!=CACHE_CLASS_RECORDS or cache["configuration_records"]!=CACHE_CONFIGURATION_RECORDS or cache["expected_checks"]!=194 or cache["actual_checks"]!=194 or [r.get("index") for r in cache["label_rows"]]!=list(range(194)) or [r.get("label") for r in cache["label_rows"]]!=cache_labels or cache["label_sha256"]!=sha256_bytes(canonical_json(cache_labels)) or cache["dynamic_cache_negative_trips"]!=1 or cache["dynamic_cache_actual_trips"]!=0 or cache["returned_pkv_count"]!=0 or cache["configuration_drift_count"]!=0 or cache["restored"] is not True: raise T0ContractError("T0 cache evidence differs")
    memory=value["memory"]
    expected_memory_labels=[r["label"] for r in memory_schedule["rows"]]
    if set(memory)!={"schedule_file_sha256","expected_count","rows","label_sha256","complete"} or memory["schedule_file_sha256"]!=sha256_bytes(canonical_json(memory_schedule)+b"\n") or memory["expected_count"]!=534 or len(memory["rows"])!=534 or not memory["complete"]: raise T0ContractError("T0 memory evidence differs")
    memory_keys={"index","label","current_allocated_bytes","current_reserved_bytes","peak_allocated_bytes","peak_reserved_bytes"}
    if [r.get("index") for r in memory["rows"]]!=list(range(534)) or [r.get("label") for r in memory["rows"]]!=expected_memory_labels or memory["label_sha256"]!=sha256_bytes(canonical_json(expected_memory_labels)): raise T0ContractError("T0 memory order differs")
    for row in memory["rows"]:
        values=[row.get(key) for key in ("current_allocated_bytes","current_reserved_bytes","peak_allocated_bytes","peak_reserved_bytes")]
        if set(row)!=memory_keys or any(not isinstance(item,int) or isinstance(item,bool) or not 0<=item<=40*(1<<30) for item in values) or values[0]>values[2] or values[1]>values[3]: raise T0ContractError("T0 memory row differs")
    resources=value["resources"]; _keys(resources,{"gpu_name","physical_gpu","visible_device","free_gpu_bytes_pre","available_ram_bytes_pre","free_disk_bytes_pre","max_allocated_bytes","max_reserved_bytes","artifact_bytes","timing"},"T0 resources")
    if resources["gpu_name"]!="NVIDIA RTX A6000" or resources["physical_gpu"]!="0" or resources["visible_device"]!="cuda:0" or resources["free_gpu_bytes_pre"]<44*(1<<30) or resources["available_ram_bytes_pre"]<64*(1<<30) or resources["free_disk_bytes_pre"]<16*(1<<30) or max(resources["max_allocated_bytes"],resources["max_reserved_bytes"])>40*(1<<30) or resources["artifact_bytes"]>33554432: raise T0ContractError("T0 resources differ")
    timing=resources["timing"]; timing_keys={"outer_seconds","startup_seconds","compute_seconds","audit_seconds","failure_seconds","terminal_seconds","postexit_seconds","alarm_safety_margin_seconds","compute_enter_ns","compute_exit_ns","compute_duration_ns","audit_enter_ns","audit_exit_ns","audit_duration_ns","failure_enter_ns","failure_exit_ns","failure_duration_ns","terminal_enter_ns","prepublication_elapsed_ns"}
    if set(timing)!=timing_keys or [timing[k] for k in ("outer_seconds","startup_seconds","compute_seconds","audit_seconds","failure_seconds","terminal_seconds","postexit_seconds","alarm_safety_margin_seconds")]!=[21600,600,18000,1200,1200,300,300,1] or timing["compute_duration_ns"]!=timing["compute_exit_ns"]-timing["compute_enter_ns"] or timing["audit_duration_ns"]!=timing["audit_exit_ns"]-timing["audit_enter_ns"] or timing["prepublication_elapsed_ns"]!=timing["terminal_enter_ns"] or timing["terminal_enter_ns"]>19200*10**9: raise T0ContractError("T0 timing differs")
    freeze=value["full_freeze"]; _keys(freeze,{"target_count","head_before","head_after","tree_before","tree_after","clean_before","clean_after","pre_entries","post_entries","pre_sha256","post_sha256","complete"},"T0 full freeze")
    if freeze["target_count"]!=38 or freeze["head_before"]!=freeze["head_after"] or freeze["tree_before"]!=freeze["tree_after"] or freeze["clean_before"] is not True or freeze["clean_after"] is not True or freeze["pre_entries"]!=freeze["post_entries"] or freeze["pre_sha256"]!=freeze["post_sha256"] or freeze["complete"] is not True: raise T0ContractError("T0 full freeze differs")
    tamper=value["tamper_audit"]; _keys(tamper,{"schedule_file_sha256","expected_count","results","rejected_count","all_rejected"},"T0 tamper")
    if tamper["expected_count"]!=98 or tamper["rejected_count"]!=98 or tamper["all_rejected"] is not True or len(tamper["results"])!=98 or any(set(row)!={"index","name","rejected","observed_error_type"} or row["index"]!=index or row["name"]!=_TAMPER_NAMES[index] or row["rejected"] is not True or not isinstance(row["observed_error_type"],str) or not row["observed_error_type"] for index,row in enumerate(tamper["results"])): raise T0ContractError("T0 tamper evidence differs")
    decision=value["decision_boundary"]
    claim="train_only_matched_sidecar_learning_screen_passed" if go else "train_only_matched_sidecar_learning_screen_stopped"
    expected_decision={"claim":claim,"candidate_modules_updated":True,"protected_models_updated":False,"validation_contract_design_authorized":go,"validation_execution_authorized":False,"validation_or_heldout_opened":False,"live_trajectory_count":0,"admission":False,"nomination":False,"promotion":False,"four_live_floor_unchanged":True}
    if decision!=expected_decision: raise T0ContractError("T0 decision boundary differs")
    if output_inventory is not None:
        expected=["T0-PROOF.json"] if not go else sorted([*CANDIDATE_FILES,"derived-validation-thresholds.json","T0-PROOF.json"])
        if output_inventory!=expected: raise T0ContractError("T0 output inventory differs")

def validate_t0_failure(value:dict[str,Any], *, partition:dict[str,Any]|None=None, capture_schedule:dict[str,Any]|None=None, training_schedule:dict[str,Any]|None=None, memory_schedule:dict[str,Any]|None=None, output_inventory:list[str]|None=None, output_dir:Any|None=None)->None:
    if partition is None or capture_schedule is None or training_schedule is None or memory_schedule is None: raise T0ContractError("T0 failure frozen validation sources missing")
    statuses={"h_iter_phase1_t0_incomplete":"T0ContractError","h_iter_phase1_t0_capture_mechanism_rejected":"T0CaptureMechanismRejected","h_iter_phase1_t0_exposure_boundary_rejected":"T0ExposureBoundaryRejected","infrastructure_invalid":"InfrastructureInvalid"}
    expected=(T0_PROOF_KEYS-{"proof_sha256"})|{"error_type","error_message","traceback","stage","execution_progress","audit_errors","failure_sha256"}
    _keys(value,expected,"T0 failure")
    if value.get("schema_version")!=FAILURE_SCHEMA or value.get("mechanism")!=MECHANISM or value.get("run_identity")!=RUN_ID or statuses.get(value.get("status"))!=value.get("error_type"): raise T0ContractError("T0 failure taxonomy differs")
    if not all(isinstance(value.get(field),str) and digest_string(value[field]) for field in ("plan_file_sha256","plan_sha256")) or not all(isinstance(value.get(field),str) and len(value[field])==40 and all(character in "0123456789abcdef" for character in value[field]) for field in ("execution_commit","mechanism_code_commit")) or not isinstance(value.get("error_message"),str) or not value["error_message"] or not isinstance(value.get("traceback"),str): raise T0ContractError("T0 failure authority or error schema differs")
    if value.get("failure_sha256")!=sha256_bytes(canonical_json({k:v for k,v in value.items() if k!="failure_sha256"})): raise T0ContractError("T0 failure self hash differs")
    progress=value.get("execution_progress")
    progress_keys={"stage","capture_rows_completed","tokenizer_calls_completed","model_forwards_completed","sequences_completed","cache_checks_completed","candidates_initialized","preconnect_arms_completed","operations_completed","current_operation_index","current_phase","current_arm","current_epoch","current_depth","sidecar_forwards_completed","sidecar_backwards_completed","optimizer_steps_completed","cell_calls_completed","metric_row_records_completed","aggregate_records_completed","gates_evaluated","candidate_files_present","threshold_file_present"}
    stages=["startup_pre_model","model_load","capture","model_release","candidate_init","preconnect","precal","train","postcal","postfit","gate_evaluation","candidate_write","postflight_audit","terminal_publication"]
    stage_rank={name:index for index,name in enumerate(stages)}
    if not isinstance(progress,dict) or set(progress)!=progress_keys or progress.get("stage") not in stages or value.get("stage")!=progress.get("stage"): raise T0ContractError("T0 failure progress differs")
    stage=progress["stage"]
    numeric=["capture_rows_completed","tokenizer_calls_completed","model_forwards_completed","sequences_completed","cache_checks_completed","candidates_initialized","preconnect_arms_completed","operations_completed","sidecar_forwards_completed","sidecar_backwards_completed","optimizer_steps_completed","cell_calls_completed","metric_row_records_completed","aggregate_records_completed","gates_evaluated"]
    if any(not isinstance(progress[k],int) or isinstance(progress[k],bool) or progress[k]<0 for k in numeric) or not isinstance(progress["candidate_files_present"],list) or not isinstance(progress["threshold_file_present"],bool): raise T0ContractError("T0 failure counter differs")
    captures=progress["capture_rows_completed"]; tokenizers=progress["tokenizer_calls_completed"]; model_forwards=progress["model_forwards_completed"]; cache_checks=progress["cache_checks_completed"]
    if not 0<=captures<=model_forwards<=tokenizers<=96 or model_forwards>captures+1 or tokenizers>model_forwards+1 or progress["sequences_completed"]!=24*tokenizers or not 0<=cache_checks<=194: raise T0ContractError("T0 failure capture progress differs")
    counts=value.get("counts")
    if not isinstance(counts,dict) or set(counts)!=T0_COUNT_KEYS or any(not isinstance(item,int) or isinstance(item,bool) or item<0 for item in counts.values()): raise T0ContractError("T0 failure counts schema differs")
    direct_counts={"capture_rows":captures,"tokenizer_calls":tokenizers,"model_forwards":model_forwards,"sequences":progress["sequences_completed"],"cache_checks":cache_checks,"candidate_objects":progress["candidates_initialized"],"sidecar_forwards":progress["sidecar_forwards_completed"],"sidecar_backwards":progress["sidecar_backwards_completed"],"optimizer_steps":progress["optimizer_steps_completed"],"cell_calls":progress["cell_calls_completed"],"candidate_files":len(progress["candidate_files_present"]),"threshold_files":int(progress["threshold_file_present"])}
    if any(counts[key]!=expected_value for key,expected_value in direct_counts.items()): raise T0ContractError("T0 failure counts/progress differ")
    capture=value.get("capture_evidence")
    if capture is not None:
        _keys(capture,{"schedule_sha256","rows","counts","aggregate_sha256"},"T0 failure capture")
        rows=capture["rows"]
        inflight_row=isinstance(rows,list) and len(rows)==captures+1 and stage=="capture" and model_forwards==captures+1
        evidence_row_count=capture_failure_evidence_row_count(rows,captures,model_forwards)
        if stage_rank[stage]<stage_rank["capture"] or not isinstance(rows,list) or not rows or capture["schedule_sha256"]!=capture_schedule["schedule_sha256"] or len(rows)!=captures and not inflight_row or capture["counts"]!={"rows":evidence_row_count,"tokenizer_calls":tokenizers,"model_forwards":model_forwards,"sequences":24*tokenizers} or capture["aggregate_sha256"]!=sha256_bytes(canonical_json(rows)): raise T0ContractError("T0 failure capture evidence differs")
        capture_keys={"capture_index","row_id","row_sha256","receiver_input_sha256","node_count","token_count_min","token_count_max","input_ids_sha256","attention_mask_sha256","hidden_shape","hidden_dtype","hidden_sha256","finite"}
        for index,(row,source) in enumerate(zip(rows,capture_schedule["rows"])):
            if set(row)!=capture_keys or {key:row[key] for key in ("capture_index","row_id","row_sha256","receiver_input_sha256","node_count")}!=source or row["capture_index"]!=index or not 1<=row["token_count_min"]<=row["token_count_max"]<=128 or row["hidden_shape"]!=[24,2048] or row["hidden_dtype"]!="torch.bfloat16" or not all(digest_string(row[key]) for key in ("input_ids_sha256","attention_mask_sha256","hidden_sha256")) or not isinstance(row["finite"],bool): raise T0ContractError("T0 failure capture row differs")
    elif captures:
        raise T0ContractError("T0 failure capture evidence missing")
    operations=[*training_schedule["batches"]["preconnect"],*training_schedule["batches"]["precal"],*training_schedule["batches"]["train"],*training_schedule["batches"]["postcal"],*training_schedule["batches"]["postfit"]]
    operations=sorted(operations,key=lambda row:row["operation_index"])
    completed=progress["operations_completed"]
    if not 0<=completed<=385 or [r["operation_index"] for r in operations]!=list(range(385)): raise T0ContractError("T0 failure operation prefix differs")
    def cell_count(row:dict[str,Any])->int: return 0 if row["arm"]=="STATIC" else 1 if row["arm"]=="FFN" else 4 if row["arm"]=="FIXED_T4" else row["depth"]
    done=operations[:completed]; base_forwards=len(done); base_backwards=sum(bool(r["gradient"]) for r in done); base_steps=sum(bool(r["optimizer_step"]) for r in done); base_cells=sum(cell_count(r) for r in done)
    current_index=progress["current_operation_index"]; in_flight=current_index==completed and completed<385
    if completed and not in_flight and current_index!=completed-1: raise T0ContractError("T0 failure current operation differs")
    if not completed and not in_flight and current_index is not None: raise T0ContractError("T0 failure current operation differs")
    delta_f=progress["sidecar_forwards_completed"]-base_forwards; delta_b=progress["sidecar_backwards_completed"]-base_backwards; delta_s=progress["optimizer_steps_completed"]-base_steps; delta_c=progress["cell_calls_completed"]-base_cells
    if in_flight:
        current=operations[completed]
        if (progress["current_phase"],progress["current_arm"],progress["current_epoch"],progress["current_depth"])!=(current["phase"],current["arm"],current.get("epoch"),current["depth"]): raise T0ContractError("T0 failure current operation identity differs")
        if delta_f not in {0,1} or delta_b not in ({0,1} if current["gradient"] else {0}) or delta_s not in ({0,1} if current["optimizer_step"] else {0}) or delta_b>delta_f or delta_s>delta_b or delta_c!=(cell_count(current) if delta_f else 0): raise T0ContractError("T0 failure in-flight operation differs")
    elif any((delta_f,delta_b,delta_s,delta_c)): raise T0ContractError("T0 failure completed operation counters differ")
    elif completed:
        last=operations[completed-1]
        if (progress["current_phase"],progress["current_arm"],progress["current_epoch"],progress["current_depth"])!=(last["phase"],last["arm"],last.get("epoch"),last["depth"]): raise T0ContractError("T0 failure completed operation identity differs")
    elif any(progress[key] is not None for key in ("current_phase","current_arm","current_epoch","current_depth")):
        raise T0ContractError("T0 failure empty operation identity differs")
    if progress["preconnect_arms_completed"]!=min(completed,5) or not 0<=progress["candidates_initialized"]<=5 or completed and progress["candidates_initialized"]!=5: raise T0ContractError("T0 failure initialization progress differs")
    runtime=value.get("runtime"); runtime_exact=True
    if runtime is not None:
        _keys(runtime,set(T0_RUNTIME),"T0 failure runtime")
        if any(not isinstance(runtime[key],str) for key in runtime): raise T0ContractError("T0 failure runtime type differs")
        runtime_exact=runtime==T0_RUNTIME
    asset_audit=value.get("asset_audit"); assets_exact=True
    if asset_audit is not None:
        _keys(asset_audit,{"target_count","pre_entries","pre_sha256","post_entries","post_sha256","all_exact"},"T0 failure asset audit")
        pre_entries=asset_audit["pre_entries"]
        if asset_audit["target_count"]!=38 or not isinstance(pre_entries,list) or len(pre_entries)!=38 or any(set(row)!=T0_ASSET_ROW_KEYS or not isinstance(row["path"],str) or not row["path"] or not digest_string(row["sha256"]) or not isinstance(row["bytes"],int) or isinstance(row["bytes"],bool) or row["bytes"]<0 for row in pre_entries) or asset_audit["pre_sha256"]!=sha256_bytes(canonical_json(pre_entries)): raise T0ContractError("T0 failure asset preflight differs")
        post_entries=asset_audit["post_entries"]
        if post_entries is None:
            if asset_audit["post_sha256"] is not None or asset_audit["all_exact"] is not None: raise T0ContractError("T0 failure asset postflight null relation differs")
        elif not isinstance(post_entries,list) or any(set(row)!=T0_ASSET_ROW_KEYS or not isinstance(row["path"],str) or not row["path"] or not digest_string(row["sha256"]) or not isinstance(row["bytes"],int) or isinstance(row["bytes"],bool) or row["bytes"]<0 for row in post_entries) or asset_audit["post_sha256"]!=sha256_bytes(canonical_json(post_entries)) or asset_audit["all_exact"] is not (post_entries==pre_entries): raise T0ContractError("T0 failure asset postflight differs")
        assets_exact=post_entries is None or post_entries==pre_entries
    antecedent=value.get("antecedent_binding"); antecedent_exact=True
    if antecedent is not None:
        _keys(antecedent,{"manifest_path","manifest_file_sha256","manifest_internal_sha256","mf0_archive_exact","cap0_r1_archive_exact"},"T0 failure antecedent")
        if not digest_string(antecedent["manifest_file_sha256"]) or not digest_string(antecedent["manifest_internal_sha256"]) or not isinstance(antecedent["mf0_archive_exact"],bool) or not isinstance(antecedent["cap0_r1_archive_exact"],bool): raise T0ContractError("T0 failure antecedent schema differs")
        antecedent_exact=antecedent["manifest_path"]==f"{ARTIFACT_DIR}/t0-antecedent-evidence-manifest.json" and antecedent["mf0_archive_exact"] is True and antecedent["cap0_r1_archive_exact"] is True
    data=value.get("data_binding")
    if data is not None:
        _keys(data,{"train_bank_path","train_bank_file_sha256","train_bank_internal_sha256","source_rows","fit_rows","calibration_rows","fit_row_ids","calibration_row_ids","fit_calibration_intersection","complete_train_union","validation_open_count","heldout_open_count"},"T0 failure data")
        expected_fit=[row["row_id"] for row in partition["fit_rows"]]; expected_cal=[row["row_id"] for row in partition["calibration_rows"]]
        if data!={"train_bank_path":TRAIN_BANK_PATH,"train_bank_file_sha256":TRAIN_BANK_FILE_SHA256,"train_bank_internal_sha256":TRAIN_BANK_INTERNAL_SHA256,"source_rows":96,"fit_rows":64,"calibration_rows":32,"fit_row_ids":expected_fit,"calibration_row_ids":expected_cal,"fit_calibration_intersection":[],"complete_train_union":True,"validation_open_count":0,"heldout_open_count":0}: raise T0ContractError("T0 failure data differs")
    resources=value.get("resources"); resources_exact=True
    if resources is not None:
        _keys(resources,T0_RESOURCE_KEYS,"T0 failure resources")
        integer_fields=("free_gpu_bytes_pre","available_ram_bytes_pre","free_disk_bytes_pre","max_allocated_bytes","max_reserved_bytes","artifact_bytes")
        if not all(isinstance(resources[key],str) for key in ("gpu_name","physical_gpu","visible_device")) or any(not isinstance(resources[key],int) or isinstance(resources[key],bool) or resources[key]<0 for key in integer_fields): raise T0ContractError("T0 failure resources schema differs")
        resources_exact=resources["gpu_name"]=="NVIDIA RTX A6000" and resources["physical_gpu"]=="0" and resources["visible_device"]=="cuda:0" and resources["free_gpu_bytes_pre"]>=44*(1<<30) and resources["available_ram_bytes_pre"]>=64*(1<<30) and resources["free_disk_bytes_pre"]>=16*(1<<30) and max(resources["max_allocated_bytes"],resources["max_reserved_bytes"])<=40*(1<<30) and resources["artifact_bytes"]<=33554432
        timing=resources["timing"]
        if timing is not None:
            _keys(timing,T0_TIMING_KEYS,"T0 failure timing")
            if [timing[key] for key in ("outer_seconds","startup_seconds","compute_seconds","audit_seconds","failure_seconds","terminal_seconds","postexit_seconds","alarm_safety_margin_seconds")]!=[21600,600,18000,1200,1200,300,300,1]: raise T0ContractError("T0 failure timing constants differ")
            nullable=("compute_enter_ns","compute_exit_ns","compute_duration_ns","audit_enter_ns","audit_exit_ns","audit_duration_ns","failure_enter_ns","failure_exit_ns","failure_duration_ns","terminal_enter_ns","prepublication_elapsed_ns")
            if any(item is not None and (not isinstance(item,int) or isinstance(item,bool) or item<0) for item in (timing[key] for key in nullable)): raise T0ContractError("T0 failure timing value differs")
            for enter,exit_,duration in (("compute_enter_ns","compute_exit_ns","compute_duration_ns"),("audit_enter_ns","audit_exit_ns","audit_duration_ns"),("failure_enter_ns","failure_exit_ns","failure_duration_ns")):
                values=(timing[enter],timing[exit_],timing[duration])
                if all(item is None for item in values): continue
                if any(item is None for item in values) or values[2]!=values[1]-values[0]: raise T0ContractError("T0 failure timing interval differs")
            if timing["terminal_enter_ns"] is None:
                if timing["prepublication_elapsed_ns"] is not None: raise T0ContractError("T0 failure terminal timing null relation differs")
            elif timing["prepublication_elapsed_ns"]!=timing["terminal_enter_ns"] or timing["terminal_enter_ns"]>20700*10**9: raise T0ContractError("T0 failure terminal timing differs")
            if timing["compute_duration_ns"] is not None and timing["compute_duration_ns"]>18000*10**9 or timing["audit_duration_ns"] is not None and timing["audit_duration_ns"]>1200*10**9 or timing["failure_duration_ns"] is not None and timing["failure_duration_ns"]>1200*10**9: raise T0ContractError("T0 failure timing cap differs")
    required_foundations=stage_rank[stage]>=stage_rank["model_load"]
    if required_foundations and any(item is None for item in (runtime,asset_audit,antecedent,data,resources)): raise T0ContractError("T0 failure foundational evidence missing")
    infrastructure_observed=not runtime_exact or not assets_exact or not antecedent_exact or not resources_exact
    if infrastructure_observed and value["status"]!="infrastructure_invalid": raise T0ContractError("T0 infrastructure evidence masked")
    stage_operation_ranges={
        "startup_pre_model":(0,0),"model_load":(0,0),"capture":(0,0),"model_release":(0,0),"candidate_init":(0,0),
        "preconnect":(0,5),"precal":(5,25),"train":(25,345),"postcal":(345,365),"postfit":(365,385),
        "gate_evaluation":(385,385),"candidate_write":(385,385),"postflight_audit":(385,385),"terminal_publication":(385,385),
    }
    low,high=stage_operation_ranges[stage]
    if not low<=completed<=high or stage in {"startup_pre_model","model_load"} and any((captures,tokenizers,model_forwards,cache_checks,progress["candidates_initialized"])) or stage in {"model_release","candidate_init","preconnect","precal","train","postcal","postfit","gate_evaluation","candidate_write","postflight_audit","terminal_publication"} and (captures,tokenizers,model_forwards,cache_checks)!=(96,96,96,194) or stage in {"preconnect","precal","train","postcal","postfit","gate_evaluation","candidate_write","postflight_audit","terminal_publication"} and progress["candidates_initialized"]!=5:
        raise T0ContractError("T0 failure stage crossing differs")
    initial=value.get("candidate_initial_state")
    if initial is not None:
        if stage_rank[stage]<stage_rank["candidate_init"] or progress["candidates_initialized"]!=5: raise T0ContractError("T0 failure future candidate initial evidence")
        _keys(initial,{"seed_payload","seed_payload_sha256","seed_u64_be","arm_order","action_order","rows","all_equal"},"T0 failure candidate initial")
        initial_keys={"arm","parameter_names","parameter_shapes","parameter_name_count","parameter_count","device","dtype","initial_state_sha256","finite"}
        if (initial["seed_payload"],initial["seed_payload_sha256"],initial["seed_u64_be"])!=(INIT_PAYLOAD,INIT_SHA256,INIT_SEED) or initial["arm_order"]!=ARMS or initial["action_order"]!=ACTIONS or initial["all_equal"] is not True or [row.get("arm") for row in initial["rows"]]!=ARMS or any(set(row)!=initial_keys or row["parameter_names"]!=PARAMETER_NAMES or row["parameter_shapes"]!=PARAMETER_SHAPES or row["parameter_name_count"]!=16 or row["parameter_count"]!=366340 or row["device"]!="cuda:0" or row["dtype"]!="torch.float32" or row["finite"] is not True or not digest_string(row["initial_state_sha256"]) for row in initial["rows"]) or len({row["initial_state_sha256"] for row in initial["rows"]})!=1: raise T0ContractError("T0 failure candidate initial differs")
    elif stage_rank[stage]>stage_rank["candidate_init"]:
        raise T0ContractError("T0 failure candidate initial missing")
    preconnect=value.get("preconnect_evidence")
    if preconnect is not None:
        if stage_rank[stage]<stage_rank["preconnect"] or completed<5 or progress["preconnect_arms_completed"]!=5: raise T0ContractError("T0 failure future preconnect evidence")
        _keys(preconnect,{"row_id","depth","action_index","rows","counts","all_qualify"},"T0 failure preconnect")
        if preconnect["row_id"]!="hi_be342227610f62e0" or preconnect["depth"]!=4 or preconnect["action_index"]!=0 or preconnect["counts"]!={"forwards":5,"backwards":5,"optimizer_steps":0} or preconnect["all_qualify"] is not True or [row.get("arm") for row in preconnect["rows"]]!=ARMS: raise T0ContractError("T0 failure preconnect differs")
        preconnect_keys={"arm","operation_index","loss","codec_grad_l2","cell_grad_l2","readout_grad_l2","state_before_sha256","state_after_sha256","gradients_cleared","finite","qualifies"}
        for index,row in enumerate(preconnect["rows"]):
            cell=row.get("cell_grad_l2")
            if set(row)!=preconnect_keys or row["operation_index"]!=index or not finite_number(row["loss"]) or not finite_number(row["codec_grad_l2"]) or row["codec_grad_l2"]<1e-8 or not finite_number(row["readout_grad_l2"]) or row["readout_grad_l2"]<1e-8 or (cell is not None if index==0 else not finite_number(cell) or cell<1e-8) or row["state_before_sha256"]!=row["state_after_sha256"] or not digest_string(row["state_before_sha256"]) or row["gradients_cleared"] is not True or row["finite"] is not True or row["qualifies"] is not True: raise T0ContractError("T0 failure preconnect row differs")
    elif stage_rank[stage]>stage_rank["preconnect"]:
        raise T0ContractError("T0 failure preconnect evidence missing")
    expected_metric=[]
    partition_by_id={r["row_id"]:r for r in [*partition["fit_rows"],*partition["calibration_rows"]]}
    for phase_name,batch_name in (("PRECAL","precal"),("POSTCAL","postcal"),("POSTFIT","postfit")):
        for operation in training_schedule["batches"][batch_name]:
            for row_id in operation["row_ids"]:
                source=partition_by_id[row_id]; expected_metric.append({"phase":phase_name,"operation_index":operation["operation_index"],"arm":operation["arm"],"depth":operation["depth"],"row_id":row_id,"action_index":source["action_index"],"replicate":source["replicate"]})
    complete_metric_count=sum(len(row["row_ids"]) for row in done if row["phase"] in {"PRECAL","POSTCAL","POSTFIT"})
    max_metric_count=complete_metric_count
    if in_flight and operations[completed]["phase"] in {"PRECAL","POSTCAL","POSTFIT"} and delta_f: max_metric_count+=len(operations[completed]["row_ids"])
    metric_count=progress["metric_row_records_completed"]
    f3_incomplete=stage=="gate_evaluation" and completed==385 and metric_count==639 and progress["aggregate_records_completed"]==0 and progress["gates_evaluated"]==0
    if not (complete_metric_count<=metric_count<=max_metric_count or f3_incomplete) or progress["aggregate_records_completed"] not in ({0,135} if completed==385 and metric_count==640 else {0}) or progress["gates_evaluated"] not in ({0,9} if progress["aggregate_records_completed"]==135 else {0}): raise T0ContractError("T0 failure metric progress differs")
    if (value.get("metric_evidence") is None)!=(metric_count==0): raise T0ContractError("T0 failure metric evidence presence differs")
    candidate_prefix=progress["candidate_files_present"]
    if candidate_prefix!=CANDIDATE_FILES[:len(candidate_prefix)] or len(candidate_prefix)>5 or progress["threshold_file_present"] and (len(candidate_prefix)!=5 or progress["gates_evaluated"]!=9 or stage not in {"candidate_write","postflight_audit","terminal_publication"}): raise T0ContractError("T0 failure candidate prefix differs")
    derived=value.get("derived_thresholds")
    if not progress["threshold_file_present"]:
        if derived is not None: raise T0ContractError("T0 failure threshold evidence differs")
    else:
        if not isinstance(derived,dict) or set(derived)!={"path","file_sha256","bytes","payload"} or derived["path"]!="derived-validation-thresholds.json" or not digest_string(derived["file_sha256"]) or not isinstance(derived["bytes"],int) or derived["bytes"]<=0: raise T0ContractError("T0 failure threshold evidence differs")
        payload=derived["payload"]
        if not isinstance(payload,dict) or payload.get("threshold_sha256")!=sha256_bytes(canonical_json({k:v for k,v in payload.items() if k!="threshold_sha256"})): raise T0ContractError("T0 failure threshold payload differs")
    gate=value.get("gate_evidence")
    if progress["gates_evaluated"]:
        if not isinstance(gate,dict) or set(gate)!={"ordered_rows","pass_count","all_pass"} or len(gate["ordered_rows"])!=9 or [row.get("name") for row in gate["ordered_rows"]]!=GATE_NAMES: raise T0ContractError("T0 failure gate evidence differs")
    elif gate is not None:
        raise T0ContractError("T0 failure future gate evidence differs")
    raw_cache=value.get("cache_guard")
    if raw_cache is not None and (not isinstance(raw_cache,dict) or not raw_cache or stage_rank[stage]<stage_rank["capture"]): raise T0ContractError("T0 failure future cache evidence")
    cache=raw_cache if isinstance(raw_cache,dict) else {}
    entry_drift=negative_control_missing=config_drift=False
    safety=value.get("safety") if isinstance(value.get("safety"),dict) else {}
    protected=value.get("protected_state") if isinstance(value.get("protected_state"),dict) else {}
    if protected:
        _keys(protected,{"e33_disk_tree_before","e33_disk_tree_after","e33_state_before","e33_state_after","e33_metadata_before","e33_metadata_after","e33_grads_none","h176_tree_sha256","h176_loaded","e33_released"},"T0 failure protected state")
        if protected["e33_disk_tree_before"]!=E33_TREE_SHA256 or protected["e33_disk_tree_after"]!=E33_TREE_SHA256 or protected["e33_state_before"]!=E33_STATE_SHA256 or protected["e33_metadata_before"]!=METADATA_SHA256 or protected["e33_metadata_after"]!=METADATA_SHA256 or protected["h176_tree_sha256"]!=H176_TREE_SHA256 or not isinstance(protected["e33_state_after"],str) or not digest_string(protected["e33_state_after"]) or not isinstance(protected["e33_grads_none"],bool) or not isinstance(protected["h176_loaded"],bool) or not isinstance(protected["e33_released"],bool): raise T0ContractError("T0 failure protected state differs")
        if stage=="startup_pre_model" or stage=="capture" and protected["e33_released"] is not False or stage in {"candidate_init","preconnect","precal","train","postcal","postfit","gate_evaluation","candidate_write","postflight_audit","terminal_publication"} and protected["e33_released"] is not True: raise T0ContractError("T0 failure model release evidence differs")
        if protected["e33_released"] and protected["e33_state_after"]!=E33_STATE_SHA256: raise T0ContractError("T0 released model state differs")
    elif stage not in {"startup_pre_model","model_load"}:
        raise T0ContractError("T0 failure protected evidence missing")
    if cache:
        expected_cache_labels=["CACHE_ENTRY",*[x for i in range(96) for x in (f"CACHE_PRE_{i:03d}",f"CACHE_POST_{i:03d}")],"CACHE_EXIT"]
        cache_keys={"class_records","configuration_records","label_rows","label_sha256","expected_checks","actual_checks","dynamic_cache_negative_trips","dynamic_cache_actual_trips","returned_pkv_count","configuration_drift_count","restored"}
        if set(cache)!=cache_keys or cache["class_records"]!=CACHE_CLASS_RECORDS or cache["actual_checks"]!=cache_checks or [r.get("index") for r in cache["label_rows"]]!=list(range(cache_checks)) or [r.get("label") for r in cache["label_rows"]]!=expected_cache_labels[:cache_checks] or cache["label_sha256"]!=sha256_bytes(canonical_json(expected_cache_labels[:cache_checks])) or cache["expected_checks"]!=194 or not isinstance(cache["restored"],bool) or any(not isinstance(cache[key],int) or isinstance(cache[key],bool) or cache[key]<0 for key in ("dynamic_cache_negative_trips","dynamic_cache_actual_trips","returned_pkv_count","configuration_drift_count")): raise T0ContractError("T0 failure cache evidence differs")
        config_drift=stage=="capture" and cache_checks<194 and cache["configuration_drift_count"]==1 and cache["dynamic_cache_actual_trips"]==0 and cache["returned_pkv_count"]==0 and cache["dynamic_cache_negative_trips"]==(0 if cache_checks==0 else 1)
        entry_drift=config_drift and stage=="capture" and cache_checks==0
        negative_control_missing=stage=="capture" and cache_checks==1 and not any((captures,tokenizers,model_forwards)) and cache["dynamic_cache_negative_trips"]==0 and cache["dynamic_cache_actual_trips"]==0 and cache["returned_pkv_count"]==0 and cache["configuration_drift_count"]==0
        if config_drift:
            expected_by_source={row["source"]:row for row in CACHE_CONFIGURATION_RECORDS}; records=cache["configuration_records"]
            if not isinstance(records,list) or [row.get("source") for row in records]!=[row["source"] for row in CACHE_CONFIGURATION_RECORDS] or any(set(row)!={"source","value_before","value_during","value_after"} or row["value_before"]!=expected_by_source[row["source"]]["value_before"] or not isinstance(row["value_during"],bool) or not isinstance(row["value_after"],bool) for row in records) or not any(row["value_during"] is not False for row in records) or cache["restored"] is not all(row["value_after"]==row["value_before"] for row in records): raise T0ContractError("T0 failure cache entry drift evidence differs")
        elif negative_control_missing:
            if cache["configuration_records"]!=CACHE_CONFIGURATION_RECORDS or cache["restored"] is not True: raise T0ContractError("T0 failure cache negative control evidence differs")
        elif cache["configuration_records"]!=CACHE_CONFIGURATION_RECORDS or cache["dynamic_cache_negative_trips"]!=1 or cache["configuration_drift_count"]!=0 or cache["restored"] is not True:
            raise T0ContractError("T0 failure cache evidence differs")
    elif cache_checks: raise T0ContractError("T0 failure cache evidence missing")
    safety_keys={"network_attempts","validation_opens","heldout_opens","h176_loads","generation_calls","e33_backwards","e33_optimizer_steps","e33_updates","live_trajectory_count","object_census_errors","object_census_uninspectable","forbidden_inputs_detected"}
    if not safety or set(safety)!=safety_keys or any(not isinstance(safety[key],int) or isinstance(safety[key],bool) or safety[key]<0 for key in safety_keys-{"forbidden_inputs_detected"}) or not isinstance(safety["forbidden_inputs_detected"],list) or any(not isinstance(item,str) or not item for item in safety["forbidden_inputs_detected"]): raise T0ContractError("T0 failure safety differs")
    mechanism_positive=bool(cache.get("dynamic_cache_actual_trips",0)>0 or cache.get("returned_pkv_count",0)>0 or cache.get("configuration_drift_count",0)>0 or any(row.get("finite") is False for row in (value.get("capture_evidence") or {}).get("rows",[]) if isinstance(row,dict)))
    exposure_positive=bool(safety.get("validation_opens",0)>0 or safety.get("heldout_opens",0)>0 or safety.get("h176_loads",0)>0 or safety.get("generation_calls",0)>0 or safety.get("network_attempts",0)>0 or safety.get("e33_backwards",0)>0 or safety.get("e33_optimizer_steps",0)>0 or safety.get("e33_updates",0)>0 or safety.get("forbidden_inputs_detected") or protected.get("h176_loaded") is True or protected.get("e33_grads_none") is False or protected and protected.get("e33_state_after")!=protected.get("e33_state_before"))
    census_degraded=bool(safety.get("object_census_errors",0) or safety.get("object_census_uninspectable",0))
    if exposure_positive:
        if value["status"]!="h_iter_phase1_t0_exposure_boundary_rejected": raise T0ContractError("T0 positive exposure masked")
    elif mechanism_positive:
        if value["status"]!="h_iter_phase1_t0_capture_mechanism_rejected": raise T0ContractError("T0 positive capture rejection masked")
    elif value["status"] in {"h_iter_phase1_t0_exposure_boundary_rejected","h_iter_phase1_t0_capture_mechanism_rejected"}: raise T0ContractError("T0 rejection lacks positive evidence")
    if census_degraded and not exposure_positive and value["status"]!="infrastructure_invalid": raise T0ContractError("T0 census degradation masked")
    if value["status"]=="infrastructure_invalid" and (not isinstance(value["audit_errors"],list) or not value["audit_errors"] or any(not isinstance(item,str) or not item for item in value["audit_errors"])): raise T0ContractError("T0 infrastructure failure lacks audit evidence")
    if value["status"]!="infrastructure_invalid" and value["audit_errors"]!=[]: raise T0ContractError("T0 non-infrastructure audit errors differ")
    metric=value.get("metric_evidence")
    if isinstance(metric,dict) and isinstance(metric.get("row_records"),list):
        _keys(metric,{"row_records","aggregate_records","postcal_aggregate_sha256","counts"},"T0 failure metric")
        rows=metric["row_records"]
        if len(rows)!=progress["metric_row_records_completed"] or not isinstance(metric["aggregate_records"],list) or len(metric["aggregate_records"])!=progress["aggregate_records_completed"]: raise T0ContractError("T0 failure metric prefix differs")
        for record,identity in zip(rows,expected_metric):
            if set(record)!={"phase","operation_index","arm","depth","row_id","action_index","replicate","logits","logits_sha256","prediction","nll"} or any(record.get(k)!=v for k,v in identity.items()) or not isinstance(record.get("logits"),list) or len(record["logits"])!=4 or any(not finite_number(item) for item in record["logits"]): raise T0ContractError("T0 failure metric identity differs")
            try: import torch
            except Exception as error: raise T0ContractError("T0 failure metric validator requires CPU float64 runtime") from error
            vector=torch.tensor(record["logits"],dtype=torch.float64,device="cpu"); raw=vector.contiguous().view(torch.uint8).numpy().tobytes()
            if record.get("logits_sha256")!=sha256_bytes(canonical_json({"dtype":"torch.float64","shape":[4]})+b"\n"+raw) or record.get("prediction")!=int(torch.argmax(vector)) or record.get("nll")!=float(-torch.log_softmax(vector,dim=-1)[record["action_index"]]): raise T0ContractError("T0 failure metric logits differ")
        presentations={phase:sum(record["phase"]==phase for record in rows) for phase in ("PRECAL","POSTCAL","POSTFIT")}
        expected_counts={"row_records":len(rows),"aggregate_records":len(metric["aggregate_records"]),"precal_presentations":presentations["PRECAL"],"postcal_presentations":presentations["POSTCAL"],"postfit_presentations":presentations["POSTFIT"]}
        if metric["counts"]!=expected_counts: raise T0ContractError("T0 failure metric counts differ")
        if len(rows)==640:
            recomputed=[]
            for phase_name in ("PRECAL","POSTCAL","POSTFIT"):
                for arm in ARMS:
                    subset=[record for record in rows if record["phase"]==phase_name and record["arm"]==arm]
                    scopes=[("overall",None,subset),*[("depth",depth,[record for record in subset if record["depth"]==depth]) for depth in range(1,5)],*[("action",action,[record for record in subset if record["action_index"]==action]) for action in range(4)]]
                    for scope,scope_value,scope_rows in scopes:
                        correct=sum(record["prediction"]==record["action_index"] for record in scope_rows)
                        recomputed.append({"phase":phase_name,"arm":arm,"scope":scope,"scope_value":scope_value,"count":len(scope_rows),"correct":correct,"accuracy_numerator":correct,"accuracy_denominator":len(scope_rows),"mean_nll":math.fsum(record["nll"] for record in scope_rows)/len(scope_rows)})
            if progress["aggregate_records_completed"]==135 and metric["aggregate_records"]!=recomputed or progress["aggregate_records_completed"]==0 and metric["aggregate_records"]!=[]: raise T0ContractError("T0 failure aggregates differ")
            postcal=[record for record in recomputed if record["phase"]=="POSTCAL"]
            if metric["postcal_aggregate_sha256"]!=sha256_bytes(canonical_json(postcal if progress["aggregate_records_completed"] else [])): raise T0ContractError("T0 failure aggregate hash differs")
            if progress["gates_evaluated"]==9:
                get_gate=lambda phase,arm,scope="overall",scope_value=None:next(record for record in recomputed if (record["phase"],record["arm"],record["scope"],record["scope_value"])==(phase,arm,scope,scope_value))
                rec_gate=get_gate("POSTCAL","REC_K"); reset_gate=get_gate("POSTCAL","RESET_K"); static_gate=get_gate("POSTCAL","STATIC"); ffn_gate=get_gate("POSTCAL","FFN"); pre_gate=get_gate("PRECAL","REC_K"); fit_gate=get_gate("POSTFIT","REC_K")
                lhs=[[rec_gate["correct"],32],[rec_gate["correct"]-reset_gate["correct"],32],[rec_gate["correct"]-max(static_gate["correct"],ffn_gate["correct"]),32],[[get_gate("POSTCAL","REC_K","depth",depth)["correct"],8] for depth in range(1,5)],[[get_gate("POSTCAL","REC_K","action",action)["correct"],8] for action in range(4)],reset_gate["mean_nll"]-rec_gate["mean_nll"],min(static_gate["mean_nll"],ffn_gate["mean_nll"])-rec_gate["mean_nll"],rec_gate["mean_nll"]/pre_gate["mean_nll"],[fit_gate["correct"],64]]
                rhs=[[20,32],[4,32],[2,32],[4,8],[3,8],.05,.02,.75,[48,64]]; operators=[">=",">=",">=","all>=","all>=",">=",">=","<=",">="]
                passed=[lhs[0][0]>=20,lhs[1][0]>=4,lhs[2][0]>=2,all(item[0]>=4 for item in lhs[3]),all(item[0]>=3 for item in lhs[4]),lhs[5]>=.05,lhs[6]>=.02,lhs[7]<=.75,lhs[8][0]>=48]
                expected_gates={"ordered_rows":[{"index":index,"name":GATE_NAMES[index],"lhs":lhs[index],"operator":operators[index],"rhs":rhs[index],"passed":passed[index]} for index in range(9)],"pass_count":sum(passed),"all_pass":all(passed)}
                if gate!=expected_gates: raise T0ContractError("T0 failure gates do not recompute")
            if progress["threshold_file_present"]:
                get=lambda arm,scope="overall",scope_value=None:next(record for record in recomputed if (record["phase"],record["arm"],record["scope"],record["scope_value"])==("POSTCAL",arm,scope,scope_value))
                rec=get("REC_K"); reset=get("RESET_K"); ar=rec["correct"]/32; min_depth=min(get("REC_K","depth",depth)["correct"]/8 for depth in range(1,5)); a_reset=reset["correct"]/32; n_reset=reset["mean_nll"]; nr=rec["mean_nll"]
                formulas={"fixed_rec_nll_min":".01","per_depth_min":"min(18/24,max(10/24,floor_24(min_depth_AR-1/8)))","rec_accuracy_min":"min(36/48,max(24/48,floor_48(AR-4/32)))","rec_fixed_accuracy_min":"2/48","rec_reset_accuracy_min":"min(8/48,max(2/48,floor_48(.5*(AR-A_RESET))))","reset_rec_nll_min":"min(.15,max(.02,floor_1e6(.5*(N_RESET-NR))))"}
                thresholds={"rec_accuracy_min":[min(36,max(24,math.floor(48*(ar-4/32)))),48],"per_depth_min":[min(18,max(10,math.floor(24*(min_depth-1/8)))),24],"rec_reset_accuracy_min":[min(8,max(2,math.floor(48*.5*(ar-a_reset)))),48],"reset_rec_nll_min":min(.15,max(.02,math.floor(1_000_000*.5*(n_reset-nr))/1_000_000)),"rec_fixed_accuracy_min":[2,48],"fixed_rec_nll_min":.01}
                threshold_payload=derived["payload"]
                if threshold_payload.get("schema_version")!="prime-rl/latent-h-iter-phase1-t0-derived-validation-thresholds/v1" or threshold_payload.get("status")!="h_iter_phase1_t0_validation_thresholds_derived" or threshold_payload.get("mechanism")!=MECHANISM or threshold_payload.get("run_identity")!=RUN_ID or threshold_payload.get("postcal_aggregate_sha256")!=sha256_bytes(canonical_json(postcal)) or threshold_payload.get("inputs")!={"AR":ar,"min_depth_AR":min_depth,"A_RESET":a_reset,"N_RESET":n_reset,"NR":nr} or threshold_payload.get("formulas")!=formulas or threshold_payload.get("thresholds")!=thresholds: raise T0ContractError("T0 failure threshold formula differs")
    elif progress["metric_row_records_completed"]:
        raise T0ContractError("T0 failure metric evidence missing")
    memory=value.get("memory")
    if isinstance(memory,dict) and isinstance(memory.get("rows"),list):
        labels=[r["label"] for r in memory_schedule["rows"]]; rows=memory["rows"]
        if memory.get("schedule_file_sha256")!=sha256_bytes(canonical_json(memory_schedule)+b"\n") or memory.get("expected_count")!=534 or [r.get("index") for r in rows]!=list(range(len(rows))) or [r.get("label") for r in rows]!=labels[:len(rows)] or len(rows)>534 or memory.get("label_sha256")!=sha256_bytes(canonical_json(labels[:len(rows)])) or memory.get("complete") is not (len(rows)==534): raise T0ContractError("T0 failure memory prefix differs")
        memory_keys={"index","label","current_allocated_bytes","current_reserved_bytes","peak_allocated_bytes","peak_reserved_bytes"}
        for row in rows:
            numbers=[row.get(key) for key in ("current_allocated_bytes","current_reserved_bytes","peak_allocated_bytes","peak_reserved_bytes")]
            if set(row)!=memory_keys or any(not isinstance(item,int) or isinstance(item,bool) or not 0<=item<=40*(1<<30) for item in numbers) or numbers[0]>numbers[2] or numbers[1]>numbers[3]: raise T0ContractError("T0 failure memory row differs")
        validate_failure_memory_stage(stage,len(rows))
        row_labels=[r["label"] for r in rows]
        milestone_rules=(
            ("CACHE_GUARD_EXITED",captures==96,stage_rank[stage]>stage_rank["capture"]),
            ("PROTECTED_POSTCAPTURE_VERIFIED",stage_rank[stage]>=stage_rank["model_release"],stage_rank[stage]>stage_rank["model_release"]),
            ("MODEL_RELEASED",stage_rank[stage]>=stage_rank["candidate_init"],stage_rank[stage]>stage_rank["candidate_init"]),
            ("CANDIDATES_INITIALIZED",progress["candidates_initialized"]==5,stage_rank[stage]>stage_rank["candidate_init"]),
            ("PRECONNECT_COMPLETE",completed>=5,stage_rank[stage]>stage_rank["preconnect"]),
            ("PRECAL_COMPLETE",completed>=25,stage_rank[stage]>stage_rank["precal"]),
            ("TRAINING_COMPLETE",completed>=345,stage_rank[stage]>stage_rank["train"]),
            ("POSTCAL_COMPLETE",completed>=365,stage_rank[stage]>stage_rank["postcal"]),
            ("POSTFIT_COMPLETE",completed>=385,stage_rank[stage]>stage_rank["postfit"]),
            ("GATES_EVALUATED",progress["gates_evaluated"]==9,stage_rank[stage]>stage_rank["gate_evaluation"]),
            ("CANDIDATE_DISPOSITION_COMPLETE",stage_rank[stage]>=stage_rank["postflight_audit"],stage_rank[stage]>stage_rank["postflight_audit"]),
            ("PROTECTED_POSTFLIGHT_VERIFIED",stage_rank[stage]>=stage_rank["postflight_audit"],stage=="terminal_publication"),
            ("FULL_FREEZE_POSTFLIGHT_VERIFIED",stage_rank[stage]>=stage_rank["postflight_audit"],stage=="terminal_publication"),
            ("TERMINAL_PREWRITE",stage=="terminal_publication",stage=="terminal_publication"),
        )
        for label,allowed,required in milestone_rules:
            present=label in row_labels
            if present and not allowed or required and not present: raise T0ContractError("T0 failure memory milestone differs")
        train_labels=sum(label.startswith("POST_TRAIN_UPDATE_") for label in row_labels)
        expected_train_completed=max(0,min(completed,345)-25)
        if train_labels!=expected_train_completed: raise T0ContractError("T0 failure training memory prefix differs")
    elif captures or completed or progress["stage"] not in {"startup_pre_model","model_load"}: raise T0ContractError("T0 failure memory evidence missing")
    memory_count=len(memory["rows"]) if isinstance(memory,dict) and isinstance(memory.get("rows"),list) else 0
    validate_capture_failure_microstate(stage,captures,tokenizers,model_forwards,cache_checks,memory_count)
    if config_drift:
        drift_memory=(cache_checks==0 and memory_count==6) or (tokenizers==captures+1 and model_forwards==captures and cache_checks==1+2*captures and memory_count==9+2*captures) or (tokenizers==model_forwards==captures+1 and cache_checks==2+2*captures and memory_count==9+2*captures) or (captures==tokenizers==model_forwards==96 and cache_checks==193 and memory_count==201)
        if not drift_memory: raise T0ContractError("T0 failure cache drift transition differs")
    if negative_control_missing and memory_count!=6: raise T0ContractError("T0 failure cache entry memory differs")
    if memory_count and any(item is None for item in (runtime,asset_audit,antecedent,data,resources)): raise T0ContractError("T0 failure memory lacks foundational evidence")
    metric_rows=metric.get("row_records",[]) if isinstance(metric,dict) else []
    presentations={phase:sum(row.get("phase")==phase for row in metric_rows if isinstance(row,dict)) for phase in ("PRECAL","POSTCAL","POSTFIT")}
    tamper=value.get("tamper_audit")
    if tamper is not None:
        _keys(tamper,{"schedule_file_sha256","expected_count","results","rejected_count","all_rejected"},"T0 failure tamper")
        results=tamper["results"]
        if not digest_string(tamper["schedule_file_sha256"]) or tamper["expected_count"]!=98 or not isinstance(results,list) or len(results)>98 or any(set(row)!={"index","name","rejected","observed_error_type"} or row["index"]!=index or row["name"]!=_TAMPER_NAMES[index] or row["rejected"] is not True or not isinstance(row["observed_error_type"],str) or not row["observed_error_type"] for index,row in enumerate(results)) or tamper["rejected_count"]!=len(results) or tamper["all_rejected"] is not (len(results)==98): raise T0ContractError("T0 failure tamper evidence differs")
    full_freeze=value.get("full_freeze")
    freeze_exact=None
    if full_freeze is not None:
        _keys(full_freeze,{"target_count","head_before","head_after","tree_before","tree_after","clean_before","clean_after","pre_entries","post_entries","pre_sha256","post_sha256","complete"},"T0 failure full freeze")
        freeze_pre=full_freeze["pre_entries"]; freeze_post=full_freeze["post_entries"]
        if full_freeze["target_count"]!=38 or not isinstance(freeze_pre,list) or len(freeze_pre)!=38 or any(set(row)!=T0_ASSET_ROW_KEYS for row in freeze_pre) or full_freeze["pre_sha256"]!=sha256_bytes(canonical_json(freeze_pre)) or not isinstance(full_freeze["clean_before"],bool): raise T0ContractError("T0 failure freeze preflight differs")
        if freeze_post is not None and (not isinstance(freeze_post,list) or any(set(row)!=T0_ASSET_ROW_KEYS for row in freeze_post) or full_freeze["post_sha256"]!=sha256_bytes(canonical_json(freeze_post))): raise T0ContractError("T0 failure freeze postflight differs")
        if freeze_post is None and full_freeze["post_sha256"] is not None: raise T0ContractError("T0 failure freeze postflight null relation differs")
        if any(item is not None and (not isinstance(item,str) or len(item)!=40 or any(character not in "0123456789abcdef" for character in item)) for item in (full_freeze["head_before"],full_freeze["head_after"],full_freeze["tree_before"],full_freeze["tree_after"])) or not isinstance(full_freeze["clean_after"],(bool,type(None))): raise T0ContractError("T0 failure freeze identity differs")
        freeze_exact=bool(full_freeze["head_before"]==value["execution_commit"]==full_freeze["head_after"] and full_freeze["tree_before"]==full_freeze["tree_after"] and full_freeze["clean_before"] is True and full_freeze["clean_after"] is True and freeze_post==freeze_pre)
        if full_freeze["complete"] is not freeze_exact: raise T0ContractError("T0 failure freeze truth differs")
        if asset_audit is None or asset_audit["pre_entries"]!=freeze_pre or asset_audit["pre_sha256"]!=full_freeze["pre_sha256"] or asset_audit["post_entries"]!=freeze_post or asset_audit["post_sha256"]!=full_freeze["post_sha256"] or asset_audit["all_exact"] is not (freeze_post==freeze_pre): raise T0ContractError("T0 failure freeze/asset audit differs")
    decision=value.get("decision_boundary")
    if progress["gates_evaluated"]==9:
        if not isinstance(gate,dict): raise T0ContractError("T0 failure decision lacks gates")
        go=gate.get("all_pass") is True
        expected_decision={**T0_DECISION_BASE,"claim":"train_only_matched_sidecar_learning_screen_passed" if go else "train_only_matched_sidecar_learning_screen_stopped","validation_contract_design_authorized":go}
        if decision!=expected_decision: raise T0ContractError("T0 failure decision differs")
    elif decision is not None:
        raise T0ContractError("T0 failure future decision evidence differs")
    if stage_rank[stage]>=stage_rank["postflight_audit"] and any(item is None for item in (tamper,full_freeze)):
        raise T0ContractError("T0 failure postflight evidence missing")
    if stage_rank[stage]>=stage_rank["postflight_audit"] and resources["timing"] is None:
        raise T0ContractError("T0 failure postflight timing missing")
    if stage_rank[stage]<stage_rank["postflight_audit"] and (tamper is not None or full_freeze is not None or isinstance(asset_audit,dict) and asset_audit["post_entries"] is not None):
        raise T0ContractError("T0 failure future postflight evidence present")
    if stage=="terminal_publication" and (len(tamper["results"])!=98 or freeze_post is None): raise T0ContractError("T0 failure terminal audit incomplete")
    if freeze_exact is False and value["status"]!="infrastructure_invalid": raise T0ContractError("T0 failure freeze mismatch masked")
    expected_remaining={
        "memory_rows":memory_count,
        "optimizer_objects":5 if stage_rank[stage]>=stage_rank["precal"] else 0,
        "precal_presentations":presentations["PRECAL"],
        "train_presentations":16*(sum(row["phase"]=="TRAIN" for row in done)+(1 if in_flight and operations[completed]["phase"]=="TRAIN" and delta_f else 0)),
        "postcal_presentations":presentations["POSTCAL"],
        "postfit_presentations":presentations["POSTFIT"],
        "tamper_count":len(tamper["results"]) if isinstance(tamper,dict) else 0,
        "validation_opens":safety.get("validation_opens",0),
        "heldout_opens":safety.get("heldout_opens",0),
        "h176_loads":safety.get("h176_loads",0),
        "generation_calls":safety.get("generation_calls",0),
        "network_attempts":safety.get("network_attempts",0),
        "e33_backwards":safety.get("e33_backwards",0),
        "e33_updates":safety.get("e33_updates",0),
        "checkpoints":0,
        "live_trajectories":safety.get("live_trajectory_count",0),
    }
    if any(counts[key]!=expected_value for key,expected_value in expected_remaining.items()): raise T0ContractError("T0 failure derived counts differ")
    if output_inventory is not None:
        expected=sorted([*candidate_prefix,*( ["derived-validation-thresholds.json"] if progress["threshold_file_present"] else []),"T0-FAILURE.json"])
        if output_inventory!=expected: raise T0ContractError("T0 failure output inventory differs")
    disposition=value.get("candidate_disposition")
    if isinstance(disposition,dict):
        rows=disposition.get("rows")
        if disposition.get("arm_order")!=ARMS or disposition.get("files_present")!=candidate_prefix or disposition.get("threshold_file_present") is not progress["threshold_file_present"] or disposition.get("reusable") is not False or not isinstance(rows,list) or len(rows)!=len(candidate_prefix): raise T0ContractError("T0 failure candidate evidence differs")
        row_keys={"arm","path","file_sha256","bytes","module_state_sha256","parameter_names","parameter_name_count","parameter_count","dtype","device","finite","safe_reload_passed"}
        for arm,name,row in zip(ARMS,candidate_prefix,rows):
            if set(row)!=row_keys or row["arm"]!=arm or row["path"]!=name or row["parameter_names"]!=PARAMETER_NAMES or row["parameter_name_count"]!=16 or row["parameter_count"]!=366340 or row["dtype"]!="torch.float32" or row["device"]!="cpu" or row["finite"] is not True or row["safe_reload_passed"] is not True or not digest_string(row["file_sha256"]) or not digest_string(row["module_state_sha256"]) or not isinstance(row["bytes"],int) or row["bytes"]<=0: raise T0ContractError("T0 failure candidate row differs")
        if output_dir is not None:
            from pathlib import Path
            root=Path(output_dir)
            for name,row in zip(candidate_prefix,rows):
                if row.get("path")!=name: raise T0ContractError("T0 failure candidate path differs")
                validate_candidate_artifact(root/name,row)
            if progress["threshold_file_present"]:
                threshold_path=root/"derived-validation-thresholds.json"
                if not threshold_path.is_file() or threshold_path.is_symlink() or threshold_path.stat().st_size!=derived["bytes"] or sha256_bytes(threshold_path.read_bytes())!=derived["file_sha256"] or strict_loads(threshold_path.read_bytes())!=derived["payload"]: raise T0ContractError("T0 failure threshold file differs")
    else:
        raise T0ContractError("T0 failure candidate evidence missing")
