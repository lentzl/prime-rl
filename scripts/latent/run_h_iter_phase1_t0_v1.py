#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import resource
import signal
import socket
import subprocess
import sys
import time
import traceback
from unittest import mock
from pathlib import Path

from prime_rl.latent.h_iter_phase1_t0 import *

PLAN_REL="experiments/qwen35-2b-latent-workspace-v1/h-iter-phase1-t0-train-calibration-v1/t0-plan.json"
SIDECAR_REL="experiments/qwen35-2b-latent-workspace-v1/h-iter-phase1-t0-train-calibration-v1/t0-plan.sha256"
PROGRESS={"stage":"startup_pre_model","capture_rows_completed":0,"tokenizer_calls_completed":0,"model_forwards_completed":0,"sequences_completed":0,"cache_checks_completed":0,"candidates_initialized":0,"preconnect_arms_completed":0,"operations_completed":0,"current_operation_index":None,"current_phase":None,"current_arm":None,"current_epoch":None,"current_depth":None,"sidecar_forwards_completed":0,"sidecar_backwards_completed":0,"optimizer_steps_completed":0,"cell_calls_completed":0,"metric_row_records_completed":0,"aggregate_records_completed":0,"gates_evaluated":0,"candidate_files_present":[],"threshold_file_present":False}
FAILURE_EVIDENCE={"runtime":None,"asset_audit":None,"antecedent_binding":None,"data_binding":None,"resources":None,"full_freeze":None,"tamper_audit":None,"decision_boundary":None,"cache_guard":None,"protected_state":None,"capture_evidence":None,"metric_evidence":None,"gate_evidence":None,"candidate_initial_state":None,"preconnect_evidence":None,"memory":None,"safety":{"network_attempts":0,"validation_opens":0,"heldout_opens":0,"h176_loads":0,"generation_calls":0,"e33_backwards":0,"e33_optimizer_steps":0,"e33_updates":0,"live_trajectory_count":0,"object_census_errors":0,"object_census_uninspectable":0,"forbidden_inputs_detected":[]}}

def timeout_alarm(_signum:int,_frame:object)->None: raise TimeoutError("T0 phase timeout")
def git(repo:Path,*args:str)->str: return subprocess.check_output(["git",*args],cwd=repo,text=True).strip()
def file_sha(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()

def frozen_validation_sources(repo:Path)->dict:
    phase=repo/ARTIFACT_DIR
    mf0=repo/MF0_DIR
    return {
        "partition":strict_loads((mf0/"train-partition.json").read_bytes()),
        "capture_schedule":strict_loads((phase/"t0-capture-schedule.json").read_bytes()),
        "training_schedule":strict_loads((mf0/"training-schedule.json").read_bytes()),
        "memory_schedule":strict_loads((phase/"t0-memory-schedule.json").read_bytes()),
    }

def preflight(repo:Path,execution_commit:str,plan_file_sha256:str)->dict:
    plan_path=repo/PLAN_REL
    if not plan_path.is_file() or plan_path.is_symlink() or file_sha(plan_path)!=plan_file_sha256: raise RuntimeError("T0 final plan unavailable or differs")
    plan=strict_loads(plan_path.read_bytes()); validate_plan(plan,proof_input=False,repo=repo)
    sidecar=repo/SIDECAR_REL
    if not sidecar.is_file() or sidecar.is_symlink() or sidecar.read_bytes()!=plan_file_sha256.encode()+b"\n": raise RuntimeError("T0 final plan sidecar differs")
    assets=plan["asset_sha256"]
    entries={p:file_sha(repo/p) for p in sorted(assets)}
    if entries!=assets or len(entries)!=37: raise RuntimeError("T0 final assets differ")
    if git(repo,"rev-parse","HEAD")!=execution_commit or git(repo,"status","--porcelain","--untracked-files=all"): raise RuntimeError("T0 final freeze differs")
    validate_archive_bytes(repo,MF0_BINDING,kind="MF0"); validate_archive_bytes(repo,CAP0_BINDING,kind="CAP0-R1")
    antecedent=strict_loads((repo/ARTIFACT_DIR/"t0-antecedent-evidence-manifest.json").read_bytes())
    if antecedent!=build_antecedent_manifest(): raise RuntimeError("T0 antecedent manifest differs")
    source=(repo/"scripts/latent/run_h_iter_phase1_t0_v1.py").read_text(encoding="utf-8"); tree=ast.parse(source)
    if any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in {"generate","save_pretrained"} for node in ast.walk(tree)) or "validation-bank.json" in source or "heldout-bank.json" in source: raise RuntimeError("T0 static exposure guard differs")
    if Path(plan["remote_paths"]["t0_output"]).exists() or Path(plan["remote_paths"]["t0_output"]).is_symlink(): raise RuntimeError("T0 output namespace exists")
    if any(name in sys.modules for name in ("torch","transformers")): raise RuntimeError("T0 preflight imported modeling runtime")
    value={"schema_version":PREFLIGHT_SCHEMA,"status":"h_iter_phase1_t0_preflight_validated","mechanism":MECHANISM,"run_identity":RUN_ID,"execution_commit":execution_commit,"mechanism_code_commit":plan["mechanism_code_commit"],"tree_sha256":git(repo,"rev-parse","HEAD^{tree}"),"plan_file_sha256":plan_file_sha256,"plan_sha256":plan["plan_sha256"],"runtime":plan["runtime"],"asset_count":37,"assets_exact":True,"antecedents_exact":True,"static_contract_checks":True,"output_namespace_absent":not Path(plan["remote_paths"]["t0_output"]).exists(),"torch_imported":False,"tokenizer_loaded":False,"model_loaded":False,"scientific_exposure":False,"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"preflight_sha256":""}
    value["preflight_sha256"]=sha256_bytes(canonical_json({k:v for k,v in value.items() if k!="preflight_sha256"})); return value

def tensor_sha(torch:object,tensor:object)->str:
    header=canonical_json({"dtype":str(tensor.dtype),"shape":list(tensor.shape)})+b"\n"; raw=tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes(); return sha256_bytes(header+raw)

def batch_graph(torch:object,features:dict[str,object],bank_rows:dict[str,dict],row_ids:list[str])->tuple[object,object,object,object]:
    chunks=[]; successors=[]; starts=[]; targets=[]
    for graph_index,row_id in enumerate(row_ids):
        row=bank_rows[row_id]; receiver=row["receiver_input"]; ids=[n["node_id"] for n in receiver["nodes"]]; index={node:i for i,node in enumerate(ids)}; successor=[None]*24
        for edge in receiver["edges"]: successor[index[edge["source_node_id"]]]=index[edge["successor_node_id"]]
        if any(x is None for x in successor): raise T0ContractError("T0 graph successor differs")
        chunks.append(features[row_id]); successors.extend(graph_index*24+x for x in successor); starts.append(graph_index*24+index[receiver["start_node_id"]]); targets.append(row["action_index"])
    return torch.cat(chunks).to("cuda:0",dtype=torch.float32),torch.tensor(successors,device="cuda:0",dtype=torch.int64),torch.tensor(starts,device="cuda:0",dtype=torch.int64),torch.tensor(targets,device="cuda:0",dtype=torch.int64)

class MemoryLedger:
    def __init__(self,torch:object,labels:list[str],schedule_file_sha256:str)->None: self.torch=torch; self.labels=labels; self.schedule_file_sha256=schedule_file_sha256; self.rows=[]
    def add(self,label:str)->None:
        if label!=self.labels[len(self.rows)]: raise T0ContractError("T0 memory order differs")
        self.torch.cuda.synchronize(0); allocated=self.torch.cuda.memory_allocated(0); reserved=self.torch.cuda.memory_reserved(0); peak_a=self.torch.cuda.max_memory_allocated(0); peak_r=self.torch.cuda.max_memory_reserved(0)
        if max(allocated,reserved,peak_a,peak_r)>40*(1<<30): raise InfrastructureInvalid("T0 GPU memory cap exceeded")
        self.rows.append({"index":len(self.rows),"label":label,"current_allocated_bytes":allocated,"current_reserved_bytes":reserved,"peak_allocated_bytes":peak_a,"peak_reserved_bytes":peak_r})
        FAILURE_EVIDENCE["memory"]={"schedule_file_sha256":self.schedule_file_sha256,"expected_count":534,"rows":list(self.rows),"label_sha256":sha256_bytes(canonical_json([r["label"] for r in self.rows])),"complete":len(self.rows)==534}

class CacheGuard:
    def __init__(self,model:object,cache_utils:object,expected:list[dict])->None:
        self.model=model; self.cache_utils=cache_utils; self.expected=expected; self.stack=contextlib.ExitStack(); self.classes=set(); self.labels=[]; self.trips=0; self.negative_trips=0; self.in_negative_control=False; self.returned=0; self.drift=0; self.configs=[]; self.during={}; self.restored=False
        stack=[cache_utils.Cache]
        while stack:
            cls=stack.pop()
            if cls in self.classes: continue
            self.classes.add(cls); stack.extend(cls.__subclasses__())
        seen=set()
        for source,obj in [("model.config",model.config),("model.generation_config",getattr(model,"generation_config",None)),*((f"module:{n}.config",getattr(m,"config",None)) for n,m in model.named_modules())]:
            if obj is not None and id(obj) not in seen and hasattr(obj,"use_cache"): seen.add(id(obj)); self.configs.append((source,obj,getattr(obj,"use_cache")))
    def reject(self,cls:object,*args:object,**kwargs:object)->object:
        self.trips+=1
        if self.in_negative_control: self.negative_trips+=1
        FAILURE_EVIDENCE["cache_guard"]=self.evidence(); raise T0CaptureMechanismRejected(f"cache allocation {cls}")
    def check(self,label:str)->None:
        expected=["CACHE_ENTRY",*[x for i in range(96) for x in (f"CACHE_PRE_{i:03d}",f"CACHE_POST_{i:03d}")],"CACHE_EXIT"]
        if label!=expected[len(self.labels)]: raise T0ContractError("T0 cache label differs")
        self.during={id(config):getattr(config,"use_cache",None) for _,config,_ in self.configs}
        if any(value is not False for value in self.during.values()): self.drift+=1; FAILURE_EVIDENCE["cache_guard"]=self.evidence(); raise T0CaptureMechanismRejected("cache configuration drift")
        self.labels.append(label)
    def __enter__(self):
        try:
            for cls in sorted(self.classes,key=lambda c:(c.__module__,c.__qualname__)): self.stack.enter_context(mock.patch.object(cls,"__new__",lambda target,*a,_cls=cls,**k:self.reject(_cls,*a,**k)))
            for _,config,_ in self.configs: config.use_cache=False
            self.check("CACHE_ENTRY")
            self.in_negative_control=True
            try: self.cache_utils.DynamicCache()
            except T0CaptureMechanismRejected: pass
            finally: self.in_negative_control=False
            if self.negative_trips!=1 or self.trips!=1: raise T0ContractError("T0 cache negative control differs")
            PROGRESS["cache_checks_completed"]=len(self.labels)
            return self
        except BaseException:
            self.in_negative_control=False; self.stack.close()
            for _,config,before in self.configs: config.use_cache=before
            self.restored=all(getattr(config,"use_cache",None)==before for _,config,before in self.configs)
            FAILURE_EVIDENCE["cache_guard"]=self.evidence() if self.drift or self.trips or self.labels else None; PROGRESS["cache_checks_completed"]=len(self.labels)
            raise
    def __exit__(self,*args):
        try:
            if len(self.labels)==193: self.check("CACHE_EXIT")
        finally:
            self.stack.close()
            for _,config,before in self.configs: config.use_cache=before
            self.restored=all(getattr(c,"use_cache",None)==before for _,c,before in self.configs)
            FAILURE_EVIDENCE["cache_guard"]=self.evidence()
            PROGRESS["cache_checks_completed"]=len(self.labels)
    def evidence(self)->dict:
        records=[]
        for cls in sorted(self.classes,key=lambda c:(c.__module__,c.__qualname__)):
            module=sys.modules[cls.__module__]; path=Path(module.__file__).resolve(); dist="flash-linear-attention" if cls.__module__.startswith("fla") else "transformers"; records.append({"fqcn":f"{cls.__module__}.{cls.__qualname__}","module_path":str(path),"module_sha256":file_sha(path),"distribution":f"{dist}=={importlib.metadata.version(dist)}"})
        return {"class_records":records,"configuration_records":[{"source":source,"value_before":before,"value_during":self.during.get(id(config),False),"value_after":getattr(config,"use_cache",None)} for source,config,before in self.configs],"label_rows":[{"index":i,"label":x} for i,x in enumerate(self.labels)],"label_sha256":sha256_bytes(canonical_json(self.labels)),"expected_checks":194,"actual_checks":len(self.labels),"dynamic_cache_negative_trips":self.negative_trips,"dynamic_cache_actual_trips":max(0,self.trips-self.negative_trips),"returned_pkv_count":self.returned,"configuration_drift_count":self.drift,"restored":self.restored}

def aggregate_metrics(records:list[dict])->tuple[list[dict],dict]:
    aggregates=[]
    for phase in ("PRECAL","POSTCAL","POSTFIT"):
        for arm in ARMS:
            subset=[r for r in records if r["phase"]==phase and r["arm"]==arm]
            scopes=[("overall",None,subset),*[("depth",d,[r for r in subset if r["depth"]==d]) for d in range(1,5)],*[("action",a,[r for r in subset if r["action_index"]==a]) for a in range(4)]]
            for scope,value,rows in scopes:
                correct=sum(r["prediction"]==r["action_index"] for r in rows); aggregates.append({"phase":phase,"arm":arm,"scope":scope,"scope_value":value,"count":len(rows),"correct":correct,"accuracy_numerator":correct,"accuracy_denominator":len(rows),"mean_nll":math.fsum(r["nll"] for r in rows)/len(rows)})
    get=lambda p,a,s="overall",v=None:next(r for r in aggregates if (r["phase"],r["arm"],r["scope"],r["scope_value"])==(p,a,s,v))
    rec=get("POSTCAL","REC_K"); reset=get("POSTCAL","RESET_K"); static=get("POSTCAL","STATIC"); ffn=get("POSTCAL","FFN"); pre=get("PRECAL","REC_K"); fit=get("POSTFIT","REC_K")
    lhs=[[rec["correct"],32],[rec["correct"]-reset["correct"],32],[rec["correct"]-max(static["correct"],ffn["correct"]),32],[[get("POSTCAL","REC_K","depth",d)["correct"],8] for d in range(1,5)],[[get("POSTCAL","REC_K","action",a)["correct"],8] for a in range(4)],reset["mean_nll"]-rec["mean_nll"],min(static["mean_nll"],ffn["mean_nll"])-rec["mean_nll"],rec["mean_nll"]/pre["mean_nll"],[fit["correct"],64]]
    rhs=[[20,32],[4,32],[2,32],[4,8],[3,8],.05,.02,.75,[48,64]]
    passed=[lhs[0][0]>=20,lhs[1][0]>=4,lhs[2][0]>=2,all(x[0]>=4 for x in lhs[3]),all(x[0]>=3 for x in lhs[4]),lhs[5]>=.05,lhs[6]>=.02,lhs[7]<=.75,lhs[8][0]>=48]
    gates={"ordered_rows":[{"index":i,"name":GATE_NAMES[i],"lhs":lhs[i],"operator":[">=",">=",">=","all>=","all>=",">=",">=","<=",">="][i],"rhs":rhs[i],"passed":passed[i]} for i in range(9)],"pass_count":sum(passed),"all_pass":all(passed)}
    return aggregates,gates

def atomic_write(path:Path,value:dict)->None:
    data=canonical_json(value)+b"\n"; tmp=path.with_name("."+path.name+".tmp"); fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        with os.fdopen(fd,"wb") as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp,path); dfd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
    except BaseException:
        try: tmp.unlink()
        except FileNotFoundError: pass
        raise

def run_full(repo:Path,execution_commit:str,plan_file_sha256:str)->None:
    phase_start=time.monotonic_ns(); compute_enter=0; signal.signal(signal.SIGALRM,timeout_alarm); signal.alarm(17999)
    FAILURE_EVIDENCE["_phase_start_ns"]=phase_start; FAILURE_EVIDENCE["_compute_enter_ns"]=compute_enter
    plan=strict_loads((repo/PLAN_REL).read_bytes()); output=Path(OUTPUT_ROOT)
    if output.exists() or output.is_symlink(): raise InfrastructureInvalid("T0 output namespace exists")
    phase=repo/ARTIFACT_DIR; mf0=repo/"experiments/qwen35-2b-latent-workspace-v1/h-iter-phase1-train-calibration-v1"
    asset_pre=[{"path":p,"sha256":file_sha(repo/p),"bytes":(repo/p).stat().st_size} for p in sorted(plan["asset_sha256"])]
    if {r["path"]:r["sha256"] for r in asset_pre}!=plan["asset_sha256"]: raise InfrastructureInvalid("T0 asset preflight differs")
    asset_pre_sha=sha256_bytes(canonical_json(asset_pre)); FAILURE_EVIDENCE["asset_audit"]={"target_count":37,"pre_entries":asset_pre,"pre_sha256":asset_pre_sha,"post_entries":None,"post_sha256":None,"all_exact":None}
    load=lambda path:strict_loads(Path(path).read_bytes())
    bank=load(repo/TRAIN_BANK_PATH); partition=load(mf0/"train-partition.json"); schedule=load(mf0/"training-schedule.json"); capture_schedule=load(phase/"t0-capture-schedule.json"); memory_schedule=load(phase/"t0-memory-schedule.json"); tampers=load(phase/"t0-tamper-schedule.json")
    validate_assets(bank,partition,schedule,capture_schedule,memory_schedule,tampers); bank_rows={r["row_id"]:r for r in bank["rows"]}
    antecedent_manifest=load(phase/"t0-antecedent-evidence-manifest.json"); validate_archive_bytes(repo,MF0_BINDING,kind="MF0"); validate_archive_bytes(repo,CAP0_BINDING,kind="CAP0-R1")
    FAILURE_EVIDENCE["antecedent_binding"]={"manifest_path":f"{ARTIFACT_DIR}/t0-antecedent-evidence-manifest.json","manifest_file_sha256":file_sha(phase/"t0-antecedent-evidence-manifest.json"),"manifest_internal_sha256":antecedent_manifest["manifest_sha256"],"mf0_archive_exact":True,"cap0_r1_archive_exact":True}
    FAILURE_EVIDENCE["data_binding"]={"train_bank_path":TRAIN_BANK_PATH,"train_bank_file_sha256":TRAIN_BANK_FILE_SHA256,"train_bank_internal_sha256":TRAIN_BANK_INTERNAL_SHA256,"source_rows":96,"fit_rows":64,"calibration_rows":32,"fit_row_ids":[r["row_id"] for r in partition["fit_rows"]],"calibration_row_ids":[r["row_id"] for r in partition["calibration_rows"]],"fit_calibration_intersection":[],"complete_train_union":True,"validation_open_count":0,"heldout_open_count":0}
    network_attempts=[0]
    def deny_network(*_args,**_kwargs): network_attempts[0]+=1; FAILURE_EVIDENCE["safety"]["network_attempts"]=network_attempts[0]; raise T0ExposureBoundaryRejected("T0 network operation denied")
    for owner,name in ((socket.socket,"connect"),(socket.socket,"connect_ex"),(socket,"create_connection"),(socket,"getaddrinfo")): setattr(owner,name,deny_network)
    sys.addaudithook(lambda event,args: deny_network() if event in {"socket.connect","socket.getaddrinfo"} else None)
    import torch
    import transformers
    import transformers.cache_utils as cache_utils
    import transformers.models.qwen3_5.modeling_qwen3_5  # noqa: F401
    import fla.models.utils  # noqa: F401
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    free_gpu_pre=int(torch.cuda.mem_get_info(0)[0]); available_ram_pre=next(int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")); free_disk_pre=int(__import__("shutil").disk_usage(output.parent).free)
    if free_gpu_pre<44*(1<<30) or available_ram_pre<64*(1<<30) or free_disk_pre<16*(1<<30): raise InfrastructureInvalid("T0 runtime resources differ")
    if torch.cuda.is_initialized() and torch.cuda.current_device()!=0: raise InfrastructureInvalid("T0 visible CUDA device differs")
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.cuda.reset_peak_memory_stats(0)
    FAILURE_EVIDENCE["runtime"]={"python":".".join(map(str,sys.version_info[:3])),"sys_executable":sys.executable,"sys_prefix":sys.prefix,"transformers":transformers.__version__,"tokenizers":importlib.metadata.version("tokenizers"),"torch_distribution":importlib.metadata.version("torch"),"torch_runtime":torch.__version__,"flash_linear_attention":importlib.metadata.version("flash-linear-attention"),"gpu_name":torch.cuda.get_device_name(0),"physical_gpu":"0","visible_device":"cuda:0","shared_project_pyproject_sha256":"504907808f992f1e6883f54c2695a4814ae77d6b80814239cbfc98d81a543656","shared_project_uv_lock_sha256":"fca5fa6183345b5b68974078c38d58e0320f79eef13a695af11ceab12fdf36d5"}
    FAILURE_EVIDENCE["resources"]={"gpu_name":torch.cuda.get_device_name(0),"physical_gpu":"0","visible_device":"cuda:0","free_gpu_bytes_pre":free_gpu_pre,"available_ram_bytes_pre":available_ram_pre,"free_disk_bytes_pre":free_disk_pre,"max_allocated_bytes":0,"max_reserved_bytes":0,"artifact_bytes":0,"timing":None}
    ledger=MemoryLedger(torch,[r["label"] for r in memory_schedule["rows"]],file_sha(phase/"t0-memory-schedule.json")); ledger.add("RUNTIME_VERIFIED"); ledger.add("FULL_FREEZE_PREFLIGHT_VERIFIED"); ledger.add("ANTECEDENTS_VALIDATED"); ledger.add("TRAIN_INPUTS_VALIDATED"); ledger.add("PROTECTED_PREFLIGHT_VERIFIED")
    e33=Path(E33_PATH); h176=Path(H176_PATH)
    if file_sha(e33/"model.safetensors")!=E33_TREE_SHA256 or file_sha(h176/"model.safetensors")!=H176_TREE_SHA256: raise InfrastructureInvalid("T0 protected disk differs")
    e33_meta={n:file_sha(e33/n) for n in METADATA_SHA256}; h176_meta={n:file_sha(h176/n) for n in METADATA_SHA256}
    if e33_meta!=METADATA_SHA256 or h176_meta!=METADATA_SHA256: raise InfrastructureInvalid("T0 protected metadata differs")
    PROGRESS["stage"]="model_load"; tokenizer=AutoTokenizer.from_pretrained(E33_PATH,local_files_only=True); tokenizer.padding_side="left"
    model=AutoModelForImageTextToText.from_pretrained(E33_PATH,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation="eager").to("cuda:0"); model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    state_before=module_state_sha256(torch,dict(model.state_dict()))
    if state_before!=E33_STATE_SHA256: raise InfrastructureInvalid("T0 loaded model state differs")
    FAILURE_EVIDENCE["protected_state"]={"e33_disk_tree_before":E33_TREE_SHA256,"e33_disk_tree_after":E33_TREE_SHA256,"e33_state_before":state_before,"e33_state_after":state_before,"e33_metadata_before":e33_meta,"e33_metadata_after":e33_meta,"e33_grads_none":True,"h176_tree_sha256":H176_TREE_SHA256,"h176_loaded":False,"e33_released":False}
    ledger.add("MODEL_LOADED_FROZEN")
    source_capture=load(mf0/"capture-contract.json"); expected_classes=source_capture["cache_guard"]["class_closure"]; features={}; capture_rows=[]
    PROGRESS["stage"]="capture"
    with CacheGuard(model,cache_utils,expected_classes) as guard:
        ledger.add("CACHE_GUARD_ENTERED")
        feature_bytes=0; ledger.add("CAPTURE_STORAGE_INITIALIZED")
        for item in capture_schedule["rows"]:
            index=item["capture_index"]; row=bank_rows[item["row_id"]]; texts=[node["local_text"] for node in row["receiver_input"]["nodes"]]
            if len(texts)!=24 or any(len(x.encode())!=68 for x in texts): raise T0ContractError("T0 local text geometry differs")
            encoded=tokenizer(texts,add_special_tokens=True,padding="max_length",max_length=128,truncation=False,return_tensors="pt")
            PROGRESS.update({"tokenizer_calls_completed":index+1,"sequences_completed":24*(index+1)})
            ids=encoded.input_ids; mask=encoded.attention_mask
            if ids.shape!=(24,128) or mask.shape!=(24,128) or ids.dtype!=torch.int64 or mask.dtype!=torch.int64: raise T0ContractError("T0 token tensor differs")
            lengths=mask.sum(1); 
            if int(lengths.min())<1 or int(lengths.max())>128: raise T0ContractError("T0 token length differs")
            ledger.add(f"PRE_CAPTURE_{index:03d}_{item['row_id']}"); guard.check(f"CACHE_PRE_{index:03d}"); PROGRESS["cache_checks_completed"]=len(guard.labels)
            with torch.inference_mode(): result=model(input_ids=ids.to("cuda:0"),attention_mask=mask.to("cuda:0"),use_cache=False,output_hidden_states=True,return_dict=True,logits_to_keep=1)
            PROGRESS["model_forwards_completed"]=index+1
            if result.past_key_values is not None: guard.returned+=1; FAILURE_EVIDENCE["cache_guard"]=guard.evidence(); raise T0CaptureMechanismRejected("T0 returned PKV")
            hidden=result.hidden_states[-1][:,-1,:].detach().cpu().contiguous(); hidden_finite=bool(torch.isfinite(hidden).all())
            if hidden.shape!=(24,2048) or hidden.dtype!=torch.bfloat16 or not hidden_finite:
                failed_row={"capture_index":index,"row_id":item["row_id"],"row_sha256":item["row_sha256"],"receiver_input_sha256":item["receiver_input_sha256"],"node_count":24,"token_count_min":int(lengths.min()),"token_count_max":int(lengths.max()),"input_ids_sha256":tensor_sha(torch,ids),"attention_mask_sha256":tensor_sha(torch,mask),"hidden_shape":list(hidden.shape),"hidden_dtype":str(hidden.dtype),"hidden_sha256":tensor_sha(torch,hidden),"finite":hidden_finite}
                FAILURE_EVIDENCE["capture_evidence"]={"schedule_sha256":capture_schedule["schedule_sha256"],"rows":[*capture_rows,failed_row],"counts":{"rows":len(capture_rows),"tokenizer_calls":PROGRESS["tokenizer_calls_completed"],"model_forwards":PROGRESS["model_forwards_completed"],"sequences":PROGRESS["sequences_completed"]},"aggregate_sha256":sha256_bytes(canonical_json([*capture_rows,failed_row]))}
                if hidden.shape!=(24,2048): raise T0CaptureMechanismRejected("T0 hidden capture geometry differs")
                if hidden.dtype!=torch.bfloat16: raise T0CaptureMechanismRejected("T0 hidden capture dtype differs")
                raise T0CaptureMechanismRejected("T0 hidden capture nonfinite")
            features[item["row_id"]]=hidden; feature_bytes+=hidden.numel()*hidden.element_size()
            capture_rows.append({"capture_index":index,"row_id":item["row_id"],"row_sha256":item["row_sha256"],"receiver_input_sha256":item["receiver_input_sha256"],"node_count":24,"token_count_min":int(lengths.min()),"token_count_max":int(lengths.max()),"input_ids_sha256":tensor_sha(torch,ids),"attention_mask_sha256":tensor_sha(torch,mask),"hidden_shape":[24,2048],"hidden_dtype":"torch.bfloat16","hidden_sha256":tensor_sha(torch,hidden),"finite":True})
            FAILURE_EVIDENCE["capture_evidence"]={"schedule_sha256":capture_schedule["schedule_sha256"],"rows":list(capture_rows),"counts":{"rows":len(capture_rows),"tokenizer_calls":PROGRESS["tokenizer_calls_completed"],"model_forwards":PROGRESS["model_forwards_completed"],"sequences":PROGRESS["sequences_completed"]},"aggregate_sha256":sha256_bytes(canonical_json(capture_rows))}
            guard.check(f"CACHE_POST_{index:03d}"); PROGRESS["cache_checks_completed"]=len(guard.labels); ledger.add(f"POST_CAPTURE_{index:03d}_{item['row_id']}")
            PROGRESS.update({"capture_rows_completed":index+1,"cache_checks_completed":len(guard.labels)})
        ledger.add("CACHE_GUARD_EXITED")
    cache_evidence=guard.evidence()
    PROGRESS["cache_checks_completed"]=cache_evidence["actual_checks"]
    if cache_evidence["class_records"]!=expected_classes or cache_evidence["configuration_records"]!=CACHE_CONFIGURATION_RECORDS or cache_evidence["actual_checks"]!=194 or cache_evidence["dynamic_cache_actual_trips"]!=0 or cache_evidence["returned_pkv_count"]!=0 or cache_evidence["configuration_drift_count"]!=0 or not cache_evidence["restored"]: raise T0CaptureMechanismRejected("T0 cache closure differs")
    PROGRESS["stage"]="model_release"; state_after=module_state_sha256(torch,dict(model.state_dict()))
    if state_after!=state_before or any(p.grad is not None for p in model.parameters()):
        FAILURE_EVIDENCE["protected_state"]={"e33_disk_tree_before":E33_TREE_SHA256,"e33_disk_tree_after":E33_TREE_SHA256,"e33_state_before":state_before,"e33_state_after":state_after,"e33_metadata_before":e33_meta,"e33_metadata_after":{n:file_sha(e33/n) for n in METADATA_SHA256},"e33_grads_none":all(p.grad is None for p in model.parameters()),"h176_tree_sha256":H176_TREE_SHA256,"h176_loaded":False,"e33_released":False}
        raise T0ExposureBoundaryRejected("T0 protected model changed")
    ledger.add("PROTECTED_POSTCAPTURE_VERIFIED"); del model,tokenizer,encoded,ids,mask,result,hidden,guard; gc.collect(); torch.cuda.empty_cache()
    remaining_models=0; census_errors=0
    for obj in gc.get_objects():
        try:
            module=getattr(type(obj),"__module__",""); name=getattr(type(obj),"__name__","")
            if isinstance(module,str) and module.startswith("transformers.models.qwen3_5") and isinstance(name,str) and "ForConditionalGeneration" in name: remaining_models+=1
        except Exception: census_errors+=1
    if remaining_models or census_errors: raise InfrastructureInvalid("T0 model release census differs")
    ledger.add("MODEL_RELEASED")
    FAILURE_EVIDENCE["protected_state"]={"e33_disk_tree_before":E33_TREE_SHA256,"e33_disk_tree_after":E33_TREE_SHA256,"e33_state_before":state_before,"e33_state_after":state_after,"e33_metadata_before":e33_meta,"e33_metadata_after":{n:file_sha(e33/n) for n in METADATA_SHA256},"e33_grads_none":True,"h176_tree_sha256":H176_TREE_SHA256,"h176_loaded":False,"e33_released":True}
    PROGRESS["stage"]="candidate_init"; Candidate=candidate_class(torch); torch.manual_seed(INIT_SEED); canonical=Candidate().float(); initial={n:t.detach().clone() for n,t in canonical.state_dict().items()}; candidates={}
    for arm in ARMS:
        candidate=Candidate().to("cuda:0",dtype=torch.float32); candidate.load_state_dict(initial,strict=True); candidates[arm]=candidate
        PROGRESS["candidates_initialized"]=len(candidates)
    del canonical; ledger.add("CANDIDATES_INITIALIZED")
    names=list(initial); shapes=[{"name":n,"shape":list(initial[n].shape),"dtype":str(initial[n].dtype)} for n in names]; initial_hash=module_state_sha256(torch,initial)
    initial_rows=[{"arm":a,"parameter_names":names,"parameter_shapes":shapes,"parameter_name_count":16,"parameter_count":366340,"device":"cuda:0","dtype":"torch.float32","initial_state_sha256":initial_hash,"finite":True} for a in ARMS]
    FAILURE_EVIDENCE["candidate_initial_state"]={"seed_payload":INIT_PAYLOAD,"seed_payload_sha256":INIT_SHA256,"seed_u64_be":INIT_SEED,"arm_order":ARMS,"action_order":ACTIONS,"rows":initial_rows,"all_equal":True}
    pre_rows=[]
    preop=schedule["batches"]["preconnect"]
    PROGRESS["stage"]="preconnect"
    for op in preop:
        arm=op["arm"]; PROGRESS.update({"current_operation_index":op["operation_index"],"current_phase":"PRECONNECT","current_arm":arm,"current_epoch":None,"current_depth":4}); item=candidates[arm]; f,s,starts,target=batch_graph(torch,features,bank_rows,op["row_ids"]); before=module_state_sha256(torch,dict(item.state_dict())); logits=item(f,s,arm,4,starts)
        PROGRESS["sidecar_forwards_completed"]+=1; PROGRESS["cell_calls_completed"]+=0 if arm=="STATIC" else 1 if arm=="FFN" else 4
        loss=torch.nn.functional.cross_entropy(logits,target); loss.backward(); PROGRESS["sidecar_backwards_completed"]+=1
        def gradnorm(prefixes):
            values=[p.grad for n,p in item.named_parameters() if n.startswith(prefixes)]
            if any(x is None for x in values): return None
            return math.sqrt(sum(float(x.double().square().sum()) for x in values))
        codec,cell,readout=gradnorm(("codec_",)),gradnorm(("self_norm","message_norm","cell_","post_norm")),gradnorm(("readout",)); after=module_state_sha256(torch,dict(item.state_dict())); qualifies=codec is not None and codec>=1e-8 and readout is not None and readout>=1e-8 and ((arm=="STATIC" and cell is None) or (arm!="STATIC" and cell is not None and cell>=1e-8)) and before==after
        pre_rows.append({"arm":arm,"operation_index":op["operation_index"],"loss":float(loss.detach().cpu()),"codec_grad_l2":codec,"cell_grad_l2":cell,"readout_grad_l2":readout,"state_before_sha256":before,"state_after_sha256":after,"gradients_cleared":True,"finite":True,"qualifies":qualifies}); item.zero_grad(set_to_none=True)
        PROGRESS["preconnect_arms_completed"]+=1; PROGRESS["operations_completed"]=op["operation_index"]+1
    if not all(r["qualifies"] for r in pre_rows): raise T0ContractError("T0 preconnectivity differs")
    ledger.add("PRECONNECT_COMPLETE")
    FAILURE_EVIDENCE["preconnect_evidence"]={"row_id":"hi_be342227610f62e0","depth":4,"action_index":0,"rows":pre_rows,"counts":{"forwards":5,"backwards":5,"optimizer_steps":0},"all_qualify":True}
    optimizers={}
    try:
        for arm in ARMS: optimizers[arm]=torch.optim.AdamW(candidates[arm].parameters(),lr=.001,betas=(.9,.95),eps=1e-8,weight_decay=.01)
    except BaseException:
        optimizers.clear(); gc.collect(); raise
    PROGRESS["stage"]="precal"
    metric_rows=[]; cell_calls=PROGRESS["cell_calls_completed"]; forwards=PROGRESS["sidecar_forwards_completed"]; backwards=PROGRESS["sidecar_backwards_completed"]; steps=0
    def operation(op:dict,training:bool,record:bool)->None:
        nonlocal cell_calls,forwards,backwards,steps
        arm=op["arm"]; depth=op["depth"]; item=candidates[arm]; f,s,starts,target=batch_graph(torch,features,bank_rows,op["row_ids"])
        PROGRESS.update({"stage":op["phase"].lower(),"current_operation_index":op["operation_index"],"current_phase":op["phase"],"current_arm":arm,"current_epoch":op.get("epoch"),"current_depth":depth})
        if training:
            optimizers[arm].zero_grad(set_to_none=True); logits=item(f,s,arm,depth,starts); forwards+=1; cell_calls+=0 if arm=="STATIC" else 1 if arm=="FFN" else 4 if arm=="FIXED_T4" else depth; PROGRESS.update({"sidecar_forwards_completed":forwards,"cell_calls_completed":cell_calls}); loss=torch.nn.functional.cross_entropy(logits,target); loss.backward(); backwards+=1; PROGRESS["sidecar_backwards_completed"]=backwards; torch.nn.utils.clip_grad_norm_(item.parameters(),1.0); optimizers[arm].step(); steps+=1; PROGRESS["optimizer_steps_completed"]=steps
        else:
            with torch.no_grad(): logits=item(f,s,arm,depth,starts)
            forwards+=1; cell_calls+=0 if arm=="STATIC" else 1 if arm=="FFN" else 4 if arm=="FIXED_T4" else depth; PROGRESS.update({"sidecar_forwards_completed":forwards,"cell_calls_completed":cell_calls})
        if record:
            cpu=logits.detach().cpu().double()
            for row_id,vector in zip(op["row_ids"],cpu):
                row=bank_rows[row_id]; nll=float(-torch.log_softmax(vector,dim=-1)[row["action_index"]]); pred=int(torch.argmax(vector)); metric_rows.append({"phase":op["phase"],"operation_index":op["operation_index"],"arm":arm,"depth":depth,"row_id":row_id,"action_index":row["action_index"],"replicate":row["replicate"],"logits":[float(x) for x in vector.tolist()],"logits_sha256":tensor_sha(torch,vector),"prediction":pred,"nll":nll}); PROGRESS["metric_row_records_completed"]=len(metric_rows)
                FAILURE_EVIDENCE["metric_evidence"]={"row_records":list(metric_rows),"aggregate_records":[],"postcal_aggregate_sha256":sha256_bytes(canonical_json([])),"counts":{"row_records":len(metric_rows),"aggregate_records":0,"precal_presentations":sum(r["phase"]=="PRECAL" for r in metric_rows),"postcal_presentations":sum(r["phase"]=="POSTCAL" for r in metric_rows),"postfit_presentations":sum(r["phase"]=="POSTFIT" for r in metric_rows)}}
        PROGRESS.update({"operations_completed":op["operation_index"]+1,"sidecar_forwards_completed":forwards,"sidecar_backwards_completed":backwards,"optimizer_steps_completed":steps,"cell_calls_completed":cell_calls,"metric_row_records_completed":len(metric_rows)})
    for op in schedule["batches"]["precal"]: operation(op,False,True)
    ledger.add("PRECAL_COMPLETE")
    for op in schedule["batches"]["train"]: operation(op,True,False); ledger.add(f"POST_TRAIN_UPDATE_{op['update_index']:03d}_{op['arm']}_E{op['epoch']:02d}_D{op['depth']}")
    ledger.add("TRAINING_COMPLETE")
    for op in schedule["batches"]["postcal"]: operation(op,False,True)
    ledger.add("POSTCAL_COMPLETE")
    for op in schedule["batches"]["postfit"]: operation(op,False,True)
    ledger.add("POSTFIT_COMPLETE")
    PROGRESS["stage"]="gate_evaluation"; aggregates,gates=aggregate_metrics(metric_rows); PROGRESS.update({"aggregate_records_completed":len(aggregates),"gates_evaluated":9}); FAILURE_EVIDENCE["metric_evidence"]={"row_records":metric_rows,"aggregate_records":aggregates,"postcal_aggregate_sha256":sha256_bytes(canonical_json([r for r in aggregates if r["phase"]=="POSTCAL"])),"counts":{"row_records":640,"aggregate_records":135,"precal_presentations":160,"postcal_presentations":160,"postfit_presentations":320}}; FAILURE_EVIDENCE["gate_evidence"]=gates; ledger.add("GATES_EVALUATED"); go=gates["all_pass"]
    FAILURE_EVIDENCE["decision_boundary"]={"claim":"train_only_matched_sidecar_learning_screen_passed" if go else "train_only_matched_sidecar_learning_screen_stopped","candidate_modules_updated":True,"protected_models_updated":False,"validation_contract_design_authorized":go,"validation_execution_authorized":False,"validation_or_heldout_opened":False,"live_trajectory_count":0,"admission":False,"nomination":False,"promotion":False,"four_live_floor_unchanged":True}
    output.mkdir(mode=0o700,parents=True); candidate_rows=[]
    if go:
        PROGRESS["stage"]="candidate_write"
        from safetensors.torch import load_file,save_file
        for arm in ARMS:
            name=f"candidate-{arm}.safetensors"; tmp=output/("."+name+".tmp"); final=output/name; cpu_state={n:t.detach().cpu().contiguous() for n,t in candidates[arm].state_dict().items()}
            try:
                save_file(cpu_state,str(tmp))
                fd=os.open(tmp,os.O_RDONLY|os.O_NOFOLLOW)
                try: os.fsync(fd)
                finally: os.close(fd)
                os.replace(tmp,final); dfd=os.open(output,os.O_RDONLY|os.O_DIRECTORY)
                try: os.fsync(dfd)
                finally: os.close(dfd)
            except BaseException:
                try: tmp.unlink()
                except FileNotFoundError: pass
                raise
            loaded=load_file(str(final),device="cpu")
            candidate_rows.append({"arm":arm,"path":name,"file_sha256":file_sha(final),"bytes":final.stat().st_size,"module_state_sha256":module_state_sha256(torch,loaded),"parameter_names":PARAMETER_NAMES,"parameter_name_count":16,"parameter_count":366340,"dtype":"torch.float32","device":"cpu","finite":all(torch.isfinite(x).all() for x in loaded.values()),"safe_reload_passed":True})
            PROGRESS["candidate_files_present"].append(name)
        get=lambda arm,scope="overall",value=None:next(r for r in aggregates if r["phase"]=="POSTCAL" and r["arm"]==arm and r["scope"]==scope and r["scope_value"]==value)
        rec,reset=get("REC_K"),get("RESET_K"); ar=rec["correct"]/32; min_depth=min(get("REC_K","depth",d)["correct"]/8 for d in range(1,5)); a_reset=reset["correct"]/32; nr=rec["mean_nll"]; n_reset=reset["mean_nll"]
        floor=lambda n,x:math.floor(n*x)/n; formula=load(mf0/"threshold-builder-contract.json")["validation_threshold_formulas"]
        threshold_values={"rec_accuracy_min":[min(36,max(24,math.floor(48*(ar-4/32)))),48],"per_depth_min":[min(18,max(10,math.floor(24*(min_depth-1/8)))),24],"rec_reset_accuracy_min":[min(8,max(2,math.floor(48*.5*(ar-a_reset)))),48],"reset_rec_nll_min":min(.15,max(.02,floor(1000000,.5*(n_reset-nr)))),"rec_fixed_accuracy_min":[2,48],"fixed_rec_nll_min":.01}
        thresholds={"schema_version":"prime-rl/latent-h-iter-phase1-t0-derived-validation-thresholds/v1","status":"h_iter_phase1_t0_validation_thresholds_derived","mechanism":MECHANISM,"run_identity":RUN_ID,"postcal_aggregate_sha256":sha256_bytes(canonical_json([r for r in aggregates if r["phase"]=="POSTCAL"])),"inputs":{"AR":ar,"min_depth_AR":min_depth,"A_RESET":a_reset,"N_RESET":n_reset,"NR":nr},"formulas":formula,"thresholds":threshold_values,"threshold_sha256":""}; thresholds["threshold_sha256"]=sha256_bytes(canonical_json({k:v for k,v in thresholds.items() if k!="threshold_sha256"})); atomic_write(output/"derived-validation-thresholds.json",thresholds)
        PROGRESS["threshold_file_present"]=True
    ledger.add("CANDIDATE_DISPOSITION_COMPLETE")
    signal.alarm(0); compute_exit=time.monotonic_ns()-phase_start; audit_enter=compute_exit; signal.alarm(1199)
    FAILURE_EVIDENCE["_compute_exit_ns"]=compute_exit; FAILURE_EVIDENCE["_audit_enter_ns"]=audit_enter
    status="h_iter_phase1_t0_validation_contract_design_authorized" if go else "h_iter_phase1_train_calibration_stop"; decision={"claim":"train_only_matched_sidecar_learning_screen_passed" if go else "train_only_matched_sidecar_learning_screen_stopped","candidate_modules_updated":True,"protected_models_updated":False,"validation_contract_design_authorized":go,"validation_execution_authorized":False,"validation_or_heldout_opened":False,"live_trajectory_count":0,"admission":False,"nomination":False,"promotion":False,"four_live_floor_unchanged":True}
    asset_post=[{"path":p,"sha256":file_sha(repo/p),"bytes":(repo/p).stat().st_size} for p in sorted(plan["asset_sha256"])]; asset_hash=sha256_bytes(canonical_json(asset_pre)); runtime={"python":".".join(map(str,sys.version_info[:3])),"sys_executable":sys.executable,"sys_prefix":sys.prefix,"transformers":transformers.__version__,"tokenizers":importlib.metadata.version("tokenizers"),"torch_distribution":importlib.metadata.version("torch"),"torch_runtime":torch.__version__,"flash_linear_attention":importlib.metadata.version("flash-linear-attention"),"gpu_name":torch.cuda.get_device_name(0),"physical_gpu":"0","visible_device":"cuda:0","shared_project_pyproject_sha256":"504907808f992f1e6883f54c2695a4814ae77d6b80814239cbfc98d81a543656","shared_project_uv_lock_sha256":"fca5fa6183345b5b68974078c38d58e0320f79eef13a695af11ceab12fdf36d5"}
    proof={"schema_version":PROOF_SCHEMA,"status":status,"mechanism":MECHANISM,"run_identity":RUN_ID,"execution_commit":execution_commit,"mechanism_code_commit":plan["mechanism_code_commit"],"plan_file_sha256":plan_file_sha256,"plan_sha256":plan["plan_sha256"],"runtime":runtime,"asset_audit":{"target_count":37,"pre_entries":asset_pre,"pre_sha256":asset_hash,"post_entries":asset_post,"post_sha256":sha256_bytes(canonical_json(asset_post)),"all_exact":asset_pre==asset_post},"antecedent_binding":{"manifest_path":f"{ARTIFACT_DIR}/t0-antecedent-evidence-manifest.json","manifest_file_sha256":file_sha(phase/"t0-antecedent-evidence-manifest.json"),"manifest_internal_sha256":antecedent_manifest["manifest_sha256"],"mf0_archive_exact":True,"cap0_r1_archive_exact":True},"data_binding":{"train_bank_path":TRAIN_BANK_PATH,"train_bank_file_sha256":TRAIN_BANK_FILE_SHA256,"train_bank_internal_sha256":TRAIN_BANK_INTERNAL_SHA256,"source_rows":96,"fit_rows":64,"calibration_rows":32,"fit_row_ids":[r["row_id"] for r in partition["fit_rows"]],"calibration_row_ids":[r["row_id"] for r in partition["calibration_rows"]],"fit_calibration_intersection":[],"complete_train_union":True,"validation_open_count":0,"heldout_open_count":0},"capture_evidence":{"schedule_sha256":capture_schedule["schedule_sha256"],"rows":capture_rows,"counts":{"rows":96,"tokenizer_calls":96,"model_forwards":96,"sequences":2304},"aggregate_sha256":sha256_bytes(canonical_json(capture_rows))},"candidate_initial_state":{"seed_payload":INIT_PAYLOAD,"seed_payload_sha256":INIT_SHA256,"seed_u64_be":INIT_SEED,"arm_order":ARMS,"action_order":ACTIONS,"rows":initial_rows,"all_equal":True},"preconnect_evidence":{"row_id":"hi_be342227610f62e0","depth":4,"action_index":0,"rows":pre_rows,"counts":{"forwards":5,"backwards":5,"optimizer_steps":0},"all_qualify":True},"metric_evidence":{"row_records":metric_rows,"aggregate_records":aggregates,"postcal_aggregate_sha256":sha256_bytes(canonical_json([r for r in aggregates if r["phase"]=="POSTCAL"])),"counts":{"row_records":640,"aggregate_records":135,"precal_presentations":160,"postcal_presentations":160,"postfit_presentations":320}},"gate_evidence":gates,"derived_thresholds":None if not go else {"path":"derived-validation-thresholds.json","file_sha256":file_sha(output/"derived-validation-thresholds.json"),"bytes":int((output/"derived-validation-thresholds.json").stat().st_size),"payload":thresholds},"candidate_disposition":{"arm_order":ARMS,"files_present":CANDIDATE_FILES if go else [],"threshold_file_present":go,"rows":candidate_rows,"reusable":go},"protected_state":{"e33_disk_tree_before":E33_TREE_SHA256,"e33_disk_tree_after":E33_TREE_SHA256,"e33_state_before":state_before,"e33_state_after":state_after,"e33_metadata_before":e33_meta,"e33_metadata_after":{n:file_sha(e33/n) for n in METADATA_SHA256},"e33_grads_none":True,"h176_tree_sha256":H176_TREE_SHA256,"h176_loaded":False,"e33_released":True},"cache_guard":cache_evidence,"safety":{"network_attempts":network_attempts[0],"validation_opens":0,"heldout_opens":0,"h176_loads":0,"generation_calls":0,"e33_backwards":0,"e33_optimizer_steps":0,"e33_updates":0,"live_trajectory_count":0,"object_census_errors":0,"object_census_uninspectable":0,"forbidden_inputs_detected":[]},"counts":{**T0_COMPLETE_COUNTS,"candidate_files":5 if go else 0,"threshold_files":1 if go else 0},"resources":{"gpu_name":torch.cuda.get_device_name(0),"physical_gpu":"0","visible_device":"cuda:0","free_gpu_bytes_pre":free_gpu_pre,"available_ram_bytes_pre":available_ram_pre,"free_disk_bytes_pre":free_disk_pre,"max_allocated_bytes":torch.cuda.max_memory_allocated(0),"max_reserved_bytes":torch.cuda.max_memory_reserved(0),"artifact_bytes":sum(p.stat().st_size for p in output.iterdir()),"timing":{}},"memory":{"schedule_file_sha256":file_sha(phase/"t0-memory-schedule.json"),"expected_count":534,"rows":ledger.rows,"label_sha256":sha256_bytes(canonical_json([r["label"] for r in ledger.rows])),"complete":True},"full_freeze":{"target_count":37,"head_before":execution_commit,"head_after":git(repo,"rev-parse","HEAD"),"tree_before":git(repo,"rev-parse","HEAD^{tree}"),"tree_after":git(repo,"rev-parse","HEAD^{tree}"),"clean_before":True,"clean_after":not bool(git(repo,"status","--porcelain","--untracked-files=all")),"pre_entries":asset_pre,"post_entries":asset_post,"pre_sha256":asset_hash,"post_sha256":sha256_bytes(canonical_json(asset_post)),"complete":asset_pre==asset_post},"tamper_audit":{"schedule_file_sha256":file_sha(phase/"t0-tamper-schedule.json"),"expected_count":98,"results":[{"index":r["index"],"name":r["name"],"rejected":True,"observed_error_type":"T0ContractError"} for r in tampers["rows"]],"rejected_count":98,"all_rejected":True},"decision_boundary":decision,"proof_sha256":""}
    from run_h_iter_phase1_t0_model_free_proof_v1 import tamper_results
    tamper_rows=tamper_results(plan,load(phase/"t0-contract.json"),capture_schedule,schedule,tampers,partition,memory_schedule,repo,plan_proof_input=False)
    proof["tamper_audit"]={"schedule_file_sha256":file_sha(phase/"t0-tamper-schedule.json"),"expected_count":98,"results":tamper_rows,"rejected_count":sum(r["rejected"] for r in tamper_rows),"all_rejected":all(r["rejected"] for r in tamper_rows)}
    proof["full_freeze"]["complete"]=bool(proof["full_freeze"]["head_before"]==proof["full_freeze"]["head_after"] and proof["full_freeze"]["tree_before"]==proof["full_freeze"]["tree_after"] and proof["full_freeze"]["clean_before"] is True and proof["full_freeze"]["clean_after"] is True and proof["full_freeze"]["pre_entries"]==proof["full_freeze"]["post_entries"])
    FAILURE_EVIDENCE["runtime"]=runtime; FAILURE_EVIDENCE["asset_audit"]=proof["asset_audit"]; FAILURE_EVIDENCE["resources"]=dict(proof["resources"]); FAILURE_EVIDENCE["full_freeze"]=proof["full_freeze"]; FAILURE_EVIDENCE["tamper_audit"]=proof["tamper_audit"]
    PROGRESS["stage"]="postflight_audit"
    if not proof["asset_audit"]["all_exact"] or not proof["full_freeze"]["complete"] or not proof["tamper_audit"]["all_rejected"]: raise InfrastructureInvalid("T0 postflight freeze differs")
    if file_sha(e33/"model.safetensors")!=E33_TREE_SHA256 or file_sha(h176/"model.safetensors")!=H176_TREE_SHA256 or {n:file_sha(e33/n) for n in METADATA_SHA256}!=METADATA_SHA256 or {n:file_sha(h176/n) for n in METADATA_SHA256}!=METADATA_SHA256 or state_after!=E33_STATE_SHA256: raise InfrastructureInvalid("T0 protected postflight differs")
    ledger.add("PROTECTED_POSTFLIGHT_VERIFIED"); ledger.add("FULL_FREEZE_POSTFLIGHT_VERIFIED"); ledger.add("TERMINAL_PREWRITE")
    proof["memory"]={"schedule_file_sha256":file_sha(phase/"t0-memory-schedule.json"),"expected_count":534,"rows":ledger.rows,"label_sha256":sha256_bytes(canonical_json([r["label"] for r in ledger.rows])),"complete":True}
    signal.alarm(0); audit_exit=time.monotonic_ns()-phase_start; terminal_enter=audit_exit; signal.alarm(299)
    proof["resources"]["timing"]={"outer_seconds":21600,"startup_seconds":600,"compute_seconds":18000,"audit_seconds":1200,"failure_seconds":1200,"terminal_seconds":300,"postexit_seconds":300,"alarm_safety_margin_seconds":1,"compute_enter_ns":compute_enter,"compute_exit_ns":compute_exit,"compute_duration_ns":compute_exit-compute_enter,"audit_enter_ns":audit_enter,"audit_exit_ns":audit_exit,"audit_duration_ns":audit_exit-audit_enter,"failure_enter_ns":None,"failure_exit_ns":None,"failure_duration_ns":None,"terminal_enter_ns":terminal_enter,"prepublication_elapsed_ns":terminal_enter}
    PROGRESS["stage"]="terminal_publication"; proof["proof_sha256"]=sha256_bytes(canonical_json({k:v for k,v in proof.items() if k!="proof_sha256"})); validation=frozen_validation_sources(repo); validate_t0_proof(proof,output_dir=output,**validation); atomic_write(output/"T0-PROOF.json",proof); validate_t0_proof(strict_loads((output/"T0-PROOF.json").read_bytes()),output_inventory=sorted(p.name for p in output.iterdir()),output_dir=output,**validation); signal.alarm(0)

def publish_failure(repo:Path,execution_commit:str,plan_file_sha256:str,error:BaseException)->None:
    signal.alarm(0); signal.signal(signal.SIGALRM,timeout_alarm); signal.alarm(1199)
    output=Path(OUTPUT_ROOT)
    if output.is_symlink() or (output.exists() and (not output.is_dir() or (output/"T0-PROOF.json").exists() or (output/"T0-FAILURE.json").exists())): return
    output.mkdir(mode=0o700,parents=True,exist_ok=True)
    if isinstance(error,T0CaptureMechanismRejected): status,error_type="h_iter_phase1_t0_capture_mechanism_rejected","T0CaptureMechanismRejected"
    elif isinstance(error,T0ExposureBoundaryRejected): status,error_type="h_iter_phase1_t0_exposure_boundary_rejected","T0ExposureBoundaryRejected"
    elif isinstance(error,T0ContractError): status,error_type="h_iter_phase1_t0_incomplete","T0ContractError"
    else: status,error_type="infrastructure_invalid","InfrastructureInvalid"
    try: plan=strict_loads((repo/PLAN_REL).read_bytes())
    except Exception:
        try: plan=strict_loads(subprocess.check_output(["git","show",f"{execution_commit}:{PLAN_REL}"],cwd=repo))
        except Exception: return
    candidate_rows=[]
    derived_thresholds=None
    if PROGRESS["candidate_files_present"]:
        try:
            import torch
            from safetensors.torch import load_file
            for name in PROGRESS["candidate_files_present"]:
                path=output/name; loaded=load_file(str(path),device="cpu"); arm=name[len("candidate-"):-len(".safetensors")]
                candidate_rows.append({"arm":arm,"path":name,"file_sha256":file_sha(path),"bytes":path.stat().st_size,"module_state_sha256":module_state_sha256(torch,loaded),"parameter_names":PARAMETER_NAMES,"parameter_name_count":16,"parameter_count":366340,"dtype":"torch.float32","device":"cpu","finite":all(torch.isfinite(x).all() for x in loaded.values()),"safe_reload_passed":True})
        except Exception as scan_error:
            status,error_type="infrastructure_invalid","InfrastructureInvalid"; error=scan_error
    if PROGRESS["threshold_file_present"]:
        try:
            threshold_path=output/"derived-validation-thresholds.json"; threshold_payload=strict_loads(threshold_path.read_bytes())
            derived_thresholds={"path":threshold_path.name,"file_sha256":file_sha(threshold_path),"bytes":threshold_path.stat().st_size,"payload":threshold_payload}
        except Exception as scan_error:
            status,error_type="infrastructure_invalid","InfrastructureInvalid"; error=scan_error
    resources=FAILURE_EVIDENCE["resources"]
    if isinstance(resources,dict):
        resources=dict(resources)
        try:
            import torch
            resources["max_allocated_bytes"]=int(torch.cuda.max_memory_allocated(0)); resources["max_reserved_bytes"]=int(torch.cuda.max_memory_reserved(0))
        except Exception: pass
        try: resources["artifact_bytes"]=sum(path.stat().st_size for path in output.iterdir() if path.is_file() and not path.is_symlink())
        except Exception: pass
    capture_evidence=FAILURE_EVIDENCE["capture_evidence"]
    if isinstance(capture_evidence,dict):
        capture_evidence=dict(capture_evidence)
        evidence_rows=capture_evidence.get("rows",[]); completed_rows=PROGRESS["capture_rows_completed"]
        persisted_rows=capture_failure_evidence_row_count(evidence_rows,completed_rows,PROGRESS["model_forwards_completed"])
        capture_evidence["counts"]={"rows":persisted_rows,"tokenizer_calls":PROGRESS["tokenizer_calls_completed"],"model_forwards":PROGRESS["model_forwards_completed"],"sequences":PROGRESS["sequences_completed"]}
    memory=FAILURE_EVIDENCE["memory"]; metric=FAILURE_EVIDENCE["metric_evidence"]; tamper=FAILURE_EVIDENCE["tamper_audit"]; safety=FAILURE_EVIDENCE["safety"]
    metric_rows=metric.get("row_records",[]) if isinstance(metric,dict) else []
    stage_rank={name:index for index,name in enumerate(["startup_pre_model","model_load","capture","model_release","candidate_init","preconnect","precal","train","postcal","postfit","gate_evaluation","candidate_write","postflight_audit","terminal_publication"])}
    counts={
        "capture_rows":PROGRESS["capture_rows_completed"],"tokenizer_calls":PROGRESS["tokenizer_calls_completed"],"model_forwards":PROGRESS["model_forwards_completed"],"sequences":PROGRESS["sequences_completed"],"cache_checks":PROGRESS["cache_checks_completed"],"memory_rows":len(memory["rows"]) if isinstance(memory,dict) else 0,"candidate_objects":PROGRESS["candidates_initialized"],"optimizer_objects":5 if stage_rank[PROGRESS["stage"]]>=stage_rank["precal"] else 0,"sidecar_forwards":PROGRESS["sidecar_forwards_completed"],"sidecar_backwards":PROGRESS["sidecar_backwards_completed"],"optimizer_steps":PROGRESS["optimizer_steps_completed"],"cell_calls":PROGRESS["cell_calls_completed"],"precal_presentations":sum(row.get("phase")=="PRECAL" for row in metric_rows),"train_presentations":16*(max(0,min(PROGRESS["operations_completed"],345)-25)+(1 if PROGRESS["current_operation_index"]==PROGRESS["operations_completed"] and PROGRESS["current_phase"]=="TRAIN" and PROGRESS["sidecar_forwards_completed"]>PROGRESS["operations_completed"] else 0)),"postcal_presentations":sum(row.get("phase")=="POSTCAL" for row in metric_rows),"postfit_presentations":sum(row.get("phase")=="POSTFIT" for row in metric_rows),"tamper_count":len(tamper["results"]) if isinstance(tamper,dict) else 0,"candidate_files":len(PROGRESS["candidate_files_present"]),"threshold_files":int(PROGRESS["threshold_file_present"]),"validation_opens":safety["validation_opens"],"heldout_opens":safety["heldout_opens"],"h176_loads":safety["h176_loads"],"generation_calls":safety["generation_calls"],"network_attempts":safety["network_attempts"],"e33_backwards":safety["e33_backwards"],"e33_updates":safety["e33_updates"],"checkpoints":0,"live_trajectories":safety["live_trajectory_count"],
    }
    failure={k:None for k in T0_PROOF_KEYS if k!="proof_sha256"}
    failure.update({"schema_version":FAILURE_SCHEMA,"status":status,"mechanism":MECHANISM,"run_identity":RUN_ID,"execution_commit":execution_commit,"mechanism_code_commit":plan.get("mechanism_code_commit"),"plan_file_sha256":plan_file_sha256,"plan_sha256":plan.get("plan_sha256"),"runtime":FAILURE_EVIDENCE["runtime"],"asset_audit":FAILURE_EVIDENCE["asset_audit"],"antecedent_binding":FAILURE_EVIDENCE["antecedent_binding"],"data_binding":FAILURE_EVIDENCE["data_binding"],"capture_evidence":capture_evidence,"candidate_initial_state":FAILURE_EVIDENCE["candidate_initial_state"],"preconnect_evidence":FAILURE_EVIDENCE["preconnect_evidence"],"metric_evidence":metric,"gate_evidence":FAILURE_EVIDENCE["gate_evidence"],"cache_guard":FAILURE_EVIDENCE["cache_guard"],"protected_state":FAILURE_EVIDENCE["protected_state"],"safety":safety,"counts":counts,"resources":resources,"memory":memory,"full_freeze":FAILURE_EVIDENCE["full_freeze"],"tamper_audit":tamper,"decision_boundary":FAILURE_EVIDENCE["decision_boundary"],"derived_thresholds":derived_thresholds,"candidate_disposition":{"arm_order":ARMS,"files_present":PROGRESS["candidate_files_present"],"threshold_file_present":PROGRESS["threshold_file_present"],"rows":candidate_rows,"reusable":False},"error_type":error_type,"error_message":str(error),"traceback":"".join(traceback.format_exception(error)),"stage":PROGRESS["stage"],"execution_progress":dict(PROGRESS),"audit_errors":[str(error)] if status=="infrastructure_invalid" else [],"failure_sha256":""})
    validation=frozen_validation_sources(repo)
    if isinstance(resources,dict) and FAILURE_EVIDENCE.get("_phase_start_ns") is not None:
        failure_enter=time.monotonic_ns()-FAILURE_EVIDENCE["_phase_start_ns"]
        compute_start=FAILURE_EVIDENCE.get("_compute_enter_ns",0); recorded_compute_exit=FAILURE_EVIDENCE.get("_compute_exit_ns")
        if recorded_compute_exit is None: compute_exit=failure_enter; audit_enter=audit_exit=None
        else: compute_exit=recorded_compute_exit; audit_enter=FAILURE_EVIDENCE.get("_audit_enter_ns",compute_exit); audit_exit=failure_enter
        terminal_enter=time.monotonic_ns()-FAILURE_EVIDENCE["_phase_start_ns"]
        resources["timing"]={"outer_seconds":21600,"startup_seconds":600,"compute_seconds":18000,"audit_seconds":1200,"failure_seconds":1200,"terminal_seconds":300,"postexit_seconds":300,"alarm_safety_margin_seconds":1,"compute_enter_ns":compute_start,"compute_exit_ns":compute_exit,"compute_duration_ns":compute_exit-compute_start,"audit_enter_ns":audit_enter,"audit_exit_ns":audit_exit,"audit_duration_ns":None if audit_enter is None else audit_exit-audit_enter,"failure_enter_ns":failure_enter,"failure_exit_ns":terminal_enter,"failure_duration_ns":terminal_enter-failure_enter,"terminal_enter_ns":terminal_enter,"prepublication_elapsed_ns":terminal_enter}
    failure["failure_sha256"]=sha256_bytes(canonical_json({k:v for k,v in failure.items() if k!="failure_sha256"})); validate_t0_failure(failure,**validation); signal.alarm(0); signal.alarm(299); atomic_write(output/"T0-FAILURE.json",failure); validate_t0_failure(strict_loads((output/"T0-FAILURE.json").read_bytes()),output_inventory=sorted(p.name for p in output.iterdir()),output_dir=output,**validation); signal.alarm(0)

def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--repo",type=Path,required=True); parser.add_argument("--execution-commit",required=True); parser.add_argument("--plan-file-sha256",required=True); parser.add_argument("--run-id",required=True); parser.add_argument("--preflight-only",action="store_true"); parser.add_argument("--full",action="store_true"); parser.add_argument("--validate-terminal",action="store_true")
    args=parser.parse_args(); repo=args.repo.resolve(strict=True)
    if args.run_id!=RUN_ID or sum((args.preflight_only,args.full,args.validate_terminal))!=1: raise RuntimeError("T0 invocation differs")
    if args.validate_terminal:
        output=Path(OUTPUT_ROOT); inventory=sorted(p.name for p in output.iterdir())
        validation=frozen_validation_sources(repo)
        if "T0-PROOF.json" in inventory and "T0-FAILURE.json" not in inventory: validate_t0_proof(strict_loads((output/"T0-PROOF.json").read_bytes()),output_inventory=inventory,output_dir=output,**validation); return
        if "T0-FAILURE.json" in inventory: validate_t0_failure(strict_loads((output/"T0-FAILURE.json").read_bytes()),output_inventory=inventory,output_dir=output,**validation); return
        raise RuntimeError("T0 terminal inventory differs")
    receipt=preflight(repo,args.execution_commit,args.plan_file_sha256)
    if args.preflight_only: print(canonical_json(receipt).decode()); return
    try: run_full(repo,args.execution_commit,args.plan_file_sha256)
    except BaseException as error:
        publish_failure(repo,args.execution_commit,args.plan_file_sha256,error)
        raise

if __name__=="__main__": main()
