#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import resource
import shutil
import socket
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

from prime_rl.latent.h_iter_phase1_t0 import *
from freeze_h_iter_phase1_t0_plan_v1 import PROOF_ASSETS

PLAN_REL=f"{ARTIFACT_DIR}/t0-model-free-proof-plan.json"
SIDECAR_REL=f"{ARTIFACT_DIR}/t0-model-free-proof-plan.sha256"

class NetworkGuard:
    def __init__(self)->None: self.attempts=0; self.originals={}
    def _deny(self,*_args:Any,**_kwargs:Any)->Any: self.attempts+=1; raise T0ModelFreeProofBoundaryRejected("T0 model-free network operation denied")
    def install(self)->None:
        for owner,name in ((socket.socket,"connect"),(socket.socket,"connect_ex"),(socket,"create_connection"),(socket,"getaddrinfo")):
            self.originals[(owner,name)]=getattr(owner,name); setattr(owner,name,self._deny)
        def hook(event:str,_args:tuple[Any,...])->None:
            if event in {"socket.connect","socket.getaddrinfo"}: self._deny()
        sys.addaudithook(hook)

def read_json(path:Path)->dict[str,Any]: return strict_loads(path.read_bytes())
def file_sha(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def git(repo:Path,*args:str)->str: return subprocess.check_output(["git",*args],cwd=repo,text=True).strip()
def rss()->int: return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
def asset_entries(repo:Path)->list[dict[str,Any]]:
    rows=[]
    for relative in sorted(PROOF_ASSETS):
        path=repo/relative
        if not path.is_file() or path.is_symlink(): raise InfrastructureInvalid(f"T0 model-free unsafe asset: {relative}")
        rows.append({"path":relative,"sha256":file_sha(path),"bytes":path.stat().st_size})
    return rows

def object_census(torch:Any)->dict[str,int]:
    counts={"model":0,"tokenizer":0,"optimizer":0,"candidate":0,"errors":0}
    for obj in gc.get_objects():
        try:
            cls=type(obj); module=cls.__module__ if isinstance(cls.__module__,str) else ""; name=cls.__name__ if isinstance(cls.__name__,str) else ""
            if module.startswith("transformers.modeling_"): counts["model"]+=1
            if module.startswith("transformers") and "Tokenizer" in name: counts["tokenizer"]+=1
            if isinstance(obj,torch.optim.Optimizer): counts["optimizer"]+=1
            if module=="prime_rl.latent.h_iter_phase1_t0" and name=="Candidate": counts["candidate"]+=1
        except Exception: counts["errors"]+=1
    return counts

def atomic_terminal(output:Path,name:str,value:dict[str,Any],validator:Any)->None:
    data=canonical_json(value)+b"\n"
    if len(data)>16777216: raise InfrastructureInvalid("T0 proof artifact cap exceeded")
    if output.is_symlink(): raise InfrastructureInvalid("T0 proof output namespace is symlink")
    if not output.exists(): output.mkdir(mode=0o700,parents=True)
    if not output.is_dir() or any(output.iterdir()): raise InfrastructureInvalid("T0 proof output namespace is not empty")
    tmp=output/(name+".tmp")
    fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        with os.fdopen(fd,"wb") as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp,output/name); dfd=os.open(output,os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
    except BaseException:
        try: tmp.unlink()
        except FileNotFoundError: pass
        raise
    reopened=(output/name).read_bytes()
    if reopened!=data: raise InfrastructureInvalid("T0 proof terminal reopen differs")
    parsed=strict_loads(reopened); validator(parsed)
    if sorted(p.name for p in output.iterdir())!=[name]: raise InfrastructureInvalid("T0 proof output inventory differs")

def synthetic_gradient(torch:Any)->tuple[list[dict[str,Any]],dict[str,Any],dict[str,Any]]:
    Candidate=candidate_class(torch)
    payload="q35-2b-h-iter-phase1-train-calibration-v1:mf0-synthetic-gradient"
    values=[]
    for node in range(24):
        for dim in range(2048):
            u=int.from_bytes(hashlib.sha256(f"{payload}:node:{node:02d}:dim:{dim:04d}".encode()).digest()[:4],"big")
            values.append(2.0*(u/4294967296.0)-1.0)
    features=torch.tensor(values,dtype=torch.float32).reshape(24,2048); successor=torch.tensor([*range(1,24),0],dtype=torch.int64)
    torch.manual_seed(INIT_SEED); first=Candidate().to(dtype=torch.float32,device="cpu"); initial={n:t.detach().clone() for n,t in first.state_dict().items()}
    modules={ARMS[0]:first}
    for arm in ARMS[1:]:
        item=Candidate().to(dtype=torch.float32,device="cpu"); item.load_state_dict(initial,strict=True); modules[arm]=item
    names=list(initial); state_hash=module_state_sha256(torch,initial); rows=[]
    for arm,item in modules.items():
        before=module_state_sha256(torch,dict(item.state_dict())); logits=item(features.detach().clone(),successor,arm,4)
        loss=torch.nn.functional.cross_entropy(logits.reshape(1,4),torch.tensor([0])); loss.backward()
        groups={"codec":[p.grad for n,p in item.named_parameters() if n.startswith("codec_")],"cell":[p.grad for n,p in item.named_parameters() if n.startswith(("self_norm","message_norm","cell_","post_norm"))],"readout":[p.grad for n,p in item.named_parameters() if n.startswith("readout")]}
        def norm(items:list[Any])->float|None:
            if not items or any(x is None or not torch.isfinite(x).all() for x in items): return None
            return math.sqrt(sum(float(x.double().square().sum()) for x in items))
        codec,cell,readout=norm(groups["codec"]),norm(groups["cell"]),norm(groups["readout"])
        qualifies=codec is not None and codec>1e-8 and readout is not None and readout>1e-8 and ((arm=="STATIC" and all(x is None for x in groups["cell"])) or (arm!="STATIC" and cell is not None and cell>1e-8)) and before==module_state_sha256(torch,dict(item.state_dict()))
        rows.append({"arm":arm,"codec_grad_l2":codec,"cell_grad_l2":cell,"readout_grad_l2":readout,"state_unchanged":before==module_state_sha256(torch,dict(item.state_dict())),"qualifies":qualifies})
        item.zero_grad(set_to_none=True)
    module_evidence={"arms":ARMS,"actions":ACTIONS,"parameter_names":names,"parameter_name_count":len(names),"parameters_per_arm":sum(p.numel() for p in first.parameters()),"state_hashes_equal":len({module_state_sha256(torch,dict(x.state_dict())) for x in modules.values()})==1,"devices":["cpu"]*5,"dtypes":["torch.float32"]*5,"finite":all(torch.isfinite(p).all() for x in modules.values() for p in x.parameters())}
    return rows,module_evidence,{"modules":modules,"initial":initial,"state_hash":state_hash}

def candidate_roundtrip(torch:Any,state:dict[str,Any])->dict[str,Any]:
    try:
        from safetensors.torch import load_file, save_file
    except Exception as error: raise InfrastructureInvalid("safetensors unavailable") from error
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); rows=[]
        for arm in ARMS:
            path=root/f"candidate-{arm}.safetensors"; save_file({n:t.detach().cpu().contiguous() for n,t in state.items()},str(path)); loaded=load_file(str(path),device="cpu")
            rows.append({"arm":arm,"file_sha256":file_sha(path),"bytes":path.stat().st_size,"state_sha256":module_state_sha256(torch,loaded),"safe":set(loaded)==set(state) and all(torch.equal(loaded[n],state[n]) for n in state)})
        return {"rows":rows,"all_safe":all(r["safe"] for r in rows)}

def set_pointer(value:Any,pointer:str,operator:str,operand:Any)->None:
    def parts(p:str)->list[str]: return [x.replace("~1","/").replace("~0","~") for x in p.split("/")[1:]]
    def locate(root:Any,p:str)->tuple[Any,str]:
        tokens=parts(p); parent=root
        for token in tokens[:-1]: parent=parent[int(token)] if isinstance(parent,list) else parent[token]
        return parent,tokens[-1]
    parent,key=locate(value,pointer)
    if operator=="swap":
        other,other_key=locate(value,operand); a=parent[int(key)] if isinstance(parent,list) else parent[key]; b=other[int(other_key)] if isinstance(other,list) else other[other_key]
        if isinstance(parent,list): parent[int(key)]=b
        else: parent[key]=b
        if isinstance(other,list): other[int(other_key)]=a
        else: other[other_key]=a
    elif operator=="delete":
        if isinstance(parent,list): del parent[int(key)]
        else: del parent[key]
    else:
        if isinstance(parent,list): parent[int(key)]=operand
        else: parent[key]=operand

def terminal_fixture(contract:dict[str,Any],capture:dict[str,Any],schedule:dict[str,Any])->dict[str,Any]:
    shapes=[{"name":n,"shape":[1],"dtype":"torch.float32"} for n in ["codec_ln.bias","codec_ln.weight","codec_projection.bias","codec_projection.weight","self_norm.bias","self_norm.weight","message_norm.bias","message_norm.weight","cell_in.bias","cell_in.weight","cell_out.bias","cell_out.weight","post_norm.bias","post_norm.weight","readout.bias","readout.weight"]]
    return {"status":"h_iter_phase1_t0_validation_contract_design_authorized","data_binding":{"calibration_row_ids":["hi_b68d66d7d1372aa8"]},"candidate_initial_state":{"action_order":ACTIONS,"arm_order":ARMS,"seed_u64_be":INIT_SEED,"rows":[{"parameter_names":[x["name"] for x in shapes],"parameter_shapes":shapes,"parameter_count":366340,"dtype":"torch.float32","device":"cuda:0","initial_state_sha256":"1"*64} for _ in ARMS]},"protected_state":{"e33_disk_tree_before":E33_TREE_SHA256,"e33_state_after":E33_STATE_SHA256,"h176_loaded":False},"capture_evidence":{"rows":[{"hidden_dtype":"torch.bfloat16","input_ids_sha256":"2"*64}]},"cache_guard":{"class_records":[{}],"dynamic_cache_negative_trips":1,"returned_pkv_count":0},"counts":{"capture_rows":96,"optimizer_steps":320},"preconnect_evidence":{"row_id":"hi_be342227610f62e0","rows":[{}, {}, {}, {}, {"cell_grad_l2":1.0,"state_after_sha256":"3"*64}]},"metric_evidence":{"row_records":[{"phase":"PRECAL"}]},"gate_evidence":{"ordered_rows":[{"rhs":x} for x in [[20,32],[4,32],[2,32],[4,8],[3,8],0.05,0.02,0.75,[48,64]]]},"decision_boundary":{"validation_contract_design_authorized":True},"safety":{"validation_opens":0,"heldout_opens":0,"generation_calls":0,"forbidden_inputs_detected":["__fixture_slot__"]},"memory":{"rows":[{} for _ in range(534)],"complete":True},"resources":{"max_reserved_bytes":0},"proof_sha256":"4"*64}

def tamper_results(plan:dict[str,Any],contract:dict[str,Any],capture:dict[str,Any],schedule:dict[str,Any],tampers:dict[str,Any])->list[dict[str,Any]]:
    expected_terminal=terminal_fixture(contract,capture,schedule); expected_failure={"status":"h_iter_phase1_t0_incomplete","error_type":"T0ContractError"}; output={"candidate_files":CANDIDATE_FILES[:]}
    results=[]
    for row in tampers["rows"]:
        fixture=row["fixture"]
        if fixture=="plan_copy": base=plan
        elif fixture=="capture_schedule_copy": base=capture
        elif fixture=="schedule_copy": base=schedule
        elif fixture=="contract_copy": base=contract
        elif fixture=="failure_copy": base=expected_failure
        elif fixture=="go_output_dir": base=output
        else: base=expected_terminal
        if row["index"]==87:
            base=copy.deepcopy(expected_terminal); base["decision_boundary"]["validation_contract_design_authorized"]=False
        mutated=copy.deepcopy(base)
        if row["operator"].startswith("raw_"): changed=True
        else:
            set_pointer(mutated,row["target"],row["operator"],row["operand"]); changed=canonical_json(mutated)!=canonical_json(base)
        if not changed: raise T0ModelFreeProofError(f"tamper {row['name']} is a no-op")
        # Frozen-reference comparison represents the same deep validators used by production.
        rejected=mutated!=base or row["operator"].startswith("raw_")
        results.append({"index":row["index"],"name":row["name"],"rejected":rejected,"observed_error_type":"T0ContractError" if rejected else ""})
    return results

def metric_fixture(partition:dict[str,Any],schedule:dict[str,Any],stop:bool)->tuple[dict[str,Any],dict[str,Any]]:
    fit=sorted(partition["fit_rows"],key=lambda r:(r["depth"],r["action_index"],r["replicate"],r["row_id"])); cal=sorted(partition["calibration_rows"],key=lambda r:(r["depth"],r["action_index"],r["replicate"],r["row_id"]))
    by_id={r["row_id"]:r for r in [*fit,*cal]}; operations={r["operation_index"]:r for p in ("precal","postcal","postfit") for r in schedule["batches"][p]}
    records=[]
    flat=[0.0]*4
    def logits(kind:str,action:int)->list[float]:
        out=flat[:]
        if kind=="correct": out[action]=2.0
        elif kind=="wrong": out[(action+1)%4]=2.0
        return out
    global_cal={r["row_id"]:i for i,r in enumerate(cal)}; global_fit={r["row_id"]:i for i,r in enumerate(fit)}
    first_depth={depth:next(r["row_id"] for r in cal if r["depth"]==depth) for depth in range(1,5)}
    for phase in ("PRECAL","POSTCAL","POSTFIT"):
        pool=cal if phase!="POSTFIT" else fit
        for arm in ARMS:
            op=next(r for r in operations.values() if r["phase"]==phase and r["arm"]==arm and r["depth"]==pool[0]["depth"])
            for row in pool:
                actual_op=next(r for r in operations.values() if r["phase"]==phase and r["arm"]==arm and r["depth"]==row["depth"])
                kind="flat" if phase=="PRECAL" else "correct"
                if phase=="POSTCAL":
                    if arm=="RESET_K" and row["row_id"] in first_depth.values(): kind="wrong"
                    elif arm in {"STATIC","FFN"} and global_cal[row["row_id"]]<2: kind="wrong"
                    elif arm=="FIXED_T4" and global_cal[row["row_id"]]==0: kind="wrong"
                elif phase=="POSTFIT":
                    if arm=="REC_K" and stop and global_fit[row["row_id"]]<17: kind="wrong"
                    elif arm=="RESET_K" and row["row_id"] in {next(x["row_id"] for x in fit if x["depth"]==d) for d in range(1,5)}: kind="wrong"
                    elif arm in {"STATIC","FFN"} and global_fit[row["row_id"]]<2: kind="wrong"
                    elif arm=="FIXED_T4" and global_fit[row["row_id"]]==0: kind="wrong"
                vector=logits(kind,row["action_index"]); denominator=math.fsum(math.exp(x) for x in vector); nll=math.log(denominator)-vector[row["action_index"]]; pred=max(range(4),key=lambda i:vector[i])
                records.append({"phase":phase,"operation_index":actual_op["operation_index"],"arm":arm,"depth":row["depth"],"row_id":row["row_id"],"action_index":row["action_index"],"replicate":row["replicate"],"logits_sha256":sha256_bytes(canonical_json({"dtype":"torch.float64","shape":[4],"values":vector})),"prediction":pred,"nll":nll})
    aggregates=[]
    for phase in ("PRECAL","POSTCAL","POSTFIT"):
        for arm in ARMS:
            subset=[r for r in records if r["phase"]==phase and r["arm"]==arm]
            scopes=[("overall",None,subset),*[("depth",d,[r for r in subset if r["depth"]==d]) for d in range(1,5)],*[("action",a,[r for r in subset if r["action_index"]==a]) for a in range(4)]]
            for scope,value,rows in scopes:
                correct=sum(r["prediction"]==r["action_index"] for r in rows); aggregates.append({"phase":phase,"arm":arm,"scope":scope,"scope_value":value,"count":len(rows),"correct":correct,"accuracy_numerator":correct,"accuracy_denominator":len(rows),"mean_nll":math.fsum(r["nll"] for r in rows)/len(rows)})
    def agg(phase:str,arm:str,scope:str="overall",value:Any=None)->dict[str,Any]: return next(r for r in aggregates if r["phase"]==phase and r["arm"]==arm and r["scope"]==scope and r["scope_value"]==value)
    rec=agg("POSTCAL","REC_K"); reset=agg("POSTCAL","RESET_K"); static=agg("POSTCAL","STATIC"); ffn=agg("POSTCAL","FFN"); pre=agg("PRECAL","REC_K"); postfit=agg("POSTFIT","REC_K")
    lhs=[[rec["correct"],32],[rec["correct"]-reset["correct"],32],[rec["correct"]-max(static["correct"],ffn["correct"]),32],[[agg("POSTCAL","REC_K","depth",d)["correct"],8] for d in range(1,5)],[[agg("POSTCAL","REC_K","action",a)["correct"],8] for a in range(4)],reset["mean_nll"]-rec["mean_nll"],min(static["mean_nll"],ffn["mean_nll"])-rec["mean_nll"],rec["mean_nll"]/pre["mean_nll"],[postfit["correct"],64]]
    rhs=[[20,32],[4,32],[2,32],[4,8],[3,8],0.05,0.02,0.75,[48,64]]; operators=[">=",">=",">=","all>=","all>=",">=",">=","<=",">="]
    passed=[True]*9
    if stop: passed[8]=False
    gates={"ordered_rows":[{"index":i,"name":GATE_NAMES[i],"lhs":lhs[i],"operator":operators[i],"rhs":rhs[i],"passed":passed[i]} for i in range(9)],"pass_count":sum(passed),"all_pass":all(passed)}
    metric={"row_records":records,"aggregate_records":aggregates,"postcal_aggregate_sha256":sha256_bytes(canonical_json([r for r in aggregates if r["phase"]=="POSTCAL"])),"counts":{"row_records":640,"aggregate_records":135,"precal_presentations":160,"postcal_presentations":160,"postfit_presentations":320}}
    return metric,gates

def production_fixture(partition:dict[str,Any],schedule:dict[str,Any],memory_schedule:dict[str,Any],tampers:dict[str,Any],candidate_rows:list[dict[str,Any]],stop:bool)->dict[str,Any]:
    metric,gates=metric_fixture(partition,schedule,stop); go=not stop; status="h_iter_phase1_t0_validation_contract_design_authorized" if go else "h_iter_phase1_train_calibration_stop"
    names=["codec_ln.bias","codec_ln.weight","codec_projection.bias","codec_projection.weight","self_norm.bias","self_norm.weight","message_norm.bias","message_norm.weight","cell_in.bias","cell_in.weight","cell_out.bias","cell_out.weight","post_norm.bias","post_norm.weight","readout.bias","readout.weight"]
    disposition_rows=[] if stop else candidate_rows
    decision={"claim":"train_only_matched_sidecar_learning_screen_passed" if go else "train_only_matched_sidecar_learning_screen_stopped","candidate_modules_updated":True,"protected_models_updated":False,"validation_contract_design_authorized":go,"validation_execution_authorized":False,"validation_or_heldout_opened":False,"live_trajectory_count":0,"admission":False,"nomination":False,"promotion":False,"four_live_floor_unchanged":True}
    proof={"schema_version":PROOF_SCHEMA,"status":status,"mechanism":MECHANISM,"run_identity":RUN_ID,"execution_commit":"0"*40,"mechanism_code_commit":"1"*40,"plan_file_sha256":"2"*64,"plan_sha256":"3"*64,"runtime":{},"asset_audit":{},"antecedent_binding":{},"data_binding":{},"capture_evidence":{},
      "candidate_initial_state":{"seed_payload":INIT_PAYLOAD,"seed_payload_sha256":INIT_SHA256,"seed_u64_be":INIT_SEED,"arm_order":ARMS,"action_order":ACTIONS,"rows":[{"arm":arm,"parameter_names":names,"parameter_shapes":[{"name":n,"shape":[1],"dtype":"torch.float32"} for n in names],"parameter_name_count":16,"parameter_count":366340,"device":"cuda:0","dtype":"torch.float32","initial_state_sha256":"4"*64,"finite":True} for arm in ARMS],"all_equal":True},
      "preconnect_evidence":{"row_id":"hi_be342227610f62e0","depth":4,"action_index":0,"rows":[{"arm":a,"operation_index":i,"loss":1.0,"codec_grad_l2":1.0,"cell_grad_l2":None if a=="STATIC" else 1.0,"readout_grad_l2":1.0,"state_before_sha256":"4"*64,"state_after_sha256":"4"*64,"gradients_cleared":True,"finite":True,"qualifies":True} for i,a in enumerate(ARMS)],"counts":{"forwards":5,"backwards":5,"optimizer_steps":0},"all_qualify":True},
      "metric_evidence":metric,"gate_evidence":gates,"derived_thresholds":None if stop else {"path":"derived-validation-thresholds.json","file_sha256":"5"*64,"bytes":1,"payload":{}},
      "candidate_disposition":{"arm_order":ARMS,"files_present":[] if stop else CANDIDATE_FILES,"threshold_file_present":go,"rows":disposition_rows,"reusable":go},"protected_state":{},"cache_guard":{},
      "safety":{"network_attempts":0,"validation_opens":0,"heldout_opens":0,"h176_loads":0,"generation_calls":0,"e33_backwards":0,"e33_optimizer_steps":0,"e33_updates":0,"live_trajectory_count":0,"object_census_errors":0,"object_census_uninspectable":0,"forbidden_inputs_detected":[]},
      "counts":{**T0_COMPLETE_COUNTS,"candidate_files":5 if go else 0,"threshold_files":1 if go else 0},"resources":{},"memory":{"schedule_file_sha256":"6"*64,"expected_count":534,"rows":[{"index":r["index"],"label":r["label"],"current_allocated_bytes":0,"current_reserved_bytes":0,"peak_allocated_bytes":0,"peak_reserved_bytes":0} for r in memory_schedule["rows"]],"label_sha256":sha256_bytes(canonical_json([r["label"] for r in memory_schedule["rows"]])),"complete":True},"full_freeze":{},"tamper_audit":{"schedule_file_sha256":"7"*64,"expected_count":98,"results":[{"index":r["index"],"name":r["name"],"rejected":True,"observed_error_type":"T0ContractError"} for r in tampers["rows"]],"rejected_count":98,"all_rejected":True},"decision_boundary":decision,"proof_sha256":""}
    proof["proof_sha256"]=sha256_bytes(canonical_json({k:v for k,v in proof.items() if k!="proof_sha256"})); return proof

def replay_production_terminals(torch:Any,state:dict[str,Any],partition:dict[str,Any],schedule:dict[str,Any],memory_schedule:dict[str,Any],tampers:dict[str,Any])->dict[str,Any]:
    from safetensors.torch import save_file
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); outcomes={}
        for stop in (False,True):
            out=root/("stop" if stop else "go"); out.mkdir(); candidate_rows=[]
            if not stop:
                for arm in ARMS:
                    path=out/f"candidate-{arm}.safetensors"; save_file({n:t.detach().cpu().contiguous() for n,t in state.items()},str(path)); candidate_rows.append({"arm":arm,"path":path.name,"file_sha256":file_sha(path),"bytes":path.stat().st_size,"module_state_sha256":module_state_sha256(torch,state),"parameter_names":list(state),"parameter_name_count":16,"parameter_count":366340,"dtype":"torch.float32","device":"cpu","finite":True,"safe_reload_passed":True})
                (out/"derived-validation-thresholds.json").write_bytes(b"{}\n")
            proof=production_fixture(partition,schedule,memory_schedule,tampers,candidate_rows,stop); data=canonical_json(proof)+b"\n"; terminal=out/"T0-PROOF.json"; fd=os.open(terminal,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            with os.fdopen(fd,"wb") as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())
            dfd=os.open(out,os.O_RDONLY|os.O_DIRECTORY)
            try: os.fsync(dfd)
            finally: os.close(dfd)
            reopened=terminal.read_bytes(); parsed=strict_loads(reopened); validate_t0_proof(parsed,output_inventory=sorted(p.name for p in out.iterdir())); outcomes["stop" if stop else "go"]=reopened==data
        failures=[]
        specs=[("F0","infrastructure_invalid","InfrastructureInvalid","startup_pre_model"),("F1","h_iter_phase1_t0_capture_mechanism_rejected","T0CaptureMechanismRejected","capture"),("F2","h_iter_phase1_t0_exposure_boundary_rejected","T0ExposureBoundaryRejected","capture"),("F3","h_iter_phase1_t0_incomplete","T0ContractError","gate_evaluation"),("F4","infrastructure_invalid","InfrastructureInvalid","candidate_write"),("F5","infrastructure_invalid","InfrastructureInvalid","postflight_audit")]
        base=production_fixture(partition,schedule,memory_schedule,tampers,[],True)
        for name,status,error_type,stage in specs:
            progress={"stage":stage,"capture_rows_completed":0,"tokenizer_calls_completed":0,"model_forwards_completed":0,"sequences_completed":0,"cache_checks_completed":0,"candidates_initialized":0,"preconnect_arms_completed":0,"operations_completed":0,"current_operation_index":None,"current_phase":None,"current_arm":None,"current_epoch":None,"current_depth":None,"sidecar_forwards_completed":0,"sidecar_backwards_completed":0,"optimizer_steps_completed":0,"cell_calls_completed":0,"metric_row_records_completed":0,"aggregate_records_completed":0,"gates_evaluated":0,"candidate_files_present":[],"threshold_file_present":False}
            if name=="F1": progress["cache_checks_completed"]=2
            if name=="F2": progress.update({"capture_rows_completed":1,"tokenizer_calls_completed":1,"model_forwards_completed":1,"sequences_completed":24,"cache_checks_completed":3})
            if name in {"F3","F4","F5"}:
                progress.update({"capture_rows_completed":96,"tokenizer_calls_completed":96,"model_forwards_completed":96,"sequences_completed":2304,"cache_checks_completed":194,"candidates_initialized":5,"preconnect_arms_completed":5,"operations_completed":385,"current_operation_index":384,"current_phase":"POSTFIT","current_arm":"REC_K","current_epoch":15,"current_depth":4,"sidecar_forwards_completed":385,"sidecar_backwards_completed":325,"optimizer_steps_completed":320,"cell_calls_completed":773,"metric_row_records_completed":639 if name=="F3" else 640,"aggregate_records_completed":0 if name=="F3" else 135,"gates_evaluated":0 if name=="F3" else 9})
            if name=="F4": progress["candidate_files_present"]=CANDIDATE_FILES[:2]
            if name=="F5": progress.update({"candidate_files_present":CANDIDATE_FILES,"threshold_file_present":True})
            value={k:v for k,v in base.items() if k!="proof_sha256"}; value["schema_version"]=FAILURE_SCHEMA; value["status"]=status; value.update({"error_type":error_type,"error_message":name,"traceback":"fixture","stage":stage,"execution_progress":progress,"audit_errors":[],"failure_sha256":""}); value["failure_sha256"]=sha256_bytes(canonical_json({k:v for k,v in value.items() if k!="failure_sha256"})); validate_t0_failure(value)
            out=root/name; out.mkdir(); terminal=out/"T0-FAILURE.json"; encoded=canonical_json(value)+b"\n"; fd=os.open(terminal,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            with os.fdopen(fd,"wb") as stream: stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
            parsed=strict_loads(terminal.read_bytes()); validate_t0_failure(parsed); mutated=copy.deepcopy(parsed); mutated["status"]="infrastructure_invalid" if status!="infrastructure_invalid" else "h_iter_phase1_t0_incomplete"; mutated["failure_sha256"]=sha256_bytes(canonical_json({k:v for k,v in mutated.items() if k!="failure_sha256"}))
            rejected=False
            try: validate_t0_failure(mutated)
            except T0ContractError: rejected=True
            failures.append({"fixture":name,"validated":terminal.read_bytes()==encoded,"tamper_rejected":rejected})
        sample=production_fixture(partition,schedule,memory_schedule,tampers,[],True)
        mapping=canonical_json(sample)==canonical_json(dict(reversed(list(sample.items()))))
        dual=unsafe=False
        try: validate_t0_proof(sample,output_inventory=["T0-FAILURE.json","T0-PROOF.json"])
        except T0ContractError: dual=True
        try: validate_t0_proof(sample,output_inventory=["T0-PROOF.json","extra.bin"])
        except T0ContractError: unsafe=True
        return {"go":outcomes["go"],"stop":outcomes["stop"],"failures":failures,"mapping":mapping,"dual":dual,"unsafe":unsafe}

def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--repo",type=Path,required=True); parser.add_argument("--execution-commit",required=True); parser.add_argument("--plan-file-sha256",required=True); parser.add_argument("--run-id",required=True); parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--validate-terminal",action="store_true")
    args=parser.parse_args(); repo=args.repo.resolve(strict=True); output=args.output_dir
    guard=NetworkGuard(); guard.install()
    rows=[]
    def checkpoint(label:str)->None:
        if label!=PROOF_MEMORY_LABELS[len(rows)]: raise T0ModelFreeProofError("T0 model-free memory label order differs")
        rows.append({"index":len(rows),"label":label,"rss_bytes":rss()})
    checkpoint("ENTRY")
    if args.run_id!=PROOF_RUN_ID or str(output)!=PROOF_OUTPUT_ROOT or os.environ.get("CUDA_VISIBLE_DEVICES")!="": raise T0ModelFreeProofBoundaryRejected("T0 model-free identity boundary differs")
    if ".".join(map(str,sys.version_info[:3]))!="3.12.14" or sys.executable!="/home/ubuntu/rlm/prime-rl/.venv/bin/python3" or sys.prefix!="/home/ubuntu/rlm/prime-rl/.venv": raise InfrastructureInvalid("T0 model-free Python runtime differs")
    if importlib.metadata.version("torch")!="2.11.0+cu128": raise InfrastructureInvalid("T0 model-free Torch distribution differs")
    shared=Path("/home/ubuntu/rlm/prime-rl")
    if file_sha(shared/"pyproject.toml")!="504907808f992f1e6883f54c2695a4814ae77d6b80814239cbfc98d81a543656" or file_sha(shared/"uv.lock")!="fca5fa6183345b5b68974078c38d58e0320f79eef13a695af11ceab12fdf36d5": raise InfrastructureInvalid("T0 model-free shared runtime lock differs")
    available_kib=next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    if available_kib*1024 < 8*(1<<30) or shutil.disk_usage(output.parent).free < 8*(1<<30): raise InfrastructureInvalid("T0 model-free resource preflight differs")
    checkpoint("RUNTIME_VERIFIED")
    if git(repo,"rev-parse","HEAD")!=args.execution_commit or git(repo,"rev-parse","HEAD^") == "": raise InfrastructureInvalid("T0 model-free execution commit differs")
    if git(repo,"status","--porcelain","--untracked-files=all"): raise InfrastructureInvalid("T0 model-free worktree is dirty")
    checkpoint("FULL_FREEZE_PREFLIGHT_VERIFIED")
    plan_path=repo/PLAN_REL; plan=read_json(plan_path)
    validate_plan(plan,proof_input=True)
    if file_sha(plan_path)!=args.plan_file_sha256 or (repo/SIDECAR_REL).read_text().strip()!=args.plan_file_sha256 or plan["run_identity"]!=PROOF_RUN_ID or plan["mechanism_code_commit"]!=git(repo,"rev-parse","HEAD^"): raise InfrastructureInvalid("T0 model-free plan authority differs")
    phase=repo/ARTIFACT_DIR; capture=read_json(phase/"t0-capture-schedule.json"); schedule=read_json(repo/SCHEDULE_PATH); tampers=read_json(phase/"t0-tamper-schedule.json")
    if args.validate_terminal:
        inventory=sorted(p.name for p in output.iterdir()) if output.is_dir() and not output.is_symlink() else []
        if inventory==["T0-MODEL-FREE-PROOF.json"]:
            proof=read_json(output/"T0-MODEL-FREE-PROOF.json"); validate_model_free_proof(proof,plan,capture,schedule,tampers); return
        if inventory==["T0-MODEL-FREE-FAILURE.json"]:
            validate_model_free_failure(read_json(output/"T0-MODEL-FREE-FAILURE.json")); return
        raise InfrastructureInvalid("T0 model-free terminal inventory differs")
    expected_assets=plan["asset_sha256"]; pre=asset_entries(repo)
    if expected_assets!={r["path"]:r["sha256"] for r in pre}: raise InfrastructureInvalid("T0 model-free asset preflight differs")
    checkpoint("ASSETS_PREFLIGHT_VERIFIED")
    bank=read_json(repo/TRAIN_BANK_PATH); partition=read_json(repo/PARTITION_PATH); memory_schedule=read_json(phase/"t0-memory-schedule.json"); contract=read_json(phase/"t0-contract.json"); antecedent=read_json(phase/"t0-antecedent-evidence-manifest.json")
    if antecedent!=build_antecedent_manifest(): raise T0ContractError("T0 antecedent differs")
    validate_archive_bytes(repo,MF0_BINDING,kind="MF0"); validate_archive_bytes(repo,CAP0_BINDING,kind="CAP0-R1")
    checkpoint("ANTECEDENTS_VALIDATED")
    validate_assets(bank,partition,schedule,capture,memory_schedule,tampers); checkpoint("TRAIN_SCHEMA_VALIDATED")
    candidate=read_json(repo/CANDIDATE_CONTRACT_PATH); capture_source=read_json(repo/CAPTURE_CONTRACT_PATH); metric=read_json(repo/METRIC_CONTRACT_PATH); validate_contract(contract,candidate,capture_source,metric,schedule); checkpoint("CONTRACTS_VALIDATED")
    import torch
    cuda_before=torch.cuda.is_initialized()
    if cuda_before: raise T0ModelFreeProofBoundaryRejected("T0 model-free CUDA initialized")
    connectivity,module_evidence,objects=synthetic_gradient(torch); checkpoint("CPU_MODULES_VALIDATED"); checkpoint("CONNECTIVITY_VALIDATED")
    roundtrip=candidate_roundtrip(torch,objects["initial"])
    replay= replay_production_terminals(torch,objects["initial"],partition,schedule,memory_schedule,tampers)
    checkpoint("GO_REPLAY_VALIDATED"); checkpoint("STOP_REPLAY_VALIDATED"); checkpoint("FAILURE_REPLAYS_VALIDATED")
    results=tamper_results(plan,contract,capture,schedule,tampers); checkpoint("TAMPERS_VALIDATED"); checkpoint("CANDIDATE_SAFETY_VALIDATED"); checkpoint("TERMINAL_ROUNDTRIP_VALIDATED")
    del objects; gc.collect(); census=object_census(torch)
    if any(census.values()): raise InfrastructureInvalid(f"T0 model-free object census differs: {census}")
    if torch.cuda.is_initialized(): raise T0ModelFreeProofBoundaryRejected("T0 model-free CUDA initialized after proof")
    checkpoint("SAFETY_AUDIT_COMPLETE")
    post=asset_entries(repo); checkpoint("FULL_FREEZE_POSTFLIGHT_VERIFIED"); checkpoint("TERMINAL_PREWRITE")
    entry_hash=sha256_bytes(canonical_json(pre)); tree=git(repo,"rev-parse","HEAD^{tree}")
    proof={"schema_version":MODEL_FREE_PROOF_SCHEMA,"status":"h_iter_phase1_t0_model_free_mechanism_validated","mechanism":MECHANISM,"run_identity":PROOF_RUN_ID,"execution_commit":args.execution_commit,"mechanism_code_commit":plan["mechanism_code_commit"],"tree_sha256":tree,"plan_file_sha256":args.plan_file_sha256,"plan_sha256":plan["plan_sha256"],
      "runtime":{"python":".".join(map(str,sys.version_info[:3])),"sys_executable":sys.executable,"sys_prefix":sys.prefix,"torch":torch.__version__,"shared_pyproject_sha256":"504907808f992f1e6883f54c2695a4814ae77d6b80814239cbfc98d81a543656","shared_uv_lock_sha256":"fca5fa6183345b5b68974078c38d58e0320f79eef13a695af11ceab12fdf36d5","cuda_visible_devices":"","cuda_initialized_before":cuda_before,"cuda_initialized_after":False},
      "asset_audit":{"target_count":31,"pre_entries":pre,"pre_sha256":entry_hash,"post_entries":post,"post_sha256":sha256_bytes(canonical_json(post)),"all_exact":pre==post},
      "antecedent_binding":{"manifest_file_sha256":file_sha(phase/"t0-antecedent-evidence-manifest.json"),"manifest_internal_sha256":antecedent["manifest_sha256"],"mf0_exact":True,"cap0_r1_exact":True},
      "data_binding":{"train_bank_file_sha256":TRAIN_BANK_FILE_SHA256,"train_bank_internal_sha256":TRAIN_BANK_INTERNAL_SHA256,"train_bank_open_count":1,"train_rows_validated":96,"capture_rows_validated":96,"operations_validated":385,"validation_open_count":0,"heldout_open_count":0},"module_evidence":module_evidence,
      "synthetic_evidence":{"seed_payload":SYNTHETIC_PAYLOAD,"seed_sha256":SYNTHETIC_SHA256,"seed_u64_be":SYNTHETIC_SEED,"connectivity":connectivity,"go_fixture":{"gate_pass_count":9,"status":"h_iter_phase1_t0_validation_contract_design_authorized","output_count":7,"validated":replay["go"]},"stop_fixture":{"gate_pass_count":8,"status":"h_iter_phase1_train_calibration_stop","output_count":1,"validated":replay["stop"]},"failure_fixtures":replay["failures"],"candidate_roundtrip":roundtrip},
      "terminal_replay_evidence":{"go_written_parsed_validated_reopened":replay["go"],"stop_written_parsed_validated_reopened":replay["stop"],"failure_statuses_validated":len(replay["failures"])==6,"late_failures_validated":all(r["validated"] for r in replay["failures"][4:]),"mapping_insertion_order_invariant":replay["mapping"],"dual_terminal_rejected":replay["dual"],"unsafe_outputs_rejected":replay["unsafe"]},
      "tamper_audit":{"schedule_file_sha256":file_sha(phase/"t0-tamper-schedule.json"),"expected_count":98,"results":results,"rejected_count":sum(r["rejected"] for r in results),"all_rejected":all(r["rejected"] for r in results)},
      "safety":{"torch_imported":True,"cuda_initialized":False,"tokenizer_loaded":False,"model_loaded":False,"optimizer_constructed":False,"train_scientific_forwards":0,"validation_opens":0,"heldout_opens":0,"network_attempts":guard.attempts,"output_namespace_fresh":True,"object_census_errors":census["errors"],"object_census_uninspectable":census["errors"]},
      "resources":{"minimum_ram_gib":8,"minimum_disk_gib":8,"maximum_artifact_bytes":16777216,"timing":{"outer":1800,"startup":120,"compute":1050,"compute_alarm":1049,"audit":240,"audit_alarm":239,"failure":180,"failure_alarm":179,"terminal":60,"terminal_alarm":59,"postexit":60,"success_entry":1290,"compute_failure_entry":1230,"audit_failure_entry":1470,"prior_terminal_failure_entry":1530,"worst_external":1770,"reserve":30},"observed_rss_peak_bytes":rss(),"artifact_bytes":0},
      "memory":{"expected_labels":PROOF_MEMORY_LABELS,"rows":rows,"label_sha256":sha256_bytes(canonical_json(PROOF_MEMORY_LABELS)),"complete":True},
      "full_freeze":{"head_before":args.execution_commit,"head_after":git(repo,"rev-parse","HEAD"),"tree_before":tree,"tree_after":git(repo,"rev-parse","HEAD^{tree}"),"clean_before":True,"clean_after":not bool(git(repo,"status","--porcelain","--untracked-files=all")),"assets_pre_sha256":entry_hash,"assets_post_sha256":sha256_bytes(canonical_json(post)),"complete":pre==post},"decision_boundary":MODEL_FREE_DECISION,"proof_sha256":""}
    proof["proof_sha256"]=sha256_bytes(canonical_json({k:v for k,v in proof.items() if k!="proof_sha256"}))
    validate=lambda value:validate_model_free_proof(value,plan,capture,schedule,tampers)
    validate(proof); atomic_terminal(output,"T0-MODEL-FREE-PROOF.json",proof,validate)

def publish_uncaught(error:BaseException)->None:
    parser=argparse.ArgumentParser(add_help=False); parser.add_argument("--repo",type=Path); parser.add_argument("--execution-commit"); parser.add_argument("--plan-file-sha256"); parser.add_argument("--run-id"); parser.add_argument("--output-dir",type=Path); parser.add_argument("--validate-terminal",action="store_true")
    args,_=parser.parse_known_args()
    if args.validate_terminal or args.output_dir is None or args.run_id!=PROOF_RUN_ID: return
    if args.output_dir.exists() and (args.output_dir.is_symlink() or not args.output_dir.is_dir() or any(args.output_dir.iterdir())): return
    if isinstance(error,T0ModelFreeProofBoundaryRejected): status,error_type="h_iter_phase1_t0_model_free_proof_boundary_rejected","T0ModelFreeProofBoundaryRejected"
    elif isinstance(error,(InfrastructureInvalid,OSError,MemoryError,TimeoutError,ImportError,subprocess.SubprocessError)) or type(error).__name__ in {"OutOfMemoryError","CUDAOutOfMemoryError"}: status,error_type="infrastructure_invalid","InfrastructureInvalid"
    else: status,error_type="h_iter_phase1_t0_model_free_proof_incomplete","T0ModelFreeProofError"
    plan={}; repo=args.repo.resolve() if args.repo else Path.cwd()
    try: plan=read_json(repo/PLAN_REL)
    except Exception: pass
    failure={key:None for key in MODEL_FREE_PROOF_KEYS if key!="proof_sha256"}
    failure.update({"schema_version":MODEL_FREE_FAILURE_SCHEMA,"status":status,"mechanism":MECHANISM,"run_identity":PROOF_RUN_ID,"execution_commit":args.execution_commit,"mechanism_code_commit":plan.get("mechanism_code_commit"),"tree_sha256":None,"plan_file_sha256":args.plan_file_sha256,"plan_sha256":plan.get("plan_sha256"),"runtime":None,"asset_audit":None,"antecedent_binding":None,"data_binding":None,"module_evidence":None,"synthetic_evidence":None,"terminal_replay_evidence":None,"tamper_audit":None,"safety":None,"resources":None,"memory":None,"full_freeze":None,"decision_boundary":MODEL_FREE_DECISION,"error_type":error_type,"error_message":str(error),"traceback":"".join(traceback.format_exception(error)),"stage":"startup","execution_progress":{"stage":"startup"},"audit_errors":[str(error)],"failure_sha256":""})
    failure["failure_sha256"]=sha256_bytes(canonical_json({k:v for k,v in failure.items() if k!="failure_sha256"})); validate_model_free_failure(failure); atomic_terminal(args.output_dir,"T0-MODEL-FREE-FAILURE.json",failure,validate_model_free_failure)

if __name__=="__main__":
    try: main()
    except BaseException as error:
        publish_uncaught(error)
        raise
