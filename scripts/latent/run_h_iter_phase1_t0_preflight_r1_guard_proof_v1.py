#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib.metadata
import json
import os
import resource
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from freeze_h_iter_phase1_t0_plan_v1 import (
    FAILED_ASSETS, FAILED_MANIFEST, FAILED_START_BINDING, GUARD_CASE_NAMES,
    GUARD_DECISION_DESIGN, GUARD_MECHANISM, GUARD_PROOF_ASSETS, GUARD_RUN_ID,
    validate_guard_plan,
)

PLAN_REL="experiments/qwen35-2b-latent-workspace-v1/h-iter-phase1-t0-train-calibration-v1/t0-preflight-r1-guard-proof-plan.json"
SIDECAR_REL="experiments/qwen35-2b-latent-workspace-v1/h-iter-phase1-t0-train-calibration-v1/t0-preflight-r1-guard-proof-plan.sha256"
RUNNER_REL="scripts/latent/run_h_iter_phase1_t0_v1.py"
PROOF_SCHEMA="prime-rl/latent-h-iter-phase1-t0-preflight-r1-guard-proof/v1"
FAILURE_SCHEMA="prime-rl/latent-h-iter-phase1-t0-preflight-r1-guard-failure/v1"
PROOF_STATUS="h_iter_phase1_t0_preflight_r1_guard_mechanism_validated"
PROOF_NAME="T0-PREFLIGHT-R1-GUARD-PROOF.json"
FAILURE_NAME="T0-PREFLIGHT-R1-GUARD-FAILURE.json"
MEMORY_LABELS=["ENTRY","RUNTIME_VERIFIED","FULL_FREEZE_PREFLIGHT_VERIFIED","ASSETS_PREFLIGHT_VERIFIED","FAILED_START_BOUND","BASELINE_SOURCE_VERIFIED","REPAIRED_SOURCE_VERIFIED","ACTUAL_SOURCE_PASS","GENERATE_REJECTED","SAVE_PRETRAINED_REJECTED","VALIDATION_BANK_REJECTED","HELDOUT_BANK_REJECTED","SAFETY_AUDIT_COMPLETE","TERMINAL_PREWRITE"]
PROOF_KEYS={"schema_version","status","mechanism","run_identity","execution_commit","mechanism_code_commit","tree_sha256","plan_file_sha256","plan_sha256","runtime","asset_audit","failed_start_binding","source_evidence","case_results","safety","resources","memory","full_freeze","decision_boundary","proof_sha256"}
PROOF_DECISION={"claim":"preflight_guard_robustness_mechanism_only","independent_scientific_replicate":False,"t0_preflight_authorized":False,"t0_full_authorized":False,"model_or_gpu_authorized":False,"scientific_exposure":False,"validation_or_heldout_opened":False,"training":False,"admission":False,"nomination":False,"promotion":False,"four_live_floor_unchanged":True}
STATE:dict[str,Any]={"stage":"startup","memory":[],"network_attempts":0}

class T0PreflightR1GuardProofError(RuntimeError): pass
class T0PreflightR1GuardProofBoundaryRejected(RuntimeError): pass
class InfrastructureInvalid(RuntimeError): pass

def canonical_json(value:Any)->bytes: return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def sha(value:bytes)->str: return hashlib.sha256(value).hexdigest()
def file_sha(path:Path)->str: return sha(path.read_bytes())
def strict_loads(data:bytes)->Any:
    value=json.loads(data.decode("utf-8"),object_pairs_hook=lambda pairs:_pairs(pairs),parse_constant=lambda value:(_ for _ in ()).throw(ValueError(value)))
    if canonical_json(value)+b"\n"!=data: raise RuntimeError("T0 guard JSON is not canonical")
    return value
def _pairs(pairs:list[tuple[str,Any]])->dict[str,Any]:
    value={}
    for key,item in pairs:
        if key in value: raise ValueError("duplicate JSON key")
        value[key]=item
    return value
def git(repo:Path,*args:str)->str: return subprocess.check_output(["git",*args],cwd=repo,text=True).strip()
def rss()->int: return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
def mark(label:str)->None:
    if label!=MEMORY_LABELS[len(STATE["memory"])]: raise RuntimeError("T0 guard proof memory order differs")
    STATE["memory"].append({"index":len(STATE["memory"]),"label":label,"rss_bytes":rss()})
def timeout_alarm(_signum:int,_frame:Any)->None: raise TimeoutError("T0 guard proof timeout")

class NetworkGuard:
    def __init__(self)->None: self.originals=[]
    def deny(self,*_args:Any,**_kwargs:Any)->Any:
        STATE["network_attempts"]+=1
        raise T0PreflightR1GuardProofBoundaryRejected("T0 guard proof network denied")
    def install(self)->None:
        for owner,name in ((socket.socket,"connect"),(socket.socket,"connect_ex"),(socket,"create_connection"),(socket,"getaddrinfo")):
            self.originals.append((owner,name,getattr(owner,name))); setattr(owner,name,self.deny)

def asset_entries(repo:Path)->list[dict[str,Any]]:
    rows=[]
    for relative in sorted(GUARD_PROOF_ASSETS):
        path=repo/relative
        if not path.is_file() or path.is_symlink(): raise RuntimeError(f"T0 guard unsafe asset {relative}")
        rows.append({"path":relative,"sha256":file_sha(path),"bytes":path.stat().st_size})
    return rows

def verify_runtime(repo:Path,plan:dict[str,Any],output:Path)->None:
    runtime=plan["runtime"]
    observed={"python":".".join(map(str,sys.version_info[:3])),"sys_executable":str(Path(sys.executable).resolve()),"sys_prefix":str(Path(sys.prefix).resolve()),"torch":importlib.metadata.version("torch"),"transformers":importlib.metadata.version("transformers"),"tokenizers":importlib.metadata.version("tokenizers"),"flash_linear_attention":importlib.metadata.version("flash-linear-attention"),"shared_project_pyproject_sha256":file_sha(Path(plan["remote_paths"]["shared_project"])/"pyproject.toml"),"shared_project_uv_lock_sha256":file_sha(Path(plan["remote_paths"]["shared_project"])/"uv.lock"),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES")}
    if observed!=runtime or file_sha(repo/"pyproject.toml")!=runtime["shared_project_pyproject_sha256"] or file_sha(repo/"uv.lock")!=runtime["shared_project_uv_lock_sha256"]: raise InfrastructureInvalid("T0 guard proof runtime differs")
    available=int(os.sysconf("SC_AVPHYS_PAGES"))*int(os.sysconf("SC_PAGE_SIZE")); free_disk=os.statvfs(output.parent).f_bavail*os.statvfs(output.parent).f_frsize
    if available<8*(1<<30) or free_disk<8*(1<<30): raise InfrastructureInvalid("T0 guard proof resources unavailable")

class NormalizeGuard(ast.NodeTransformer):
    def visit_FunctionDef(self,node:ast.FunctionDef)->Any:
        if node.name=="validate_static_exposure_source": return None
        if node.name=="preflight":
            body=[]
            for statement in node.body:
                legacy_tree=isinstance(statement,ast.Assign) and len(statement.targets)==1 and isinstance(statement.targets[0],ast.Name) and statement.targets[0].id=="tree" and isinstance(statement.value,ast.Call) and isinstance(statement.value.func,ast.Attribute) and statement.value.func.attr=="parse"
                if legacy_tree: continue
                helper=isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Call) and isinstance(statement.value.func,ast.Name) and statement.value.func.id=="validate_static_exposure_source"
                legacy=isinstance(statement,ast.If) and any(isinstance(part,ast.Constant) and part.value=="T0 static exposure guard differs" for part in ast.walk(statement))
                body.append(ast.Expr(value=ast.Constant(value="STATIC_EXPOSURE_GUARD")) if helper or legacy else statement)
            node.body=body
        return self.generic_visit(node)

def normalized_source(source:str)->tuple[str,list[str]]:
    tree=NormalizeGuard().visit(ast.parse(source)); ast.fix_missing_locations(tree)
    dumps=[ast.dump(node,annotate_fields=True,include_attributes=False) for node in tree.body]
    return sha(canonical_json(dumps)),dumps

def load_guard(source:str):
    module=ast.parse(source); helper=next((node for node in module.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name=="validate_static_exposure_source"),None)
    if helper is None: raise RuntimeError("T0 guard helper missing")
    namespace={"ast":ast,"RuntimeError":RuntimeError}; exec(compile(ast.Module(body=[helper],type_ignores=[]),"<guard>","exec"),namespace)
    return namespace["validate_static_exposure_source"],sha(ast.dump(helper,annotate_fields=True,include_attributes=False).encode())

def case_results(source:str)->tuple[list[dict[str,Any]],str]:
    guard,guard_hash=load_guard(source)
    validation_name="-".join(("validation","bank.json")); heldout_name="-".join(("heldout","bank.json"))
    mutations=[("production_source_pass","production",source,"pass"),("generate_call_rejected","append_attribute_call",source+"\nobject().generate()\n","reject"),("save_pretrained_call_rejected","append_attribute_call",source+"\nobject().save_pretrained('x')\n","reject"),("validation_bank_literal_rejected","append_string_constant",source+f"\nFORBIDDEN_FIXTURE={validation_name!r}\n","reject"),("heldout_bank_literal_rejected","append_string_constant",source+f"\nFORBIDDEN_FIXTURE={'/tmp/'+heldout_name!r}\n","reject")]
    rows=[]
    for index,(name,kind,mutated,expected) in enumerate(mutations):
        error=None
        try: guard(mutated); observed="pass"
        except RuntimeError as caught: observed="reject"; error=type(caught).__name__
        rows.append({"index":index,"name":name,"mutation_kind":kind,"mutated_source_sha256":sha(mutated.encode()),"expected_outcome":expected,"observed_outcome":observed,"observed_error_type":error,"qualifies":observed==expected and error==(None if expected=="pass" else "RuntimeError")})
    return rows,guard_hash

def validate_failed_start(repo:Path)->dict[str,Any]:
    expected={
      f"{Path(FAILED_MANIFEST).parent}/t0-preflight-run1-failed.plan.json":("3a872893adc44e770fcf8dd7784ac348107cc36b6b52f560dfe8ea467920ff85",17493),
      f"{Path(FAILED_MANIFEST).parent}/t0-preflight-run1-failed.plan.sha256":("024021a84f3f0caa15f31ea22668ab180eedcd86ed18c53b68c0c27f9cc0c87f",65),
      f"{Path(FAILED_MANIFEST).parent}/t0-preflight-run1-failed.stdout":("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",0),
      f"{Path(FAILED_MANIFEST).parent}/t0-preflight-run1-failed.stderr":("a73f3021da405f37db6bc0c4c2aed0d0f365515fbd20364f10360d1f2104ee08",1244),
      f"{Path(FAILED_MANIFEST).parent}/t0-preflight-run1-failed.exit.txt":("4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865",2),
      FAILED_MANIFEST:("4a8cf5d418a67c05e4b999beed417dd087846fbd01fbfee5e8410410b60a92b5",3005),
    }
    for relative,(digest,size) in expected.items():
        path=repo/relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size!=size or file_sha(path)!=digest: raise RuntimeError("T0 guard failed-start evidence differs")
    manifest=strict_loads((repo/FAILED_MANIFEST).read_bytes())
    if manifest["manifest_sha256"]!="311d19834014c44f807777e9488a2cd9842a2cdc7b99c5456b7191c7ed6065e1" or sha(canonical_json({k:v for k,v in manifest.items() if k!="manifest_sha256"}))!=manifest["manifest_sha256"]: raise RuntimeError("T0 guard failed-start manifest differs")
    top={"schema_version","status","mechanism","attempt_identity","run_identity","execution_commit","mechanism_code_commit","tree_sha256","plan_file_sha256","plan_sha256","ordered_evidence_assets","failure_evidence","claim_boundary","manifest_sha256"}
    claim={"classification":"infrastructure_invalid","validated_preflight":False,"output_namespace_created":False,"torch_imported":False,"transformers_imported":False,"tokenizer_loaded":False,"model_loaded":False,"cuda_initialized":False,"gpu_used":False,"scientific_exposure":False,"validation_or_heldout_opened":False,"training_or_update":False,"t0_full_authorized":False,"retry_requires_new_freeze":True}
    identity=(manifest.get("schema_version"),manifest.get("status"),manifest.get("mechanism"),manifest.get("attempt_identity"),manifest.get("run_identity"),manifest.get("execution_commit"),manifest.get("mechanism_code_commit"),manifest.get("tree_sha256"),manifest.get("plan_file_sha256"),manifest.get("plan_sha256"))
    expected_identity=("prime-rl/latent-h-iter-phase1-t0-preflight-failed-start-evidence-manifest/v1","h_iter_phase1_t0_preflight_failed_start_archived","q35-2b-h-iter-phase1-t0-train-calibration-v1","h-iter-phase1-t0-preflight-run1-invalid","h-iter-phase1-t0-train-calibration-run1","67b21d2ccd7cc3189154f67a80fe8db6abd3ec7b","6d43c0175256f00c61bb08e26e61ff6112489911","7670bd01e51c16f27d5eaea6fbd37826d862eecd","3a872893adc44e770fcf8dd7784ac348107cc36b6b52f560dfe8ea467920ff85","8be654db553fc986cec36b6b6ca73516d18a6104fe9f86eb3e626a242352cf6a")
    if set(manifest)!=top or identity!=expected_identity or manifest["failure_evidence"]!={"stage":"preflight_static_exposure_guard","error_type":"RuntimeError","error_message":"T0 static exposure guard differs","launcher_exit_code":1,"python_started":True} or manifest["claim_boundary"]!=claim: raise RuntimeError("T0 guard failed-start semantic evidence differs")
    if len(manifest["ordered_evidence_assets"])!=5 or [row.get("order") for row in manifest["ordered_evidence_assets"]]!=list(range(1,6)) or any(set(row)!={"order","role","path","bytes","file_sha256","internal_hash_field","internal_hash_value"} for row in manifest["ordered_evidence_assets"]): raise RuntimeError("T0 guard failed-start ordered evidence differs")
    return manifest

def validate_proof(value:dict[str,Any],repo:Path|None=None)->None:
    if set(value)!=PROOF_KEYS or value["schema_version"]!=PROOF_SCHEMA or value["status"]!=PROOF_STATUS or value["mechanism"]!=GUARD_MECHANISM or value["run_identity"]!=GUARD_RUN_ID: raise RuntimeError("T0 guard proof identity differs")
    if value["proof_sha256"]!=sha(canonical_json({k:v for k,v in value.items() if k!="proof_sha256"})): raise RuntimeError("T0 guard proof self hash differs")
    if value["failed_start_binding"]!=FAILED_START_BINDING or value["decision_boundary"]!=PROOF_DECISION: raise RuntimeError("T0 guard proof binding differs")
    rows=value["case_results"]
    row_keys={"index","name","mutation_kind","mutated_source_sha256","expected_outcome","observed_outcome","observed_error_type","qualifies"}
    if not isinstance(rows,list) or len(rows)!=5 or [r.get("name") for r in rows]!=GUARD_CASE_NAMES or [r.get("index") for r in rows]!=list(range(5)) or any(set(r)!=row_keys for r in rows) or [r["expected_outcome"] for r in rows]!=["pass","reject","reject","reject","reject"] or [r["observed_outcome"] for r in rows]!=["pass","reject","reject","reject","reject"] or [r["observed_error_type"] for r in rows]!=[None,"RuntimeError","RuntimeError","RuntimeError","RuntimeError"] or not all(r.get("qualifies") is True for r in rows): raise RuntimeError("T0 guard proof cases differ")
    expected_safety={"cuda_visible_devices":"","cuda_initialized":False,"torch_imported":False,"transformers_imported":False,"tokenizer_loaded":False,"model_loaded":False,"optimizer_constructed":False,"scientific_forwards":0,"training_operations":0,"validation_opens":0,"heldout_opens":0,"network_attempts":0,"output_namespace_fresh":True,"object_census_errors":0,"object_census_uninspectable":0}
    if value["safety"]!=expected_safety: raise RuntimeError("T0 guard proof safety differs")
    memory=value["memory"]
    if set(memory)!={"expected_labels","rows","label_sha256","complete"} or memory["expected_labels"]!=MEMORY_LABELS or [row.get("label") for row in memory["rows"]]!=MEMORY_LABELS or memory["label_sha256"]!=sha(canonical_json(MEMORY_LABELS)) or memory["complete"] is not True: raise RuntimeError("T0 guard proof memory differs")
    for index,row in enumerate(memory["rows"]):
        if set(row)!={"index","label","rss_bytes"} or row["index"]!=index or not isinstance(row["rss_bytes"],int) or row["rss_bytes"]<0: raise RuntimeError("T0 guard proof memory row differs")
    source=value["source_evidence"]; source_keys={"baseline_runner_file_sha256","repaired_runner_file_sha256","baseline_normalized_ast_sha256","repaired_normalized_ast_sha256","normalized_ast_equal","guard_function_ast_sha256","added_top_level_functions","full_forbidden_literals_absent_from_repaired_source","only_allowed_surface_changed"}
    audit=value["asset_audit"]; audit_keys={"target_count","pre_entries","pre_sha256","post_entries","post_sha256","all_exact"}; entry_keys={"path","sha256","bytes"}
    if set(source)!=source_keys or source["baseline_runner_file_sha256"]!="d6f8fcfb153e81c6aaf5faa42caf1700abba25a0594108ea8db4837034ca6bee" or source["repaired_runner_file_sha256"]!="29a7cb25cd3ec5c909716bbf30b35254423f1a9e159ec48134b8ec31fae0ab34" or source["baseline_normalized_ast_sha256"]!=source["repaired_normalized_ast_sha256"] or source["normalized_ast_equal"] is not True or source["added_top_level_functions"]!=["validate_static_exposure_source"] or source["full_forbidden_literals_absent_from_repaired_source"] is not True or source["only_allowed_surface_changed"] is not True: raise RuntimeError("T0 guard proof source closure differs")
    if set(audit)!=audit_keys or audit["target_count"]!=46 or audit["pre_entries"]!=audit["post_entries"] or len(audit["pre_entries"])!=46 or any(set(row)!=entry_keys for row in audit["pre_entries"]) or audit["pre_sha256"]!=sha(canonical_json(audit["pre_entries"])) or audit["post_sha256"]!=sha(canonical_json(audit["post_entries"])) or audit["all_exact"] is not True: raise RuntimeError("T0 guard proof asset closure differs")
    resources=value["resources"]
    if set(resources)!={"minimum_ram_gib","minimum_disk_gib","maximum_artifact_bytes","timing","observed_rss_peak_bytes","artifact_bytes"} or resources["minimum_ram_gib"]!=8 or resources["minimum_disk_gib"]!=8 or resources["maximum_artifact_bytes"]!=1048576 or not isinstance(resources["observed_rss_peak_bytes"],int) or resources["observed_rss_peak_bytes"]<0 or not isinstance(resources["artifact_bytes"],int) or not 0<=resources["artifact_bytes"]<=1048576: raise RuntimeError("T0 guard proof resources differ")
    freeze=value["full_freeze"]
    if set(freeze)!={"head_before","head_after","tree_before","tree_after","clean_before","clean_after","assets_pre_sha256","assets_post_sha256","complete"} or freeze["head_before"]!=value["execution_commit"] or freeze["head_before"]!=freeze["head_after"] or freeze["tree_before"]!=value["tree_sha256"] or freeze["tree_before"]!=freeze["tree_after"] or freeze["clean_before"] is not True or freeze["clean_after"] is not True or freeze["assets_pre_sha256"]!=audit["pre_sha256"] or freeze["assets_post_sha256"]!=audit["post_sha256"] or freeze["complete"] is not True: raise RuntimeError("T0 guard proof freeze differs")
    if repo is not None:
        repaired=(repo/RUNNER_REL).read_text(); baseline=subprocess.check_output(["git","show",f"67b21d2ccd7cc3189154f67a80fe8db6abd3ec7b:{RUNNER_REL}"],cwd=repo).decode(); expected_rows,guard_hash=case_results(repaired); baseline_hash,baseline_nodes=normalized_source(baseline); repaired_hash,repaired_nodes=normalized_source(repaired)
        if rows!=expected_rows or baseline_nodes!=repaired_nodes or source!={"baseline_runner_file_sha256":sha(baseline.encode()),"repaired_runner_file_sha256":sha(repaired.encode()),"baseline_normalized_ast_sha256":baseline_hash,"repaired_normalized_ast_sha256":repaired_hash,"normalized_ast_equal":True,"guard_function_ast_sha256":guard_hash,"added_top_level_functions":["validate_static_exposure_source"],"full_forbidden_literals_absent_from_repaired_source":all(name not in repaired for name in ("-".join(("validation","bank.json")),"-".join(("heldout","bank.json")))),"only_allowed_surface_changed":True}: raise RuntimeError("T0 guard proof source replay differs")
        if asset_entries(repo)!=audit["pre_entries"]: raise RuntimeError("T0 guard proof asset replay differs")

def validate_failure(value:dict[str,Any])->None:
    expected=(PROOF_KEYS-{"proof_sha256"})|{"error_type","error_message","traceback","stage","execution_progress","audit_errors","failure_sha256"}
    if set(value)!=expected or value["schema_version"]!=FAILURE_SCHEMA or value["mechanism"]!=GUARD_MECHANISM or value["run_identity"]!=GUARD_RUN_ID: raise RuntimeError("T0 guard failure identity differs")
    taxonomy={"h_iter_phase1_t0_preflight_r1_guard_proof_incomplete":"T0PreflightR1GuardProofError","h_iter_phase1_t0_preflight_r1_guard_proof_boundary_rejected":"T0PreflightR1GuardProofBoundaryRejected","infrastructure_invalid":"InfrastructureInvalid"}
    if taxonomy.get(value["status"])!=value["error_type"]: raise RuntimeError("T0 guard failure taxonomy differs")
    stages=["startup","runtime","assets","failed_start","baseline_source","repaired_source","cases","safety","audit","terminal_publication"]
    if value["stage"] not in stages or value["execution_progress"]!={"stage":value["stage"],"memory_rows_completed":len(value["memory"]["rows"])}: raise RuntimeError("T0 guard failure progress differs")
    memory=value["memory"]
    if memory["expected_labels"]!=MEMORY_LABELS or [row.get("label") for row in memory["rows"]]!=MEMORY_LABELS[:len(memory["rows"])] or memory["label_sha256"]!=sha(canonical_json(MEMORY_LABELS)) or memory["complete"] is not (len(memory["rows"])==14): raise RuntimeError("T0 guard failure memory differs")
    if value["status"]=="infrastructure_invalid" and not value["audit_errors"]: raise RuntimeError("T0 guard infrastructure evidence missing")
    if value["status"]!="infrastructure_invalid" and value["audit_errors"]!=[]: raise RuntimeError("T0 guard noninfra audit differs")
    if value["failure_sha256"]!=sha(canonical_json({k:v for k,v in value.items() if k!="failure_sha256"})): raise RuntimeError("T0 guard failure self hash differs")

def atomic_terminal(output:Path,name:str,value:dict[str,Any],validator:Any)->None:
    data=canonical_json(value)+b"\n"
    if len(data)>1048576 or output.exists() or output.is_symlink(): raise RuntimeError("T0 guard proof output differs")
    output.mkdir(mode=0o700,parents=True); tmp=output/("."+name+".tmp")
    fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp,output/name); dfd=os.open(output,os.O_RDONLY|os.O_DIRECTORY); os.fsync(dfd); os.close(dfd)
    if (output/name).read_bytes()!=data: raise RuntimeError("T0 guard proof reopen differs")
    validator(strict_loads(data))

def run(repo:Path,execution_commit:str,plan_file_sha256:str,output:Path)->None:
    started=time.monotonic_ns(); mark("ENTRY")
    STATE["stage"]="runtime"
    plan_path=repo/PLAN_REL
    if plan_path.is_symlink() or not plan_path.is_file() or file_sha(plan_path)!=plan_file_sha256: raise RuntimeError("T0 guard proof plan differs")
    plan=strict_loads(plan_path.read_bytes()); validate_guard_plan(plan,repo=repo); verify_runtime(repo,plan,output); mark("RUNTIME_VERIFIED")
    head=git(repo,"rev-parse","HEAD"); tree=git(repo,"rev-parse","HEAD^{tree}"); clean=not bool(git(repo,"status","--porcelain","--untracked-files=all"))
    if head!=execution_commit or not clean or plan["mechanism_code_commit"]!=head: raise RuntimeError("T0 guard proof freeze differs")
    mark("FULL_FREEZE_PREFLIGHT_VERIFIED"); STATE["stage"]="assets"
    pre=asset_entries(repo); pre_sha=sha(canonical_json(pre))
    if {row["path"]:row["sha256"] for row in pre}!=plan["asset_sha256"]: raise RuntimeError("T0 guard proof assets differ")
    mark("ASSETS_PREFLIGHT_VERIFIED"); STATE["stage"]="failed_start"; validate_failed_start(repo); mark("FAILED_START_BOUND"); STATE["stage"]="baseline_source"
    baseline=subprocess.check_output(["git","show",f"67b21d2ccd7cc3189154f67a80fe8db6abd3ec7b:{RUNNER_REL}"],cwd=repo).decode()
    repaired=(repo/RUNNER_REL).read_text(); baseline_hash,baseline_dump=normalized_source(baseline); mark("BASELINE_SOURCE_VERIFIED")
    repaired_hash,repaired_dump=normalized_source(repaired); mark("REPAIRED_SOURCE_VERIFIED"); STATE["stage"]="cases"
    if baseline_dump!=repaired_dump: raise RuntimeError("T0 guard normalized source differs")
    rows,guard_hash=case_results(repaired)
    for row,label in zip(rows,MEMORY_LABELS[7:12]):
        if not row["qualifies"]: raise RuntimeError("T0 guard case differs")
        mark(label)
    STATE["stage"]="safety"; errors=uninspectable=0
    for obj in gc.get_objects():
        try:
            module=type(obj).__module__
            if not isinstance(module,str): uninspectable+=1
        except Exception: errors+=1
    safety={"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"cuda_initialized":False,"torch_imported":"torch" in sys.modules,"transformers_imported":"transformers" in sys.modules,"tokenizer_loaded":False,"model_loaded":False,"optimizer_constructed":False,"scientific_forwards":0,"training_operations":0,"validation_opens":0,"heldout_opens":0,"network_attempts":STATE["network_attempts"],"output_namespace_fresh":not output.exists(),"object_census_errors":errors,"object_census_uninspectable":uninspectable}
    if any((safety["torch_imported"],safety["transformers_imported"],safety["tokenizer_loaded"],safety["model_loaded"],safety["optimizer_constructed"],safety["scientific_forwards"],safety["training_operations"],safety["validation_opens"],safety["heldout_opens"],safety["network_attempts"])): raise T0PreflightR1GuardProofBoundaryRejected("T0 guard proof boundary exposure observed")
    if safety!={"cuda_visible_devices":"","cuda_initialized":False,"torch_imported":False,"transformers_imported":False,"tokenizer_loaded":False,"model_loaded":False,"optimizer_constructed":False,"scientific_forwards":0,"training_operations":0,"validation_opens":0,"heldout_opens":0,"network_attempts":0,"output_namespace_fresh":True,"object_census_errors":0,"object_census_uninspectable":0}: raise InfrastructureInvalid("T0 guard proof safety differs")
    mark("SAFETY_AUDIT_COMPLETE"); STATE["stage"]="audit"
    signal.alarm(0); signal.alarm(119)
    post=asset_entries(repo); post_sha=sha(canonical_json(post)); head_after=git(repo,"rev-parse","HEAD"); tree_after=git(repo,"rev-parse","HEAD^{tree}"); clean_after=not bool(git(repo,"status","--porcelain","--untracked-files=all"))
    if pre!=post or head_after!=head or tree_after!=tree or not clean_after: raise RuntimeError("T0 guard proof postflight differs")
    mark("TERMINAL_PREWRITE")
    source_evidence={"baseline_runner_file_sha256":sha(baseline.encode()),"repaired_runner_file_sha256":sha(repaired.encode()),"baseline_normalized_ast_sha256":baseline_hash,"repaired_normalized_ast_sha256":repaired_hash,"normalized_ast_equal":True,"guard_function_ast_sha256":guard_hash,"added_top_level_functions":["validate_static_exposure_source"],"full_forbidden_literals_absent_from_repaired_source":all(name not in repaired for name in ("-".join(("validation","bank.json")),"-".join(("heldout","bank.json")))),"only_allowed_surface_changed":True}
    timing={"outer":720,"startup":60,"compute":300,"compute_alarm":299,"audit":120,"audit_alarm":119,"failure":90,"failure_alarm":89,"terminal":30,"terminal_alarm":29,"postexit":30,"reserve":90,"success_entry":480,"compute_failure_entry":450,"audit_failure_entry":570,"prior_terminal_failure_entry":600}
    proof={"schema_version":PROOF_SCHEMA,"status":PROOF_STATUS,"mechanism":GUARD_MECHANISM,"run_identity":GUARD_RUN_ID,"execution_commit":head,"mechanism_code_commit":plan["mechanism_code_commit"],"tree_sha256":tree,"plan_file_sha256":plan_file_sha256,"plan_sha256":plan["plan_sha256"],"runtime":plan["runtime"],"asset_audit":{"target_count":46,"pre_entries":pre,"pre_sha256":pre_sha,"post_entries":post,"post_sha256":post_sha,"all_exact":True},"failed_start_binding":FAILED_START_BINDING,"source_evidence":source_evidence,"case_results":rows,"safety":safety,"resources":{"minimum_ram_gib":8,"minimum_disk_gib":8,"maximum_artifact_bytes":1048576,"timing":timing,"observed_rss_peak_bytes":rss(),"artifact_bytes":0},"memory":{"expected_labels":MEMORY_LABELS,"rows":STATE["memory"],"label_sha256":sha(canonical_json(MEMORY_LABELS)),"complete":True},"full_freeze":{"head_before":head,"head_after":head_after,"tree_before":tree,"tree_after":tree_after,"clean_before":clean,"clean_after":clean_after,"assets_pre_sha256":pre_sha,"assets_post_sha256":post_sha,"complete":True},"decision_boundary":PROOF_DECISION,"proof_sha256":""}
    proof["proof_sha256"]=sha(canonical_json({k:v for k,v in proof.items() if k!="proof_sha256"})); validate_proof(proof,repo); STATE["stage"]="terminal_publication"; signal.alarm(0); signal.alarm(29); atomic_terminal(output,PROOF_NAME,proof,lambda value:validate_proof(value,repo)); signal.alarm(0)

def publish_failure(output:Path,execution_commit:str,plan_file_sha256:str,error:BaseException)->None:
    if output.exists() or output.is_symlink(): return
    if isinstance(error,T0PreflightR1GuardProofBoundaryRejected): status="h_iter_phase1_t0_preflight_r1_guard_proof_boundary_rejected"; error_type="T0PreflightR1GuardProofBoundaryRejected"; audit=[]
    elif isinstance(error,(TimeoutError,OSError,MemoryError,InfrastructureInvalid)): status="infrastructure_invalid"; error_type="InfrastructureInvalid"; audit=[f"{type(error).__name__}: {error}"]
    else: status="h_iter_phase1_t0_preflight_r1_guard_proof_incomplete"; error_type="T0PreflightR1GuardProofError"; audit=[]
    memory={"expected_labels":MEMORY_LABELS,"rows":list(STATE["memory"]),"label_sha256":sha(canonical_json(MEMORY_LABELS)),"complete":len(STATE["memory"])==14}
    value={key:None for key in PROOF_KEYS if key!="proof_sha256"}
    value.update({"schema_version":FAILURE_SCHEMA,"status":status,"mechanism":GUARD_MECHANISM,"run_identity":GUARD_RUN_ID,"execution_commit":execution_commit,"plan_file_sha256":plan_file_sha256,"safety":{"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"network_attempts":STATE["network_attempts"]},"memory":memory,"decision_boundary":PROOF_DECISION,"error_type":error_type,"error_message":str(error),"traceback":traceback.format_exc(),"stage":STATE["stage"],"execution_progress":{"stage":STATE["stage"],"memory_rows_completed":len(STATE["memory"])},"audit_errors":audit,"failure_sha256":""})
    value["failure_sha256"]=sha(canonical_json({k:v for k,v in value.items() if k!="failure_sha256"})); validate_failure(value); atomic_terminal(output,FAILURE_NAME,value,validate_failure)

def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--repo",type=Path,required=True); parser.add_argument("--execution-commit",required=True); parser.add_argument("--plan-file-sha256",required=True); parser.add_argument("--run-id",required=True); parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--validate-terminal",action="store_true"); args=parser.parse_args()
    repo=args.repo.resolve(); output=args.output_dir
    if args.run_id!=GUARD_RUN_ID or len(args.execution_commit)!=40 or len(args.plan_file_sha256)!=64: raise SystemExit(64)
    if args.validate_terminal:
        names=sorted(p.name for p in output.iterdir())
        if names==[PROOF_NAME]: validate_proof(strict_loads((output/PROOF_NAME).read_bytes()),repo)
        elif names==[FAILURE_NAME]: validate_failure(strict_loads((output/FAILURE_NAME).read_bytes()))
        else: raise RuntimeError("T0 guard terminal inventory differs")
        return
    NetworkGuard().install(); signal.signal(signal.SIGALRM,timeout_alarm); signal.alarm(299)
    try: run(repo,args.execution_commit,args.plan_file_sha256,output)
    except BaseException as error:
        traceback.print_exc()
        signal.alarm(0); signal.alarm(89)
        try: publish_failure(output,args.execution_commit,args.plan_file_sha256,error); signal.alarm(0)
        except BaseException: traceback.print_exc()
        raise SystemExit(2) from error

if __name__=="__main__": main()
