#!/usr/bin/env python3
"""Generate a diverse synthetic bootstrap for programmatic episodic memory.

V2 expands the original PRO-LONG-inspired seed into many semantic domains and
three important negative/control policies:
  * retrieve programmatically when a decision genuinely depends on prior events;
  * use IPython without touching history when the current request only needs computation;
  * obey authoritative current-turn information without consulting stale prior history.

The corpus is deterministic, structurally self-validating, and intentionally
keeps familiar-heldout and semantic-OOD splits outside training. It is bootstrap
supervision, not evidence of mastered long-horizon memory. No reasoning_content
is fabricated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    from datasets import Dataset
except Exception:  # pragma: no cover
    Dataset = None

DATASET_NAME = "programmatic-episodic-memory-v2"
DEFAULT_SEED = 20260813

HISTORY_PATHS = (
    "/workspace/history.log",
    "/workspace/session.log",
    "/workspace/events.log",
    "/workspace/journal.log",
)
NOTES_PATHS = (
    "/workspace/notes.txt",
    "/workspace/summary.md",
    "/workspace/cache.txt",
)

IPYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "ipython",
        "description": (
            "Execute Python in a persistent IPython kernel. Variables, imports, "
            "and derived indexes persist across calls. Files in /workspace may "
            "be read and written from Python."
        ),
        "parameters": {
            "type": "object",
            "required": ["code"],
            "properties": {"code": {"type": "string"}},
        },
    },
}


@dataclass(frozen=True)
class Example:
    messages: list[dict]
    files: dict[str, str]
    metadata: dict


def system_prompt(history_path: str, notes_hint: str = "") -> str:
    extra = (
        f"\nDerived notes may exist at {notes_hint}. They are caches, not source of truth."
        if notes_hint
        else ""
    )
    return f"""You are operating inside a persistent Prime Agent workspace.

The session's lossless append-only interaction history is available at
{history_path}. It contains prior observations, actions, outcomes, decisions,
corrections, and other durable events. Treat that append-only history as the
source of truth for what actually happened.{extra}

Use programmatic retrieval when the current decision depends on earlier events:
search or parse the history with Python, return only the small relevant slice to
your active context, and compute over it when useful. Do not guess distant facts
from memory. Prefer the latest valid event when earlier entries were corrected,
revoked, retracted, superseded, or invalidated.

Files in /workspace that summarize earlier work are derived state and may be
stale. Reuse compact indexes when repeated lookups make that worthwhile, but
verify conflicts against the append-only history.

Do not touch the history when the current request is self-contained or when the
current user turn explicitly supplies the authoritative value. IPython may still
be useful for computation that has nothing to do with history.

Answer the user's actual request concisely after any necessary retrieval.
"""


def tool_call(call_id: str, code: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "ipython", "arguments": json.dumps({"code": code})},
    }


def assistant_tool(call_id: str, code: str) -> dict:
    return {"role": "assistant", "content": None, "tool_calls": [tool_call(call_id, code)]}


def tool_result(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def kv_line(i: int, event_type: str, **fields: object) -> str:
    encoded = " ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in fields.items())
    return f"[{i:05d}] TYPE={event_type} {encoded}".rstrip()


def history_path_for(rng: random.Random, instance: int) -> str:
    return HISTORY_PATHS[(instance + rng.randrange(len(HISTORY_PATHS))) % len(HISTORY_PATHS)]


def notes_path_for(rng: random.Random, instance: int) -> str:
    return NOTES_PATHS[(instance + rng.randrange(len(NOTES_PATHS))) % len(NOTES_PATHS)]


def noise_line(rng: random.Random, i: int) -> str:
    return kv_line(
        i,
        "NOISE",
        domain=rng.choice(["software", "research", "data", "project", "agents", "ops", "experiments"]),
        action=rng.choice(["inspect", "measure", "review", "sync", "cache", "compare", "triage"]),
        object=rng.choice(["alpha", "beta", "gamma", "queue", "dataset", "artifact", "worker"]),
        token=f"d{i:05d}-{rng.randrange(10000,99999)}",
    )


def insert_events(rng: random.Random, horizon: int, events: list[tuple[int, str]]) -> str:
    by_pos = {pos: text for pos, text in events}
    out = [by_pos.get(i, noise_line(rng, i)) for i in range(1, horizon + 1)]
    return "\n".join(out) + "\n"


def wording(instance: int, explicit: str, natural: str, compacted: str | None = None) -> tuple[str, str]:
    mode = instance % (3 if compacted else 2)
    if compacted and mode == 2:
        return compacted, "context_reset"
    if mode == 0:
        return explicit, "explicit_history"
    return natural, "natural"


def base_row(ex: Example) -> dict:
    return {
        "messages_json": json.dumps(ex.messages, ensure_ascii=False),
        "tools": json.dumps([IPYTHON_TOOL], ensure_ascii=False),
        "workspace_files_json": json.dumps(ex.files, ensure_ascii=False),
        "metadata_json": json.dumps(ex.metadata, ensure_ascii=False, sort_keys=True),
    }


def meta(*, split: str, family: str, domain: str, instance: int, horizon: int,
         history_path: str, history_format: str, instruction_level: str,
         requires_history: bool, uses_ipython: bool, retrieval_policy: str,
         expected_answer: str, notes_path: str | None = None) -> dict:
    return {
        "dataset": DATASET_NAME,
        "split": split,
        "family": family,
        "domain": domain,
        "instance": instance,
        "horizon_events": horizon,
        "history_path": history_path,
        "history_format": history_format,
        "instruction_level": instruction_level,
        "requires_history": requires_history,
        "uses_ipython": uses_ipython,
        "retrieval_policy": retrieval_policy,
        "expected_answer": expected_answer,
        "notes_path": notes_path,
    }


def _filter_code(history_path: str, event_type: str, conditions: list[tuple[str, str]], tail: int = 4) -> str:
    literals = [f"{key}={json.dumps(value)}" for key, value in conditions]
    tests = " and ".join([f"'TYPE={event_type}' in line"] + [f"{literal!r} in line" for literal in literals])
    return (
        "from pathlib import Path\n"
        f"matches = [line.strip() for line in Path({history_path!r}).open() if {tests}]\n"
        f"matches[-{tail}:]"
    )


def latest_state_example(rng, split, instance, horizon):
    choices = {
        "deploy_region": ["eu-west", "us-east", "eu-central"],
        "active_model": ["model-a", "model-b", "model-c"],
        "release_channel": ["canary", "beta", "stable"],
        "primary_dataset": ["snapshot-17", "snapshot-21", "snapshot-24"],
    }
    key = rng.choice(list(choices)); history_path = history_path_for(rng, instance); vals = choices[key]
    p = [horizon // 6, horizon // 2, horizon - 5]
    history = insert_events(rng, horizon, [(p[i], kv_line(p[i], "STATE", key=key, value=vals[i], status="active")) for i in range(3)])
    q, level = wording(instance, f"Recover current `{key}` from session history. Return only value.", f"What is the current `{key}`? Return only value.", f"Context was compacted. What is current `{key}`? Return only value.")
    code = _filter_code(history_path, "STATE", [("key", key), ("status", "active")], 3)
    lines = [kv_line(p[i], "STATE", key=key, value=vals[i], status="active") for i in range(3)]
    messages = [{"role":"system","content":system_prompt(history_path)}, {"role":"user","content":q}, assistant_tool("pem-latest-1", code), tool_result("pem-latest-1", repr(lines)), {"role":"assistant","content":vals[-1]}]
    return Example(messages, {history_path: history}, meta(split=split,family="latest_state",domain="project",instance=instance,horizon=horizon,history_path=history_path,history_format="kv",instruction_level=level,requires_history=True,uses_ipython=True,retrieval_policy="latest_valid",expected_answer=vals[-1]))


def accepted_requirement_example(rng, split, instance, horizon):
    key = rng.choice(["api_version","storage_backend","serialization","retry_policy"])
    vals = {"api_version":["v2","v3","v4"],"storage_backend":["sqlite","postgres","duckdb"],"serialization":["json","msgpack","parquet"],"retry_policy":["fixed","linear","exponential"]}[key]
    history_path=history_path_for(rng,instance); p=[horizon//5,horizon//2,horizon-5]; statuses=["proposed","accepted","rejected"]
    history=insert_events(rng,horizon,[(p[i],kv_line(p[i],"REQUIREMENT",key=key,value=vals[i],status=statuses[i])) for i in range(3)])
    q,level=wording(instance,f"Find accepted `{key}` in history. Return only value.",f"Which `{key}` should we follow now? Return only value.",f"After context reset, which accepted `{key}` should we follow?")
    code=_filter_code(history_path,"REQUIREMENT",[("key",key),("status","accepted")],3)
    selected=kv_line(p[1],"REQUIREMENT",key=key,value=vals[1],status="accepted")
    messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},assistant_tool("pem-req-1",code),tool_result("pem-req-1",repr([selected])),{"role":"assistant","content":vals[1]}]
    return Example(messages,{history_path:history},meta(split=split,family="accepted_requirement",domain="project",instance=instance,horizon=horizon,history_path=history_path,history_format="kv",instruction_level=level,requires_history=True,uses_ipython=True,retrieval_policy="accepted_not_surface_latest",expected_answer=vals[1]))


def successful_attempt_example(rng, split, instance, horizon):
    task=rng.choice(["parser_fix","index_build","migration","latency_patch"]); history_path=history_path_for(rng,instance); cand=[f"{task}-a",f"{task}-b",f"{task}-c"]
    p=[horizon//4,horizon//2,horizon-7]; outcomes=["failed","succeeded","failed"]
    history=insert_events(rng,horizon,[(p[i],kv_line(p[i],"ATTEMPT",task=task,candidate=cand[i],outcome=outcomes[i])) for i in range(3)])
    q,level=wording(instance,f"Which `{task}` candidate succeeded earlier? Return only id.",f"Which `{task}` candidate should we preserve? Return only id.",f"Resume after compaction: which `{task}` candidate succeeded?")
    code=_filter_code(history_path,"ATTEMPT",[("task",task),("outcome","succeeded")],4); line=kv_line(p[1],"ATTEMPT",task=task,candidate=cand[1],outcome="succeeded")
    messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},assistant_tool("pem-attempt-1",code),tool_result("pem-attempt-1",repr([line])),{"role":"assistant","content":cand[1]}]
    return Example(messages,{history_path:history},meta(split=split,family="successful_attempt",domain="software",instance=instance,horizon=horizon,history_path=history_path,history_format="kv",instruction_level=level,requires_history=True,uses_ipython=True,retrieval_policy="outcome_filter",expected_answer=cand[1]))


def correction_aggregate_example(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); metric=rng.choice(["accepted_rows","verified_items","approved_cost","completed_jobs"]); base=rng.randint(30,80); deltas=[rng.randint(5,18) for _ in range(3)]; correction=rng.randint(1,6)
    p=[horizon//7,horizon//3,horizon//2,horizon*2//3,horizon-4]
    events=[(p[0],kv_line(p[0],"METRIC_BASE",metric=metric,value=base)),(p[1],kv_line(p[1],"METRIC_DELTA",metric=metric,event="e1",delta=deltas[0])),(p[2],kv_line(p[2],"METRIC_DELTA",metric=metric,event="e2",delta=deltas[1])),(p[3],kv_line(p[3],"METRIC_DELTA",metric=metric,event="e3",delta=deltas[2])),(p[4],kv_line(p[4],"METRIC_CORRECTION",metric=metric,event="e2",adjustment=-correction))]
    history=insert_events(rng,horizon,events); expected=base+sum(deltas)-correction
    q,level=wording(instance,f"Reconstruct current `{metric}` from base, deltas and corrections. Return integer.",f"What is current `{metric}` total? Return integer.",f"After compaction, recover current `{metric}` total.")
    code=("from pathlib import Path\nimport re\n"+f"metric={metric!r}\n"+f"lines=[x.strip() for x in Path({history_path!r}).open() if f'metric=\\\"{{metric}}\\\"' in x]\n"+"base=int(re.search(r'value=(-?\\d+)',next(x for x in lines if 'TYPE=METRIC_BASE' in x)).group(1))\n"+"deltas=[int(re.search(r'delta=(-?\\d+)',x).group(1)) for x in lines if 'TYPE=METRIC_DELTA' in x]\n"+"corrections=[int(re.search(r'adjustment=(-?\\d+)',x).group(1)) for x in lines if 'TYPE=METRIC_CORRECTION' in x]\nbase+sum(deltas)+sum(corrections)")
    messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},assistant_tool("pem-agg-1",code),tool_result("pem-agg-1",str(expected)),{"role":"assistant","content":str(expected)}]
    return Example(messages,{history_path:history},meta(split=split,family="correction_aggregate",domain="data",instance=instance,horizon=horizon,history_path=history_path,history_format="kv",instruction_level=level,requires_history=True,uses_ipython=True,retrieval_policy="retrieve_then_compute",expected_answer=str(expected)))


def provenance_conflict_example(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); claim=rng.choice(["cache_is_safe","feature_is_enabled","dataset_is_complete","latency_is_regressed"]); sources=[f"S{rng.randint(100,999)}" for _ in range(3)]; p=[horizon//5,horizon//2,horizon-6]; verdicts=["supports","supports","contradicts"]
    history=insert_events(rng,horizon,[(p[i],kv_line(p[i],"EVIDENCE",claim=claim,source=sources[i],verdict=verdicts[i])) for i in range(3)]); expected=f"contradicted by {sources[-1]}"
    q,level=wording(instance,f"Recover latest evidential status of `{claim}` and source.",f"Where do we currently stand on `{claim}`? Give verdict and source.",f"Resume research after compaction: current verdict on `{claim}` and source?")
    code=_filter_code(history_path,"EVIDENCE",[("claim",claim)],4); lines=[kv_line(p[i],"EVIDENCE",claim=claim,source=sources[i],verdict=verdicts[i]) for i in range(3)]
    messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},assistant_tool("pem-prov-1",code),tool_result("pem-prov-1",repr(lines)),{"role":"assistant","content":expected}]
    return Example(messages,{history_path:history},meta(split=split,family="provenance_conflict",domain="research",instance=instance,horizon=horizon,history_path=history_path,history_format="kv",instruction_level=level,requires_history=True,uses_ipython=True,retrieval_policy="latest_evidence_with_provenance",expected_answer=expected))


def checkpoint_resume_example(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); run=f"run-{rng.randint(200,999)}"; steps=[4,8,12,16]; states=["stable","stable","corrupt","started"]; p=[horizon//7,horizon//3,horizon//2,horizon-4]
    history=insert_events(rng,horizon,[(p[i],kv_line(p[i],"CHECKPOINT",run=run,step=steps[i],state=states[i])) for i in range(4)])
    q,level=wording(instance,f"Recover latest stable checkpoint for `{run}` after later corruption. Return step.",f"Which step should `{run}` resume from? Return number.",f"Context reset during recovery. Which stable step should `{run}` resume from?")
    code=_filter_code(history_path,"CHECKPOINT",[("run",run),("state","stable")],4); lines=[kv_line(p[i],"CHECKPOINT",run=run,step=steps[i],state=states[i]) for i in range(2)]
    messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},assistant_tool("pem-ckpt-1",code),tool_result("pem-ckpt-1",repr(lines)),{"role":"assistant","content":"8"}]
    return Example(messages,{history_path:history},meta(split=split,family="checkpoint_resume",domain="experiments",instance=instance,horizon=horizon,history_path=history_path,history_format="kv",instruction_level=level,requires_history=True,uses_ipython=True,retrieval_policy="latest_stable",expected_answer="8"))


def stale_note_override_example(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); notes_path=notes_path_for(rng,instance); key=rng.choice(["owner","target_branch","deadline","primary_endpoint"]); old,new={"owner":("alice","bob"),"target_branch":("main","release/v2"),"deadline":("2026-09-01","2026-09-08"),"primary_endpoint":("api-a","api-b")}[key]; p=[horizon//3,horizon-5]
    history=insert_events(rng,horizon,[(p[0],kv_line(p[0],"DECISION",key=key,value=old,status="accepted")),(p[1],kv_line(p[1],"DECISION",key=key,value=new,status="accepted"))]); notes=f"# derived summary\n{key}: {old}\n# generated before latest decision\n"
    q,level=wording(instance,f"Workspace summary may be stale. Verify current `{key}`. Return value.",f"What is current accepted `{key}`? Return value.",f"After compaction, verify current accepted `{key}`.")
    code=f"from pathlib import Path\nkey={key!r}\nnote=Path({notes_path!r}).read_text().strip()\nmatches=[x.strip() for x in Path({history_path!r}).open() if 'TYPE=DECISION' in x and f'key=\\\"{{key}}\\\"' in x and 'status=\\\"accepted\\\"' in x]\n{{'derived_note':note,'history_tail':matches[-2:]}}"
    lines=[kv_line(p[0],"DECISION",key=key,value=old,status="accepted"),kv_line(p[1],"DECISION",key=key,value=new,status="accepted")]
    messages=[{"role":"system","content":system_prompt(history_path,notes_path)},{"role":"user","content":q},assistant_tool("pem-stale-1",code),tool_result("pem-stale-1",repr({"derived_note":notes.strip(),"history_tail":lines})),{"role":"assistant","content":new}]
    return Example(messages,{history_path:history,notes_path:notes},meta(split=split,family="stale_note_override",domain="project",instance=instance,horizon=horizon,history_path=history_path,history_format="kv",instruction_level=level,requires_history=True,uses_ipython=True,retrieval_policy="source_truth_over_stale_cache",expected_answer=new,notes_path=notes_path))


def repeated_lookup_index_example(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); keys=["alpha","beta","gamma","delta"]; vals={k:f"{k}-{rng.randint(10,99)}" for k in keys}; positions=[horizon//5,horizon*2//5,horizon*3//5,horizon-5]
    history=insert_events(rng,horizon,[(p,kv_line(p,"BINDING",key=k,value=vals[k],status="current")) for p,k in zip(positions,keys)]); first,second=rng.sample(keys,2)
    q,level=wording(instance,f"Recover current binding for `{first}`.",f"What is `{first}` currently bound to?",f"Context compacted. What is `{first}` currently bound to?")
    code1=f"from pathlib import Path\nimport re\nbindings={{}}\nfor line in Path({history_path!r}).open():\n    if 'TYPE=BINDING' not in line or 'status=\\\"current\\\"' not in line: continue\n    k=re.search(r'key=\\\"([^\\\"]+)\\\"',line).group(1)\n    v=re.search(r'value=\\\"([^\\\"]+)\\\"',line).group(1)\n    bindings[k]=v\nbindings"; code2=f"bindings[{second!r}]"
    messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},assistant_tool("pem-index-1",code1),tool_result("pem-index-1",repr(vals)),{"role":"assistant","content":vals[first]},{"role":"user","content":f"And `{second}`? Reuse useful state from prior lookup."},assistant_tool("pem-index-2",code2),tool_result("pem-index-2",repr(vals[second])),{"role":"assistant","content":vals[second]}]
    return Example(messages,{history_path:history},meta(split=split,family="repeated_lookup_index",domain="project",instance=instance,horizon=horizon,history_path=history_path,history_format="kv",instruction_level=level,requires_history=True,uses_ipython=True,retrieval_policy="build_then_reuse_index",expected_answer=vals[second]))


def multi_key_join_example(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); service=rng.choice(["payments","search","training","metrics"]); version=f"v{rng.randint(3,9)}"; region=rng.choice(["eu-central","us-west","ap-south"]); p=[horizon//3,horizon-6]
    history=insert_events(rng,horizon,[(p[0],kv_line(p[0],"DEPLOY",service=service,field="version",value=version,status="current")),(p[1],kv_line(p[1],"DEPLOY",service=service,field="region",value=region,status="current"))]); expected=f"{service}@{version} in {region}"
    q,level=wording(instance,f"Reconstruct current deployment for `{service}` as `<service>@<version> in <region>`.",f"What deployment of `{service}` is active?",f"Resume after compaction: what deployment of `{service}` is active?")
    code=_filter_code(history_path,"DEPLOY",[("service",service),("status","current")],5); lines=[kv_line(p[0],"DEPLOY",service=service,field="version",value=version,status="current"),kv_line(p[1],"DEPLOY",service=service,field="region",value=region,status="current")]
    messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},assistant_tool("pem-join-1",code),tool_result("pem-join-1",repr(lines)),{"role":"assistant","content":expected}]
    return Example(messages,{history_path:history},meta(split=split,family="multi_key_join",domain="ops",instance=instance,horizon=horizon,history_path=history_path,history_format="kv",instruction_level=level,requires_history=True,uses_ipython=True,retrieval_policy="join_distant_events",expected_answer=expected))


def context_reset_resume_example(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); objective=rng.choice(["finish_migration","publish_report","repair_pipeline","qualify_model"]); actions=["inspect blockers","validate latest artifact","run frozen acceptance screen"]; p=[horizon//5,horizon//2,horizon-5]
    history=insert_events(rng,horizon,[(p[i],kv_line(p[i],"PLAN",objective=objective,next_action=actions[i],status="active")) for i in range(3)])
    q,level=wording(instance,f"Use history to resume `{objective}` after context reset. Return latest next action.",f"We lost active context on `{objective}`. What should we do next?",f"Conversation compacted. Resume `{objective}` and return next action only.")
    code=_filter_code(history_path,"PLAN",[("objective",objective),("status","active")],3); lines=[kv_line(p[i],"PLAN",objective=objective,next_action=actions[i],status="active") for i in range(3)]
    messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},assistant_tool("pem-resume-1",code),tool_result("pem-resume-1",repr(lines)),{"role":"assistant","content":actions[-1]}]
    return Example(messages,{history_path:history},meta(split=split,family="context_reset_resume",domain="project",instance=instance,horizon=horizon,history_path=history_path,history_format="kv",instruction_level=level,requires_history=True,uses_ipython=True,retrieval_policy="recover_after_context_loss",expected_answer=actions[-1]))


def constraint_update_example(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); key=rng.choice(["max_parallelism","output_format","allowed_region","review_mode"]); old,new={"max_parallelism":("4","8"),"output_format":("csv","parquet"),"allowed_region":("eu-west","eu-central"),"review_mode":("manual","automatic")}[key]; p=[horizon//4,horizon-6]
    history=insert_events(rng,horizon,[(p[0],kv_line(p[0],"USER_CONSTRAINT",key=key,value=old,status="active")),(p[1],kv_line(p[1],"USER_CONSTRAINT",key=key,value=new,status="active",supersedes=old))])
    q,level=wording(instance,f"Recover current user constraint `{key}`. Return value.",f"What `{key}` should we obey now? Return value.",f"After compaction, continue under current `{key}`.")
    code=_filter_code(history_path,"USER_CONSTRAINT",[("key",key),("status","active")],3); lines=[kv_line(p[0],"USER_CONSTRAINT",key=key,value=old,status="active"),kv_line(p[1],"USER_CONSTRAINT",key=key,value=new,status="active",supersedes=old)]
    messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},assistant_tool("pem-constraint-1",code),tool_result("pem-constraint-1",repr(lines)),{"role":"assistant","content":new}]
    return Example(messages,{history_path:history},meta(split=split,family="constraint_update",domain="project",instance=instance,horizon=horizon,history_path=history_path,history_format="kv",instruction_level=level,requires_history=True,uses_ipython=True,retrieval_policy="latest_user_constraint",expected_answer=new))


def direct_control_example(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); a,b=rng.randint(11,90),rng.randint(11,90); history=insert_events(rng,horizon,[]); expected=str(a+b); q=f"Self-contained: what is {a} + {b}? Return integer."
    messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},{"role":"assistant","content":expected}]
    return Example(messages,{history_path:history},meta(split=split,family="direct_control",domain="control",instance=instance,horizon=horizon,history_path=history_path,history_format="kv",instruction_level="natural",requires_history=False,uses_ipython=False,retrieval_policy="no_retrieval_self_contained",expected_answer=expected))


def software_debug_resolution_example(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); bug=f"BUG-{rng.randint(100,999)}"; p=[horizon//6,horizon//3,horizon//2,horizon-5]; hypotheses=["timeout","schema mismatch","stale cache"]; patch=f"patch-{rng.randint(20,99)}"
    events=[(p[0],kv_line(p[0],"DEBUG",bug=bug,entry="hypothesis",value=hypotheses[0],status="rejected")),(p[1],kv_line(p[1],"DEBUG",bug=bug,entry="hypothesis",value=hypotheses[1],status="confirmed")),(p[2],kv_line(p[2],"DEBUG",bug=bug,entry="patch",value=patch,status="applied")),(p[3],kv_line(p[3],"DEBUG",bug=bug,entry="verification",value=patch,status="passed"))]
    history=insert_events(rng,horizon,events); expected=f"{hypotheses[1]} -> {patch}"; q,level=wording(instance,f"Recover confirmed cause and verified patch for `{bug}` as `<cause> -> <patch>`.",f"What finally fixed `{bug}`?",f"After context reset, recover what fixed `{bug}`.")
    code=_filter_code(history_path,"DEBUG",[("bug",bug)],6); messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},assistant_tool("pem-debug-1",code),tool_result("pem-debug-1",repr([x[1] for x in events])),{"role":"assistant","content":expected}]
    return Example(messages,{history_path:history},meta(split=split,family="software_debug_resolution",domain="software",instance=instance,horizon=horizon,history_path=history_path,history_format="kv",instruction_level=level,requires_history=True,uses_ipython=True,retrieval_policy="confirmed_cause_plus_verified_fix",expected_answer=expected))


def research_retraction_example(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); claim=f"H{rng.randint(10,99)}"; s1,s2,s3=[f"P{rng.randint(100,999)}" for _ in range(3)]; p=[horizon//5,horizon//2,horizon*3//4,horizon-4]
    events=[(p[0],kv_line(p[0],"PAPER",claim=claim,source=s1,verdict="supports",state="active")),(p[1],kv_line(p[1],"PAPER",claim=claim,source=s2,verdict="supports",state="active")),(p[2],kv_line(p[2],"RETRACTION",claim=claim,source=s2,state="retracted")),(p[3],kv_line(p[3],"PAPER",claim=claim,source=s3,verdict="contradicts",state="active"))]
    history=insert_events(rng,horizon,events); expected=f"contradicted by {s3}; {s2} retracted"; q,level=wording(instance,f"Reconstruct current research status for `{claim}` accounting for retractions.",f"What is current evidence status for `{claim}`?",f"After compaction, recover current evidence status for `{claim}`.")
    code=f"from pathlib import Path\nclaim={claim!r}\nrows=[x.strip() for x in Path({history_path!r}).open() if f'claim=\\\"{{claim}}\\\"' in x]\nrows[-8:]"; messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},assistant_tool("pem-retract-1",code),tool_result("pem-retract-1",repr([x[1] for x in events])),{"role":"assistant","content":expected}]
    return Example(messages,{history_path:history},meta(split=split,family="research_retraction",domain="research",instance=instance,horizon=horizon,history_path=history_path,history_format="kv",instruction_level=level,requires_history=True,uses_ipython=True,retrieval_policy="evidence_with_retraction",expected_answer=expected))


def dataset_provenance_example(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); metric=rng.choice(["f1","accuracy","auc","loss"]); dataset=f"ds-{rng.randint(10,99)}"; version=f"v{rng.randint(2,9)}"; value=round(rng.uniform(.7,.95),3); run=f"run-{rng.randint(300,999)}"; p=[horizon//5,horizon//2,horizon-5]
    events=[(p[0],kv_line(p[0],"DATASET",dataset=dataset,version=version,state="frozen")),(p[1],kv_line(p[1],"EVAL_RUN",run=run,dataset=dataset,version=version,metric=metric,value=value,state="valid")),(p[2],kv_line(p[2],"EVAL_NOTE",run=run,status="accepted"))]
    history=insert_events(rng,horizon,events); expected=f"{dataset}@{version}"; q,level=wording(instance,f"Which exact dataset version produced accepted `{metric}` for `{run}`?",f"What dataset snapshot underlies `{run}`?",f"After compaction, recover provenance for `{run}`.")
    code=_filter_code(history_path,"EVAL_RUN",[("run",run),("state","valid")],4); messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},assistant_tool("pem-data-prov-1",code),tool_result("pem-data-prov-1",repr([events[1][1]])),{"role":"assistant","content":expected}]
    return Example(messages,{history_path:history},meta(split=split,family="dataset_provenance",domain="data",instance=instance,horizon=horizon,history_path=history_path,history_format="kv",instruction_level=level,requires_history=True,uses_ipython=True,retrieval_policy="result_to_dataset_provenance",expected_answer=expected))


def approval_revocation_example(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); resource=rng.choice(["deploy","publish","merge","release"]); actor=rng.choice(["alice","bob","carol"]); p=[horizon//5,horizon//2,horizon-5]
    events=[(p[0],kv_line(p[0],"APPROVAL",resource=resource,actor=actor,state="granted")),(p[1],kv_line(p[1],"APPROVAL",resource=resource,actor=actor,state="revoked")),(p[2],kv_line(p[2],"APPROVAL",resource=resource,actor=actor,state="granted"))]
    history=insert_events(rng,horizon,events); expected="granted"; q,level=wording(instance,f"Recover current approval state for `{actor}` on `{resource}`.",f"May `{actor}` currently `{resource}`? Return granted or revoked.",f"After reset, is `{actor}` currently approved to `{resource}`?")
    code=_filter_code(history_path,"APPROVAL",[("resource",resource),("actor",actor)],4); messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},assistant_tool("pem-approval-1",code),tool_result("pem-approval-1",repr([x[1] for x in events])),{"role":"assistant","content":expected}]
    return Example(messages,{history_path:history},meta(split=split,family="approval_revocation",domain="project",instance=instance,horizon=horizon,history_path=history_path,history_format="kv",instruction_level=level,requires_history=True,uses_ipython=True,retrieval_policy="grant_revoke_grant_state",expected_answer=expected))


def child_result_verification_example(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); task=f"subtask-{rng.randint(10,99)}"; a,b=rng.randint(20,60),rng.randint(20,60); true=a+b; bad=true+rng.choice([-3,-2,2,4]); p=[horizon//5,horizon//2,horizon*3//4,horizon-4]
    events=[(p[0],kv_line(p[0],"CHILD_RESULT",task=task,child="A",value=bad,status="reported")),(p[1],kv_line(p[1],"CHALLENGE",task=task,target="A",reason="mismatch")),(p[2],kv_line(p[2],"CHILD_RESULT",task=task,child="B",value=true,status="reported")),(p[3],kv_line(p[3],"VERIFICATION",task=task,value=true,status="accepted"))]
    history=insert_events(rng,horizon,events); expected=str(true); q,level=wording(instance,f"Recover verified accepted result for `{task}` after child disagreement.",f"What result should we trust for `{task}`?",f"After reset, recover trusted result for `{task}`.")
    code=f"from pathlib import Path\ntask={task!r}\nrows=[x.strip() for x in Path({history_path!r}).open() if f'task=\\\"{{task}}\\\"' in x]\nrows[-8:]"; messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},assistant_tool("pem-child-verify-1",code),tool_result("pem-child-verify-1",repr([x[1] for x in events])),{"role":"assistant","content":expected}]
    return Example(messages,{history_path:history},meta(split=split,family="child_result_verification",domain="agents",instance=instance,horizon=horizon,history_path=history_path,history_format="kv",instruction_level=level,requires_history=True,uses_ipython=True,retrieval_policy="challenge_then_verified_result",expected_answer=expected))


def ownership_reclaim_example(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); work=f"artifact-{rng.randint(100,999)}"; p=[horizon//5,horizon//2,horizon-5]
    events=[(p[0],kv_line(p[0],"OWNERSHIP",work=work,owner="child-A",state="active")),(p[1],kv_line(p[1],"CHILD_FAILURE",work=work,owner="child-A",state="failed")),(p[2],kv_line(p[2],"OWNERSHIP",work=work,owner="parent",state="reclaimed"))]
    history=insert_events(rng,horizon,events); expected="parent"; q,level=wording(instance,f"Who currently owns `{work}` after failure/reclaim?",f"Who owns `{work}` now?",f"Resume after compaction: who currently owns `{work}`?")
    code=f"from pathlib import Path\nwork={work!r}\nrows=[x.strip() for x in Path({history_path!r}).open() if f'work=\\\"{{work}}\\\"' in x]\nrows[-6:]"; messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},assistant_tool("pem-own-1",code),tool_result("pem-own-1",repr([x[1] for x in events])),{"role":"assistant","content":expected}]
    return Example(messages,{history_path:history},meta(split=split,family="ownership_reclaim",domain="agents",instance=instance,horizon=horizon,history_path=history_path,history_format="kv",instruction_level=level,requires_history=True,uses_ipython=True,retrieval_policy="phase_aware_ownership",expected_answer=expected))


def experiment_best_valid_checkpoint_example(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); run=f"exp-{rng.randint(100,999)}"; steps=[2,4,6,8]; scores=[round(rng.uniform(.4,.7),3),round(rng.uniform(.75,.9),3),round(rng.uniform(.9,.98),3),round(rng.uniform(.7,.85),3)]; states=["valid","valid","invalid","valid"]; p=[horizon//6,horizon//3,horizon*2//3,horizon-5]
    events=[(p[i],kv_line(p[i],"CHECKPOINT_SCORE",run=run,step=steps[i],score=scores[i],state=states[i])) for i in range(4)]; history=insert_events(rng,horizon,events); valid=[(s,sc) for s,sc,st in zip(steps,scores,states) if st=="valid"]; best=max(valid,key=lambda x:x[1])[0]
    q,level=wording(instance,f"Which valid checkpoint of `{run}` has highest score? Return step.",f"Which `{run}` checkpoint should we select by valid score?",f"After compaction, recover best valid checkpoint for `{run}`.")
    code=f"from pathlib import Path\nimport re\nrun={run!r}\nrows=[x.strip() for x in Path({history_path!r}).open() if 'TYPE=CHECKPOINT_SCORE' in x and f'run=\\\"{{run}}\\\"' in x and 'state=\\\"valid\\\"' in x]\n[(int(re.search(r'step=(\\d+)',x).group(1)),float(re.search(r'score=([0-9.]+)',x).group(1))) for x in rows]"; valid_lines=[x[1] for x in events if 'state="valid"' in x[1]]
    messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},assistant_tool("pem-best-1",code),tool_result("pem-best-1",repr(valid_lines)),{"role":"assistant","content":str(best)}]
    return Example(messages,{history_path:history},meta(split=split,family="experiment_best_valid_checkpoint",domain="experiments",instance=instance,horizon=horizon,history_path=history_path,history_format="kv",instruction_level=level,requires_history=True,uses_ipython=True,retrieval_policy="argmax_over_valid_history",expected_answer=str(best)))


def dependency_next_action_example(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); project=f"proj-{rng.randint(10,99)}"; tasks=["collect","transform","validate","publish"]; p=[horizon//7,horizon//3,horizon//2,horizon-5]; states=["done","done","blocked","pending"]
    events=[(p[i],kv_line(p[i],"TASK",project=project,task=tasks[i],state=states[i])) for i in range(4)]; history=insert_events(rng,horizon,events); expected="validate"; q,level=wording(instance,f"For `{project}`, return first task not done in dependency order.",f"What should we unblock next for `{project}`?",f"After reset, recover next unfinished dependency for `{project}`.")
    code=_filter_code(history_path,"TASK",[("project",project)],6); messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},assistant_tool("pem-dep-1",code),tool_result("pem-dep-1",repr([x[1] for x in events])),{"role":"assistant","content":expected}]
    return Example(messages,{history_path:history},meta(split=split,family="dependency_next_action",domain="project",instance=instance,horizon=horizon,history_path=history_path,history_format="kv",instruction_level=level,requires_history=True,uses_ipython=True,retrieval_policy="ordered_dependency_reconstruction",expected_answer=expected))


def config_snapshot_reconstruction_example(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); service=rng.choice(["api","trainer","worker","router"]); final={"batch":str(rng.choice([8,16,32])),"mode":rng.choice(["safe","fast"]),"region":rng.choice(["eu","us","ap"])}; p=[horizon//6,horizon//3,horizon//2,horizon-5]
    events=[(p[0],kv_line(p[0],"CONFIG",service=service,key="batch",value="4",state="active")),(p[1],kv_line(p[1],"CONFIG",service=service,key="mode",value=final["mode"],state="active")),(p[2],kv_line(p[2],"CONFIG",service=service,key="batch",value=final["batch"],state="active")),(p[3],kv_line(p[3],"CONFIG",service=service,key="region",value=final["region"],state="active"))]
    history=insert_events(rng,horizon,events); expected=f"batch={final['batch']},mode={final['mode']},region={final['region']}"; q,level=wording(instance,f"Reconstruct current config for `{service}` as batch/mode/region using latest per key.",f"What is current `{service}` config?",f"After compaction, reconstruct current `{service}` config.")
    code=_filter_code(history_path,"CONFIG",[("service",service),("state","active")],8); messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},assistant_tool("pem-config-1",code),tool_result("pem-config-1",repr([x[1] for x in events])),{"role":"assistant","content":expected}]
    return Example(messages,{history_path:history},meta(split=split,family="config_snapshot_reconstruction",domain="ops",instance=instance,horizon=horizon,history_path=history_path,history_format="kv",instruction_level=level,requires_history=True,uses_ipython=True,retrieval_policy="latest_value_per_key",expected_answer=expected))


def unresolved_todo_example(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); objective=f"goal-{rng.randint(10,99)}"; todos=["draft","review","revise","ship"]; states=["done","cancelled","pending","pending"]; p=[horizon//7,horizon//3,horizon//2,horizon-5]
    events=[(p[i],kv_line(p[i],"TODO",objective=objective,item=todos[i],state=states[i],order=i)) for i in range(4)]; history=insert_events(rng,horizon,events); expected="revise"; q,level=wording(instance,f"Return earliest pending item for `{objective}`; ignore done/cancelled.",f"What is next live todo for `{objective}`?",f"After reset, recover next pending item for `{objective}`.")
    code=_filter_code(history_path,"TODO",[("objective",objective)],8); messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},assistant_tool("pem-todo-1",code),tool_result("pem-todo-1",repr([x[1] for x in events])),{"role":"assistant","content":expected}]
    return Example(messages,{history_path:history},meta(split=split,family="unresolved_todo",domain="project",instance=instance,horizon=horizon,history_path=history_path,history_format="kv",instruction_level=level,requires_history=True,uses_ipython=True,retrieval_policy="first_pending_after_filter",expected_answer=expected))


def incident_resolution_example(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); incident=f"INC-{rng.randint(100,999)}"; cause=rng.choice(["dns","quota","schema","expired-token"]); recovery=rng.choice(["rollback","restart","reconfigure"]); p=[horizon//5,horizon//2,horizon*3//4,horizon-5]
    events=[(p[0],kv_line(p[0],"INCIDENT",incident=incident,event="alert",value="firing")),(p[1],kv_line(p[1],"INCIDENT",incident=incident,event="root_cause",value=cause,status="confirmed")),(p[2],kv_line(p[2],"INCIDENT",incident=incident,event="recovery",value=recovery,status="applied")),(p[3],kv_line(p[3],"INCIDENT",incident=incident,event="health",value="green",status="verified"))]
    history=insert_events(rng,horizon,events); expected=f"{cause} -> {recovery}"; q,level=wording(instance,f"Recover confirmed cause and applied recovery for `{incident}`.",f"What caused and resolved `{incident}`?",f"After compaction, recover cause and recovery for `{incident}`.")
    code=_filter_code(history_path,"INCIDENT",[("incident",incident)],8); messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},assistant_tool("pem-incident-1",code),tool_result("pem-incident-1",repr([x[1] for x in events])),{"role":"assistant","content":expected}]
    return Example(messages,{history_path:history},meta(split=split,family="incident_resolution",domain="ops",instance=instance,horizon=horizon,history_path=history_path,history_format="kv",instruction_level=level,requires_history=True,uses_ipython=True,retrieval_policy="cause_and_recovery_chain",expected_answer=expected))


def direct_ipython_control_example(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); nums=[rng.randint(10,80) for _ in range(rng.randint(5,9))]; history=insert_events(rng,horizon,[]); expected=str(sum(nums)); q=f"Values are fully in this request: {nums}. Return sum. Use computation if useful, but do not inspect prior history."; code=f"values={nums!r}\nsum(values)"
    messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},assistant_tool("pem-direct-py-1",code),tool_result("pem-direct-py-1",expected),{"role":"assistant","content":expected}]
    return Example(messages,{history_path:history},meta(split=split,family="direct_ipython_control",domain="control",instance=instance,horizon=horizon,history_path=history_path,history_format="kv",instruction_level="natural",requires_history=False,uses_ipython=True,retrieval_policy="compute_without_history",expected_answer=expected))


def prompt_override_control_example(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); key=rng.choice(["mode","region","format","priority"]); old=rng.choice(["alpha","beta","gamma"]); new=rng.choice(["delta","epsilon","zeta"]); p=horizon//2; history=insert_events(rng,horizon,[(p,kv_line(p,"STATE",key=key,value=old,status="active"))]); q=f"Authoritative update for this turn: `{key}` is now `{new}`. Return only value; do not consult older history."
    messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},{"role":"assistant","content":new}]
    return Example(messages,{history_path:history},meta(split=split,family="prompt_override_control",domain="control",instance=instance,horizon=horizon,history_path=history_path,history_format="kv",instruction_level="current_turn_authoritative",requires_history=False,uses_ipython=False,retrieval_policy="current_turn_over_prior_history",expected_answer=new))


TRAIN_FAMILIES: list[Callable[[random.Random, str, int, int], Example]] = [
    latest_state_example, accepted_requirement_example, successful_attempt_example,
    correction_aggregate_example, provenance_conflict_example, checkpoint_resume_example,
    stale_note_override_example, repeated_lookup_index_example, multi_key_join_example,
    context_reset_resume_example, constraint_update_example, direct_control_example,
    software_debug_resolution_example, research_retraction_example, dataset_provenance_example,
    approval_revocation_example, child_result_verification_example, ownership_reclaim_example,
    experiment_best_valid_checkpoint_example, dependency_next_action_example,
    config_snapshot_reconstruction_example, unresolved_todo_example, incident_resolution_example,
    direct_ipython_control_example, prompt_override_control_example,
]


def jsonl_history(events: list[dict]) -> str:
    return "\n".join(json.dumps(e, separators=(",", ":"), ensure_ascii=False) for e in events) + "\n"


def ood_jsonl_latest_revision(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); topic=rng.choice(["schema","policy","experiment","contract"]); revs=[rng.randint(1,3),rng.randint(4,6),rng.randint(7,9)]; pos=[horizon//5,horizon//2,horizon-3]; events=[]
    for i in range(1,horizon+1):
        if i in pos:
            j=pos.index(i); events.append({"seq":i,"kind":"revision","topic":topic,"revision":revs[j],"state":"adopted"})
        else: events.append({"seq":i,"kind":"noise","topic":rng.choice(["a","b","c"]),"value":rng.randint(0,999)})
    history=jsonl_history(events); expected=str(revs[-1]); q=f"Historical log is JSONL. What is latest adopted revision for `{topic}`? Return revision."; code=f"from pathlib import Path\nimport json\ntopic={topic!r}\nrows=(json.loads(x) for x in Path({history_path!r}).open())\nmatches=[r for r in rows if r.get('kind')=='revision' and r.get('topic')==topic and r.get('state')=='adopted']\nmatches[-1]"; out=repr({"seq":pos[-1],"kind":"revision","topic":topic,"revision":revs[-1],"state":"adopted"})
    messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},assistant_tool("pem-ood-json-1",code),tool_result("pem-ood-json-1",out),{"role":"assistant","content":expected}]
    return Example(messages,{history_path:history},meta(split=split,family="ood_jsonl_latest_revision",domain="ood",instance=instance,horizon=horizon,history_path=history_path,history_format="jsonl",instruction_level="natural_ood",requires_history=True,uses_ipython=True,retrieval_policy="format_generalization_latest",expected_answer=expected))


def ood_temporal_window(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); tag=rng.choice(["latency","accuracy","throughput"]); events=[]; selected=[]; stride=max(11,horizon//12)
    for i in range(1,horizon+1):
        if i%stride==0:
            e={"seq":i,"kind":"measurement","tag":tag,"value":rng.randint(10,99)}; selected.append(e)
        else: e={"seq":i,"kind":"noise","tag":rng.choice(["x","y","z"]),"value":rng.randint(1,99)}
        events.append(e)
    tail=selected[-3:]; expected=str(round(sum(e["value"] for e in tail)/3,2)); history=jsonl_history(events); q=f"Compute mean of last three `{tag}` measurements. Return number."; code=f"from pathlib import Path\nimport json\ntag={tag!r}\nvals=[r['value'] for r in (json.loads(x) for x in Path({history_path!r}).open()) if r.get('kind')=='measurement' and r.get('tag')==tag]\nround(sum(vals[-3:])/3,2)"
    messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},assistant_tool("pem-ood-win-1",code),tool_result("pem-ood-win-1",expected),{"role":"assistant","content":expected}]
    return Example(messages,{history_path:history},meta(split=split,family="ood_temporal_window",domain="ood",instance=instance,horizon=horizon,history_path=history_path,history_format="jsonl",instruction_level="natural_ood",requires_history=True,uses_ipython=True,retrieval_policy="temporal_window_compute",expected_answer=expected))


def ood_supersession_chain(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); obj=f"obj-{rng.randint(100,999)}"; pos=[horizon//5,horizon//2,horizon-4]; vals=[f"state-{rng.randint(10,29)}",f"state-{rng.randint(30,59)}",f"state-{rng.randint(60,99)}"]; events=[]
    for i in range(1,horizon+1):
        if i in pos:
            j=pos.index(i); events.append({"seq":i,"kind":"transition","object":obj,"value":vals[j],"supersedes":None if j==0 else vals[j-1]})
        else: events.append({"seq":i,"kind":"noise","object":f"obj-{rng.randint(1,9)}","value":rng.randint(0,99)})
    history=jsonl_history(events); q=f"Follow supersession chain for `{obj}` and return current value."; code=f"from pathlib import Path\nimport json\nobj={obj!r}\nchain=[json.loads(x) for x in Path({history_path!r}).open()]\nchain=[e for e in chain if e.get('kind')=='transition' and e.get('object')==obj]\nchain"; out=repr([e for e in events if e.get('kind')=='transition' and e.get('object')==obj])
    messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},assistant_tool("pem-ood-chain-1",code),tool_result("pem-ood-chain-1",out),{"role":"assistant","content":vals[-1]}]
    return Example(messages,{history_path:history},meta(split=split,family="ood_supersession_chain",domain="ood",instance=instance,horizon=horizon,history_path=history_path,history_format="jsonl",instruction_level="natural_ood",requires_history=True,uses_ipython=True,retrieval_policy="supersession_chain",expected_answer=vals[-1]))


def ood_markdown_decision_log(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); key=rng.choice(["architecture","database","runtime"]); vals=["option-A","option-B","option-C"]; pos=[horizon//5,horizon//2,horizon-4]; lines=[]
    for i in range(1,horizon+1):
        if i in pos:
            j=pos.index(i); lines.append(f"### Event {i}\n- kind: decision\n- key: {key}\n- value: {vals[j]}\n- status: accepted\n")
        else: lines.append(f"### Event {i}\n- kind: note\n- topic: noise-{rng.randint(1,9)}\n- value: {rng.randint(1,999)}\n")
    history="\n".join(lines); expected=vals[-1]; q=f"History is Markdown. What is latest accepted `{key}`? Return value."; code=f"from pathlib import Path\ntext=Path({history_path!r}).read_text()\nkey={key!r}\nblocks=text.split('### Event ')[1:]\nmatches=[b for b in blocks if '- kind: decision' in b and f'- key: {{key}}' in b and '- status: accepted' in b]\nmatches[-1]"
    messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},assistant_tool("pem-ood-md-1",code),tool_result("pem-ood-md-1",repr(lines[-5:])),{"role":"assistant","content":expected}]
    return Example(messages,{history_path:history},meta(split=split,family="ood_markdown_decision_log",domain="ood",instance=instance,horizon=horizon,history_path=history_path,history_format="markdown",instruction_level="natural_ood",requires_history=True,uses_ipython=True,retrieval_policy="markdown_format_generalization",expected_answer=expected))


def ood_csv_best_valid_run(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); rows=[]; best_id=None; best_score=-1
    for i in range(1,horizon+1):
        if i%max(13,horizon//10)==0:
            rid=f"r{i}"; score=round(rng.uniform(.5,.99),3); valid=rng.random()>.25; rows.append({"seq":i,"kind":"result","run":rid,"score":score,"valid":str(valid).lower()})
            if valid and score>best_score: best_id,best_score=rid,score
        else: rows.append({"seq":i,"kind":"noise","run":f"n{i}","score":round(rng.random(),3),"valid":"false"})
    sio=io.StringIO(); w=csv.DictWriter(sio,fieldnames=["seq","kind","run","score","valid"]); w.writeheader(); w.writerows(rows); history=sio.getvalue(); expected=str(best_id); q="History is CSV. Which valid result has highest score? Return run id."; code=f"from pathlib import Path\nimport csv\nrows=list(csv.DictReader(Path({history_path!r}).open()))\nvalid=[r for r in rows if r['kind']=='result' and r['valid']=='true']\nmax(valid,key=lambda r:float(r['score']))"
    messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},assistant_tool("pem-ood-csv-1",code),tool_result("pem-ood-csv-1",repr({"run":best_id,"score":best_score})),{"role":"assistant","content":expected}]
    return Example(messages,{history_path:history},meta(split=split,family="ood_csv_best_valid_run",domain="ood",instance=instance,horizon=horizon,history_path=history_path,history_format="csv",instruction_level="natural_ood",requires_history=True,uses_ipython=True,retrieval_policy="csv_argmax_valid",expected_answer=expected))


def ood_prose_retraction(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); claim=f"claim-{rng.randint(10,99)}"; src=f"paper-{rng.randint(100,999)}"; replacement=f"paper-{rng.randint(1000,1999)}"; pos=[horizon//4,horizon//2,horizon-5]; lines=[]
    for i in range(1,horizon+1):
        if i==pos[0]: lines.append(f"{i}. Evidence note: {src} supports {claim}.")
        elif i==pos[1]: lines.append(f"{i}. Correction: {src} was retracted and must not support {claim}.")
        elif i==pos[2]: lines.append(f"{i}. Evidence note: {replacement} contradicts {claim}.")
        else: lines.append(f"{i}. Routine note about item {rng.randint(1,999)}.")
    history="\n".join(lines)+"\n"; expected=f"contradicted by {replacement}"; q=f"Read prose journal and give current evidential status for `{claim}`."; code=f"from pathlib import Path\nclaim={claim!r}\nrows=[x.strip() for x in Path({history_path!r}).open() if claim in x]\nrows"
    messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},assistant_tool("pem-ood-prose-1",code),tool_result("pem-ood-prose-1",repr([lines[p-1] for p in pos])),{"role":"assistant","content":expected}]
    return Example(messages,{history_path:history},meta(split=split,family="ood_prose_retraction",domain="ood",instance=instance,horizon=horizon,history_path=history_path,history_format="prose",instruction_level="natural_ood",requires_history=True,uses_ipython=True,retrieval_policy="prose_retraction_resolution",expected_answer=expected))


def ood_nested_json_config(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); service=rng.choice(["router","worker","api"]); final={"threads":rng.choice([8,12,16]),"zone":rng.choice(["eu","us","ap"])}; events=[]
    for i in range(1,horizon+1):
        if i==horizon//3: events.append({"seq":i,"kind":"config","service":service,"changes":{"threads":4}})
        elif i==horizon//2: events.append({"seq":i,"kind":"config","service":service,"changes":{"zone":final["zone"]}})
        elif i==horizon-4: events.append({"seq":i,"kind":"config","service":service,"changes":{"threads":final["threads"]}})
        else: events.append({"seq":i,"kind":"noise","payload":{"x":rng.randint(1,999)}})
    history=jsonl_history(events); expected=f"threads={final['threads']},zone={final['zone']}"; q=f"Reconstruct current nested config for `{service}` as `threads=...,zone=...`."; code=f"from pathlib import Path\nimport json\nservice={service!r}\nstate={{}}\nfor row in (json.loads(x) for x in Path({history_path!r}).open()):\n    if row.get('kind')=='config' and row.get('service')==service: state.update(row['changes'])\nstate"
    messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},assistant_tool("pem-ood-nested-1",code),tool_result("pem-ood-nested-1",repr(final)),{"role":"assistant","content":expected}]
    return Example(messages,{history_path:history},meta(split=split,family="ood_nested_json_config",domain="ood",instance=instance,horizon=horizon,history_path=history_path,history_format="jsonl_nested",instruction_level="natural_ood",requires_history=True,uses_ipython=True,retrieval_policy="nested_patch_fold",expected_answer=expected))


def ood_mixed_tool_transcript(rng, split, instance, horizon):
    history_path=history_path_for(rng,instance); job=f"job-{rng.randint(10,99)}"; result=rng.randint(100,999); pos=[horizon//3,horizon//2,horizon-4]; lines=[]
    for i in range(1,horizon+1):
        if i==pos[0]: lines.append(f"[{i}] assistant tool_call name=compute job={job}")
        elif i==pos[1]: lines.append(f"[{i}] tool result job={job} value={result} status=tentative")
        elif i==pos[2]: lines.append(f"[{i}] verifier job={job} value={result} status=accepted")
        else: lines.append(f"[{i}] message noise={rng.randint(1,99999)}")
    history="\n".join(lines)+"\n"; expected=str(result); q=f"From mixed agent/tool transcript, return accepted verified value for `{job}`."; code=f"from pathlib import Path\njob={job!r}\nrows=[x.strip() for x in Path({history_path!r}).open() if job in x]\nrows"
    messages=[{"role":"system","content":system_prompt(history_path)},{"role":"user","content":q},assistant_tool("pem-ood-tool-1",code),tool_result("pem-ood-tool-1",repr([lines[p-1] for p in pos])),{"role":"assistant","content":expected}]
    return Example(messages,{history_path:history},meta(split=split,family="ood_mixed_tool_transcript",domain="ood",instance=instance,horizon=horizon,history_path=history_path,history_format="mixed_transcript",instruction_level="natural_ood",requires_history=True,uses_ipython=True,retrieval_policy="tool_transcript_verification",expected_answer=expected))


OOD_FAMILIES = [ood_jsonl_latest_revision,ood_temporal_window,ood_supersession_chain,ood_markdown_decision_log,ood_csv_best_valid_run,ood_prose_retraction,ood_nested_json_config,ood_mixed_tool_transcript]


def validate_row(row: dict) -> None:
    messages=json.loads(row["messages_json"]); tools=json.loads(row["tools"]); files=json.loads(row["workspace_files_json"]); m=json.loads(row["metadata_json"])
    assert tools == [IPYTHON_TOOL]; hp=m["history_path"]; assert hp in files; assert all("reasoning_content" not in msg for msg in messages)
    calls=[]
    for msg in messages:
        for call in msg.get("tool_calls",[]) or []:
            args=json.loads(call["function"]["arguments"]); code=args["code"]; compile(code,"<tool>","exec"); calls.append((call["id"],code))
    result_ids={msg.get("tool_call_id") for msg in messages if msg.get("role")=="tool"}; assert all(cid in result_ids for cid,_ in calls)
    if m["requires_history"]:
        assert calls and any(hp in code for _,code in calls), m
    else:
        assert all(hp not in code for _,code in calls), m
        assert bool(calls) == bool(m["uses_ipython"]), m
    if m["family"]=="repeated_lookup_index": assert hp in calls[0][1] and hp not in calls[1][1]
    history=files[hp]; assert len(history)>800, m; assert history not in json.dumps(messages,ensure_ascii=False)
    final=next(msg["content"] for msg in reversed(messages) if msg.get("role")=="assistant" and msg.get("content") is not None); assert str(final)==str(m["expected_answer"]), (final,m)


def input_fingerprint(row: dict) -> str:
    payload = {
        "messages": json.loads(row["messages_json"]),
        "tools": json.loads(row["tools"]),
        "workspace_files": json.loads(row["workspace_files_json"]),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate_split_disjointness(splits: dict[str, list[dict]]) -> None:
    fingerprints = {
        split: {input_fingerprint(row) for row in rows}
        for split, rows in splits.items()
    }
    for split, rows in splits.items():
        assert len(fingerprints[split]) == len(rows), f"duplicate full input within {split}"
    names = sorted(splits)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = fingerprints[left] & fingerprints[right]
            assert not overlap, f"full-input overlap between {left} and {right}: {len(overlap)}"


def generate(seed: int, train_per_family: int, heldout_per_family: int, ood_per_family: int) -> dict[str,list[dict]]:
    root=random.Random(seed); splits={"train":[],"familiar_heldout":[],"semantic_ood":[]}; train_h=[64,128,256,512]; held_h=[96,192,384,768]; ood_h=[256,512,1024]
    for fi,family in enumerate(TRAIN_FAMILIES):
        for i in range(train_per_family):
            rng=random.Random(root.randrange(2**63)); splits["train"].append(base_row(family(rng,"train",i,train_h[(i+fi)%len(train_h)])))
        for i in range(heldout_per_family):
            rng=random.Random(root.randrange(2**63)); splits["familiar_heldout"].append(base_row(family(rng,"familiar_heldout",i,held_h[(i+fi)%len(held_h)])))
    for fi,family in enumerate(OOD_FAMILIES):
        for i in range(ood_per_family):
            rng=random.Random(root.randrange(2**63)); splits["semantic_ood"].append(base_row(family(rng,"semantic_ood",i,ood_h[(i+fi)%len(ood_h)])))
    for rows in splits.values():
        for row in rows: validate_row(row)
    validate_split_disjointness(splits)
    return splits


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_dataset(output_dir: Path, splits: dict[str,list[dict]], seed: int) -> dict:
    output_dir.mkdir(parents=True,exist_ok=True); files={}; family_counts={}; domain_counts={}
    for split,rows in splits.items():
        jp=output_dir/f"{split}.jsonl"; jp.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows)); files[jp.name]={"rows":len(rows),"sha256":sha256_file(jp)}
        if Dataset is not None:
            pp=output_dir/f"{split}.parquet"; Dataset.from_list(rows).to_parquet(str(pp)); files[pp.name]={"rows":len(rows),"sha256":sha256_file(pp)}
        fc={}; dc={}
        for row in rows:
            mm=json.loads(row["metadata_json"]); fc[mm["family"]]=fc.get(mm["family"],0)+1; dc[mm["domain"]]=dc.get(mm["domain"],0)+1
        family_counts[split]=dict(sorted(fc.items())); domain_counts[split]=dict(sorted(dc.items()))
    manifest={"schema_version":2,"dataset":DATASET_NAME,"seed":seed,"splits":{k:len(v) for k,v in splits.items()},"family_counts":family_counts,"domain_counts":domain_counts,"files":files,"reasoning_policy":"no fabricated reasoning_content","source_of_truth_policy":"append-only history is source of truth for prior events; current user turn may supersede prior state","design":{"train_families":len(TRAIN_FAMILIES),"semantic_ood_families":len(OOD_FAMILIES),"history_paths":list(HISTORY_PATHS),"train_horizon_events":"64-512","familiar_heldout_horizon_events":"96-768","semantic_ood_horizon_events":"256-1024","history_formats":"KV train/familiar; JSONL/Markdown/CSV/prose/nested JSON/mixed transcript OOD","negative_controls":"direct no-tool, direct IPython without history, current-turn authoritative override","persistent_state":"repeated lookup builds a derived index then reuses it without rereading history","domains":["software","research","data","project","agents","ops","experiments","control"],"promotion_note":"bootstrap supervision only; executable environment must verify retrieval need, grounding, stale/current resolution, reset recovery and efficiency"}}
    (output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n"); return manifest


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0]); parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--seed",type=int,default=DEFAULT_SEED); parser.add_argument("--train-per-family",type=int,default=48); parser.add_argument("--heldout-per-family",type=int,default=12); parser.add_argument("--ood-per-family",type=int,default=12); args=parser.parse_args()
    splits=generate(args.seed,args.train_per_family,args.heldout_per_family,args.ood_per_family); manifest=write_dataset(args.output_dir,splits,args.seed); print(json.dumps(manifest["splits"],sort_keys=True)); print(f"wrote {DATASET_NAME} to {args.output_dir}")


if __name__=="__main__": main()
