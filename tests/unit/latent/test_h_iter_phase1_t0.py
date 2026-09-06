from __future__ import annotations

import json
from pathlib import Path

import pytest

from prime_rl.latent.h_iter_phase1_t0 import (
    build_capture_schedule, build_memory_schedule, build_tamper_schedule,
    candidate_class, module_state_sha256,
)

ROOT=Path(__file__).resolve().parents[3]
MF0=ROOT/"experiments/qwen35-2b-latent-workspace-v1/h-iter-phase1-train-calibration-v1"
T0=ROOT/"experiments/qwen35-2b-latent-workspace-v1/h-iter-phase1-t0-train-calibration-v1"

def load(path:Path): return json.loads(path.read_text())

def test_materialized_schedules_regenerate() -> None:
    bank=load(ROOT/"experiments/qwen35-2b-latent-workspace-v1/h-iter-phase0-generator-locality-v1/train-bank.json")
    capture=load(T0/"t0-capture-schedule.json"); schedule=load(MF0/"training-schedule.json")
    assert capture==build_capture_schedule(bank)
    assert load(T0/"t0-memory-schedule.json")==build_memory_schedule(capture,schedule)
    assert load(T0/"t0-tamper-schedule.json")==build_tamper_schedule()

def test_candidate_contract_cpu() -> None:
    torch=pytest.importorskip("torch")
    Candidate=candidate_class(torch); torch.manual_seed(17594986156060532329); item=Candidate()
    assert len(item.state_dict())==16
    assert sum(p.numel() for p in item.parameters())==366340
    assert len(module_state_sha256(torch,dict(item.state_dict())))==64
