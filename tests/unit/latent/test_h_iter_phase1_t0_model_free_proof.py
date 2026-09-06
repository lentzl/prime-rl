from __future__ import annotations

import json
from pathlib import Path

from prime_rl.latent.h_iter_phase1_t0 import PROOF_MEMORY_LABELS, SYNTHETIC_SEED, SYNTHETIC_SHA256, build_tamper_schedule

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
    plan={"archive_bindings":{"mf0":{},"cap0_r1":{}},"data_contract":{},"training_contract":{},"capture_contract":{},"metric_gate_contract":{},"remote_paths":{},"model_contract":{}}
    # Use a production-shaped plan fixture from the freezer, then exercise every literal mutation.
    from freeze_h_iter_phase1_t0_plan_v1 import plan_value
    plan=plan_value(ROOT,"0"*40)
    results=tamper_results(plan,contract,capture,schedule,tampers)
    assert len(results)==98 and all(row["rejected"] for row in results)
