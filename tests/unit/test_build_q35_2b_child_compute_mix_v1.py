from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from datasets import Dataset


def _module():
    path = Path(__file__).parents[2] / "scripts" / "build_q35_2b_child_compute_mix_v1.py"
    spec = importlib.util.spec_from_file_location("child_compute_mix", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _corpus(module, path: Path, rows: list[dict]) -> None:
    path.mkdir()
    parquet = path / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    (path / "MANIFEST.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "dataset": {"sha256": module.sha256_file(parquet)},
            }
        ),
        encoding="utf-8",
    )


def test_verified_child_rows_are_interleaved_without_root(tmp_path: Path) -> None:
    module = _module()
    compute = tmp_path / "compute"
    replay = tmp_path / "replay"
    _corpus(
        module,
        compute,
        [
            {"role": "coordinator_nonroot", "trace_id": "c1"},
            {"role": "coordinator_root", "trace_id": "root"},
            {"role": "coordinator_nonroot", "trace_id": "c2"},
        ],
    )
    _corpus(module, replay, [{"role": "coordinator_nonroot", "trace_id": "r1"}])

    compute_rows, _ = module.verified_rows(compute, source="compute")
    replay_rows, _ = module.verified_rows(replay, source="replay")
    rows = module.interleave(compute_rows, replay_rows)

    assert [row["trace_id"] for row in rows] == ["c1", "r1", "c2"]
    assert all(row["role"] == "coordinator_nonroot" for row in rows)
