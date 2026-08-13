"""Derive the first iterative full-weight harness-mastery consolidation run."""

from __future__ import annotations

import argparse
import copy
import tomllib
from pathlib import Path

import tomli_w

from prime_rl.configs.rl import RLConfig
from prime_rl.utils.config import to_toml_dict

BASE_REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"
RUN_SEED = 20260817


def _source(sources: list[dict], name: str) -> dict:
    return copy.deepcopy(next(source for source in sources if source["name"] == name))


def prepare(source_config: Path, output_config: Path, training_output: Path) -> RLConfig:
    with source_config.open("rb") as stream:
        raw = tomllib.load(stream)

    raw["max_steps"] = 8
    raw["output_dir"] = str(training_output)
    raw["clean_output_dir"] = True
    raw["model"]["name"] = "Qwen/Qwen3.5-27B"
    raw["model"]["revision"] = BASE_REVISION

    raw["deployment"].update(
        gpus_per_node=8,
        num_train_gpus=6,
        num_infer_gpus=2,
    )
    raw["trainer"]["model"].pop("lora", None)
    raw["trainer"]["ckpt"]["weights"]["save_adapter_separately"] = False
    raw["trainer"]["optim"].update(lr=5e-7, max_norm=1.0)
    raw["ckpt"].update(interval=2, keep_last=4, keep_interval=2)

    orchestrator = raw["orchestrator"]
    orchestrator.update(
        batch_size=24,
        oversampling_factor=None,
        max_inflight_episodes=8,
        max_off_policy_steps=0,
    )
    # During acquisition, task correctness and causal control must outrank brevity.
    orchestrator["algo"].pop("length_penalty", None)
    orchestrator.pop("eval", None)

    original_sources = orchestrator["train"]["source"]
    foundations = _source(original_sources, "mastery-foundations-train")
    foundations["ratio"] = 1.0
    foundations["env"]["taskset"].update(
        families=["ipython_cell", "persistence", "subagent_lifecycle", "harness_state"],
        instances_per_family=24,
        instance_offset=1000,
    )

    child = _source(original_sources, "mastery-ownership-child-train")
    child["ratio"] = 1.0
    child["env"]["taskset"].update(
        instances_per_family=24,
        instance_offset=1000,
        seed=RUN_SEED,
    )

    coordinator = _source(original_sources, "mastery-ownership-coordinator-train")
    coordinator["ratio"] = 1.0
    coordinator["env"]["taskset"].update(
        instances_per_family=24,
        instance_offset=1000,
        seed=RUN_SEED,
    )

    communication = _source(original_sources, "mastery-communication-train")
    routing = copy.deepcopy(communication)
    routing["name"] = "mastery-routing-direct-single-train"
    routing["ratio"] = 2.0
    routing["env"]["taskset"].update(
        families=["direct", "single"],
        instances_per_template=24,
        instance_offset=30000,
        seed=RUN_SEED,
    )

    coupled = copy.deepcopy(communication)
    coupled["name"] = "mastery-coupled-communication-train"
    coupled["ratio"] = 3.0
    coupled["env"]["taskset"].update(
        families=["parallel", "followup", "handshake"],
        instances_per_template=24,
        instance_offset=30000,
        seed=RUN_SEED,
    )

    # Oolong currently yields no valid gradient. Reintroduce externalization only
    # after native-success or verified bootstrap trajectories exist.
    orchestrator["train"]["source"] = [foundations, child, coordinator, routing, coupled]

    inference = raw["inference"]["vllm"]
    inference.update(
        tensor_parallel_size=2,
        gpu_memory_utilization=0.80,
        max_num_seqs=4,
        enforce_eager=True,
    )

    config = RLConfig.model_validate(raw)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    with output_config.open("wb") as stream:
        tomli_w.dump(to_toml_dict(config), stream)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_config", type=Path)
    parser.add_argument("output_config", type=Path)
    parser.add_argument("training_output", type=Path)
    args = parser.parse_args()
    prepare(args.source_config, args.output_config, args.training_output)
    print(f"wrote {args.output_config}")


if __name__ == "__main__":
    main()
