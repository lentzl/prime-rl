"""Derive a standalone checkpoint qualification from the frozen teacher run."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from prime_rl.configs.sft import SFTConfig
from prime_rl.entrypoints.sft import write_eval_subconfigs


def prepare(
    source_config: Path,
    source_output: Path,
    output_dir: Path,
    *,
    oolong_examples: int = 8,
    max_model_len: int = 65536,
    data_parallel_size: int = 2,
) -> None:
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite qualification output: {output_dir}")
    if not (source_output / "weights" / "step_16" / "STABLE").is_file():
        raise SystemExit(f"source run has no stable final checkpoint: {source_output}")

    with source_config.open("rb") as stream:
        raw = tomllib.load(stream)

    raw["output_dir"] = str(output_dir)
    raw["clean_output_dir"] = False
    raw["ckpt"]["output_dir"] = str(source_output)
    raw["deployment"]["num_gpus"] = 0
    raw["deployment"]["num_infer_gpus"] = data_parallel_size * raw["inference"]["vllm"]["tensor_parallel_size"]
    raw["inference"]["vllm"]["max_model_len"] = max_model_len
    raw["inference"]["vllm"]["data_parallel_size"] = data_parallel_size
    raw["inference"]["vllm"]["data_parallel_size_local"] = data_parallel_size
    # Let the config validator allocate one API server per local DP replica.
    raw["inference"]["vllm"].pop("api_server_count", None)

    for source in raw["eval"]["source"]:
        if source.get("name") == "oolong-externalization":
            source["num_examples"] = oolong_examples
            break
    else:
        raise SystemExit("frozen config has no oolong-externalization source")

    config = SFTConfig.model_validate(raw)
    write_eval_subconfigs(config, output_dir / "configs")

    evaluator = output_dir / "configs" / "evaluator.toml"
    inference = output_dir / "configs" / "inference.toml"
    print(f"wrote {evaluator}")
    print(f"wrote {inference}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_config", type=Path)
    parser.add_argument("source_output", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--oolong-examples", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--data-parallel-size", type=int, default=2)
    args = parser.parse_args()
    prepare(
        args.source_config,
        args.source_output,
        args.output_dir,
        oolong_examples=args.oolong_examples,
        max_model_len=args.max_model_len,
        data_parallel_size=args.data_parallel_size,
    )


if __name__ == "__main__":
    main()
