"""Export a verifiers v1 eval run (traces.jsonl) as an SFT dataset for `uv run sft`.

Reads a finished run's saved traces and reshapes them into the dataset shape the SFT trainer
consumes directly (see `prime_rl.trainer.sft.data`): a `messages_json` column (the OpenAI chat
wire messages, JSON-encoded) plus a `tools` column (the tools the model was shown, from
`Trace.tools`, also JSON-encoded). Encoding both avoids Arrow schema inference failures from
heterogeneous message content and tool schemas. One row per
branch: a linear rollout contributes one sample, a compacted/subagent rollout one per branch
(one training sample is built per branch).

Selection: generation-errored traces (`stop_condition == "error"`) always drop — a broken
transcript is not a sample. A scoring-only error keeps the generation outcome as its stop
condition and a complete conversation, so it stays; its reward may be partial/zero, which
`--min-reward` handles.

Usage (from the prime-rl repo):
    uv run python scripts/export_sft.py <run-dir> [--min-reward 1.0] [--drop-truncated]
                                        [-o OUT_DIR] [--push HF_REPO_ID]

Writes `<run-dir>/sft/train.parquet` by default — point the trainer at it with
`--data.name <run-dir>/sft`. Requires a verifiers release carrying `Trace.tools`
(PrimeIntellect-ai/verifiers#1963).
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset
from verifiers.v1 import Trace, WireTrace
from verifiers.v1.cli.output import read_episodes
from verifiers.v1.dialects.chat import message_to_wire


def sft_row(trace: Trace, branch) -> dict:
    """One branch in the Arrow-safe SFT wire format."""
    tools = json.dumps([t.model_dump(mode="json", exclude_none=True) for t in trace.tools or []])
    return {
        "messages_json": json.dumps(
            [message_to_wire(m) for m in branch.messages],
            ensure_ascii=False,
        ),
        "tools": tools,
    }


def sft_rows(trace: Trace) -> list[dict]:
    """A trace's SFT rows, one per non-empty message-graph branch."""
    return [sft_row(trace, branch) for branch in trace.branches if branch.messages]


def parse_metric_requirement(value: str) -> tuple[str, float]:
    """Parse NAME=MIN, where the saved metric must be at least MIN."""
    name, separator, minimum = value.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError("metric requirement must be NAME=MIN")
    try:
        return name, float(minimum)
    except ValueError as error:
        raise argparse.ArgumentTypeError("metric minimum must be numeric") from error


def keep(
    trace: Trace,
    min_reward: float | None,
    drop_truncated: bool,
    required_metrics: list[tuple[str, float]] | None = None,
) -> bool:
    """Whether a trace is worth training on (see module docstring for the error semantics)."""
    if trace.stop_condition == "error":
        return False
    if drop_truncated and trace.is_truncated:
        return False
    if min_reward is not None and trace.reward < min_reward:
        return False
    return all(
        isinstance(trace.metrics.get(name), (int, float)) and trace.metrics[name] >= minimum
        for name, minimum in required_metrics or []
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="the eval run dir (holds traces.jsonl)")
    parser.add_argument("--min-reward", type=float, default=None, help="keep traces with reward >= this")
    parser.add_argument("--drop-truncated", action="store_true", help="drop budget-cut traces")
    parser.add_argument(
        "--require-metric",
        action="append",
        default=[],
        type=parse_metric_requirement,
        metavar="NAME=MIN",
        help="keep traces whose saved metric is at least MIN; repeatable",
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=None, help="default: <run-dir>/sft")
    parser.add_argument("--push", default=None, help="HF repo id to push to instead of writing parquet")
    args = parser.parse_args()

    traces_path = args.run_dir / "traces.jsonl"
    if not traces_path.exists():
        raise SystemExit(f"no traces.jsonl in {args.run_dir}")

    episodes = read_episodes(args.run_dir, WireTrace)
    traces = [trace for episode in episodes for trace in episode.traces]
    rows = []
    for trace in traces:
        if keep(trace, args.min_reward, args.drop_truncated, args.require_metric):
            rows.extend(sft_rows(trace))
    print(f"export-sft: {len(episodes)} episode(s), {len(traces)} trace(s) -> {len(rows)} row(s)")
    if not rows:
        raise SystemExit("export-sft: no rows to export after selection")

    dataset = Dataset.from_list(rows)
    if args.push:
        dataset.push_to_hub(args.push)
        print(f"export-sft: pushed to {args.push}")
        return
    out = args.output_dir or args.run_dir / "sft"
    out.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(out / "train.parquet"))
    print(f"export-sft: wrote {out / 'train.parquet'} -> train with --data.name {out}")


if __name__ == "__main__":
    main()
