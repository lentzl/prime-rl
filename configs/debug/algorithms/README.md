# Algorithm — Debug Configs

Minimal end-to-end configs for the algorithms against bundled verifiers envs, using `PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT` as the policy.

| Config | Algorithm | Frozen model | Notes |
|---|---|---|---|
| `grpo.toml` | `grpo` | none | |
| `max_rl.toml` | `max_rl` | none | GRPO with mean-normalized advantages (maximum-likelihood RL) |
| `opd.toml` | `opd` | local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | |
| `opd_lora.toml` | `opd` | local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | trains a LoRA adapter (rank 8) |
| `sft_distill.toml` | `sft` | local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | |
| `sft_distill_lora.toml` | `sft` | local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | trains a LoRA adapter (rank 8) |
| `sft_distill_external.toml` | `sft` | PI inference (`openai/gpt-5-mini`) | external OAI endpoint; no local server |
| `self_distill.toml` | `opsd` | none (`model = "policy"`) | SDFT against the live policy; demo from reverse-text's `answer` field |
| `opsd_multi_turn_smoke.toml` | `opsd` | none (`model = "policy"`) | one-step multi-turn smoke; demo from a tiny fixture env's `answer` field |
| `opsd_rlm_harness_smoke.toml` | `opsd` | none (`model = "policy"`) | RLM-shaped smoke with helper-use instruction, execution feedback, recovery, and final answer |
| `sdpo_huebotter_reference_smoke.toml` | `sdpo` | none (`model = "policy"`) | Hübotter-style SDPO smoke with sibling rollouts, student-selected top-k support rescored by the teacher, IS clipping, and token rollout correction |
| `sdpo_huebotter_reference_ema_smoke.toml` | `sdpo` | live EMA teacher (`model = "policy"`) | Reference-aligned EMA teacher smoke; auto-launches a second teacher inference server on a separate GPU |
| `echo.toml` | `echo` | none | multi-turn `alphabet-sort`; CE on observation tokens |
| `mixed_grpo_opd.toml` | `grpo` + `opd` (per env) | local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | two envs, one run; heterogeneous batches (with/without `ref_logprobs`) |

The policy inference server is auto-launched on GPU 0 at `http://localhost:8000/v1` with `gpu_memory_utilization=0.5`. The local frozen model (used by `opd*.toml`, `sft_distill.toml` / `sft_distill_lora.toml`, and `mixed_grpo_opd.toml`) is **not** auto-launched — start it manually on GPU 1.

Frozen models are declared inline on the algorithm — `[orchestrator.algo.teacher]` with `name` + `base_url` — and prime-rl never hosts them; only the trainable policy's server is managed by the `rl` entrypoint.

## Start the local frozen model

Needed for `opd*.toml`, `sft_distill.toml` / `sft_distill_lora.toml`, and `mixed_grpo_opd.toml`:

```bash
CUDA_VISIBLE_DEVICES=1 uv run inference \
  --model.name PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL \
  --server.port 8001 \
  --gpu-memory-utilization 0.5 \
  --model.enforce-eager
```

## Run the debug configs

```bash
# GRPO (no frozen model)
uv run rl @ configs/debug/algorithms/grpo.toml

# MaxRL (no frozen model)
uv run rl @ configs/debug/algorithms/max_rl.toml

# OPD (needs the frozen model on port 8001)
uv run rl @ configs/debug/algorithms/opd.toml
uv run rl @ configs/debug/algorithms/opd_lora.toml

# SFT distillation (needs the frozen model on port 8001)
uv run rl @ configs/debug/algorithms/sft_distill.toml
uv run rl @ configs/debug/algorithms/sft_distill_lora.toml

# SFT distillation from openai/gpt-5-mini via PI inference
# (requires PRIME_API_KEY + PRIME_TEAM_ID in env; no local frozen model needed)
uv run rl @ configs/debug/algorithms/sft_distill_external.toml

# Self-distillation against the live policy (no frozen model)
uv run rl @ configs/debug/algorithms/self_distill.toml
PYTHONPATH=tests/fixtures uv run rl @ configs/debug/algorithms/opsd_multi_turn_smoke.toml
PYTHONPATH=tests/fixtures uv run rl @ configs/debug/algorithms/opsd_rlm_harness_smoke.toml

# SDPO Hübotter-style smoke plus strict artifact verification.
scripts/run_sdpo_smoke_and_verify.sh --check-config
scripts/run_sdpo_smoke_and_verify.sh

# On local platforms that cannot use the Linux-only uv lockfile, prefer the
# Mac-friendly validation gate; it runs both live and EMA SDPO smoke config
# checks plus the combined CUDA acceptance config check through the controlled
# uvx runner.
scripts/run_sdpo_local_validation.sh

# SDPO with Hübotter-style EMA teacher regularization.
# Requires 3+ GPUs: policy inference, teacher inference, and trainer.
scripts/run_sdpo_smoke_and_verify.sh --ema --check-config
scripts/run_sdpo_smoke_and_verify.sh --ema

# Full SDPO CUDA acceptance: runs live-policy and EMA smokes into fixed
# subdirectories, verifies both with the strict wrapper contract, and writes
# sdpo_cuda_acceptance_summary.txt after both halves pass.
scripts/run_sdpo_cuda_acceptance.sh --check-config
scripts/run_sdpo_cuda_acceptance.sh --output-root outputs/sdpo-cuda-acceptance --clean-output-dir
scripts/run_sdpo_cuda_acceptance.sh --no-run --output-root outputs/sdpo-cuda-acceptance
scripts/run_sdpo_cuda_acceptance.sh --no-run --output-root outputs/sdpo-cuda-acceptance --archive outputs/sdpo-cuda-acceptance-proof.tar.gz

# Preferred remote proof flow: train, verify, and produce one tarball to download.
# See docs/sdpo-cuda-acceptance-runbook.md for fresh-box clone/submodule setup,
# monitoring, download, and local re-verification commands.
scripts/start_sdpo_cuda_acceptance_background.sh --preflight-only
scripts/start_sdpo_cuda_acceptance_background.sh
scripts/start_sdpo_cuda_acceptance_background.sh --status

# Attached-shell equivalent of the background run.
scripts/run_sdpo_cuda_acceptance.sh --output-root outputs/sdpo-cuda-acceptance --clean-output-dir --archive outputs/sdpo-cuda-acceptance-proof.tar.gz
uv run python scripts/verify_sdpo_cuda_acceptance_archive.py --expected-acceptance-mode training outputs/sdpo-cuda-acceptance-proof.tar.gz

# Reuse a fixed output directory, deleting old checkpoints first.
scripts/run_sdpo_smoke_and_verify.sh --output-dir outputs/sdpo-smoke --clean-output-dir

# The wrapper expands to the raw launch + verifier commands below.
uv run rl @ configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml
uv run rl @ configs/debug/algorithms/sdpo_huebotter_reference_ema_smoke.toml

# Verify stable SDPO schema-v2 token exports after the SDPO smoke.
# Accepts the run output dir, a token_exports dir, a step dir, or one rank_*.jsonl.
uv run python scripts/verify_sdpo_token_exports.py /path/to/sdpo-smoke-output --require-stable --require-student-preflight --require-importance-ratio-evidence --expected-topk 100 --rollout-is-threshold 2.0 --rollout-is token

# Verify the full SDPO smoke artifact contract. This always requires stable
# token exports; --require-provenance requires wrapper-written run provenance
# in or above the artifact path being verified;
# --require-ema-teacher also requires same-step sdpo_teacher broadcasts for
# post-initial matching-support export steps. The expected provenance flags
# additionally prevent mixing live and EMA outputs or stale smoke presets.
uv run python scripts/verify_sdpo_smoke_artifacts.py /path/to/sdpo-smoke-output --expected-topk 100 --require-provenance --expected-provenance-mode live --expected-provenance-config configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml
uv run python scripts/verify_sdpo_smoke_artifacts.py /path/to/sdpo-ema-smoke-output --require-ema-teacher --expected-topk 100 --require-provenance --expected-provenance-mode ema --expected-provenance-config configs/debug/algorithms/sdpo_huebotter_reference_ema_smoke.toml

# ECHO (no frozen model; multi-turn env)
uv run rl @ configs/debug/algorithms/echo.toml

# Mixed per-env algorithms: GRPO + OPD in one run (needs the frozen model on port 8001)
uv run rl @ configs/debug/algorithms/mixed_grpo_opd.toml
```

See [docs/algorithms.md](../../../docs/algorithms.md) for what each algorithm does and how to compose custom ones.

## SDPO smoke verification contract

`verify_sdpo_token_exports.py` is the lower-level schema/data sanity check. It proves
that stable schema-v2 SDPO token exports exist and that every final SDPO record
carries transported teacher top-k support at each weighted token row, alongside
trainer-forward student support, temperatures, sample ids, and at least one paired
row where both support streams land on the same weighted token positions.
Pass `--expected-topk` to require the artifact row width to match the run's
configured `trainer.sdpo_loss.distillation_topk`.
Pass `--require-importance-ratio-evidence` when checking a Hübotter-style run
that uses token-level rollout importance correction; this requires final
weighted SDPO rows to export finite `log_importance_ratio`, `importance_ratio`,
and `prob_delta` evidence.
Pass `--rollout-is-threshold` for strict smoke-style checks that should also
require final weighted SDPO rows to carry `sdpo_rollout_is_weights` bounded by
the configured truncation threshold. Pass `--rollout-is token` or
`--rollout-is sequence` to additionally require those weights to match the
selected truncated rollout-IS mode. The verifier reports those rows as
`rollout_is_weight_rows` / `rollout_is_weight_token_rows`.
Keep `trainer.sdpo_loss.rollout_is_batch_normalize = false` for the combined
`rl` SDPO presets. The lower-level loss primitive has reference coverage for
that knob, but the packed trainer currently evaluates SDPO one sequence at a
time, so launch configs reject batch-normalized rollout-IS until normalization
can be scoped over the global SDPO component batch.

`verify_sdpo_smoke_artifacts.py` is the acceptance check for the reference SDPO smoke
presets above. In addition to the stable export checks, it requires evidence for the
student-selected top-k path:

- `student_preflight_rows > 0`: after SDPO hindsight-target pruning, the trainer
  produced student support from records explicitly marked `preflight_only=true`
  before the teacher rescoring pass.
- `matching_support_rows > 0`: the final SDPO rows transported teacher logprobs on
  the exact student-selected support ids. Separately, every final weighted SDPO
  token row must carry non-placeholder transported teacher support.
- `matched_support_samples > 0`: the preflight and final exact-support SDPO
  evidence happened for the same `sample_id` in the same training step, with
  the same non-empty env name, token ids, position ids, loss mask, and
  temperatures.
- `matched_support_token_rows > 0`: the same guarantee holds at the weighted
  token-row level, not only at the sample level.
- `distinct_teacher_logprob_rows > 0`: at least one same-support final row has
  transported teacher logprobs that differ from trainer-forward student
  logprobs. This proves the smoke did not merely copy student scores into the
  teacher channel without making every row numerically-different, which would
  be brittle for the live-policy teacher preset.
- `importance_ratio_token_rows > 0`: the matching final SDPO rows also carry
  the ratio evidence needed by the reference smoke's token-level rollout
  importance correction path (`rollout_is = "token"`).
- strict smoke verification also requires every final weighted SDPO token row
  to carry `sdpo_rollout_is_weights` that match token-level truncated
  rollout-IS, proving the exported batch exercised the configured weighting
  path. The CLI reports this as `rollout_is_weight_rows` and
  `rollout_is_weight_token_rows`.

The wrapper script fails fast before training if the selected config no longer
uses the student-support path, token exports, an unfused LM head, or the expected
live/EMA teacher mode. The EMA wrapper path also requires
`deployment.num_sdpo_teacher_gpus > 0` so the local teacher inference pool is
actually launched. Those are part of this smoke's reference-aligned contract, not
just verifier conveniences.
The smoke also pins `dont_reprompt_on_self_success = true` because that is the
active Hübotter/verl YAML setting: a successful rollout without a different
successful sibling is masked out rather than reprompted with itself as the
solution. Setting that knob to `false` remains the literal Table 2 behavior from
the paper, but it is not the reference-smoke contract.

After a training smoke completes, the wrapper writes
`sdpo_smoke_provenance.txt` into the output directory before running the strict
artifact verifier. The file records the selected mode, config path, expected
top-k width, the resolved reference SDPO knobs, git commit, branch, runner
commands, `git status --short`, and SHA-256 fingerprints of the tracked diff,
staged diff, and untracked-file content manifest. The recorded knobs include
the student-support top-k mode, live/EMA teacher regularization, self-success
masking, batch-order successful-sibling selection, feedback inclusion,
`template_target = "first_user"`, and the trainer SDPO loss settings used by
the reference smoke. It also embeds the readable untracked-file manifest, with
one SHA-256/path row per file. Treat that file as run provenance: it does not
replace the artifact verifier, but it does make a passing CUDA/vLLM smoke
reproducible and reviewable later, even before the contribution is collapsed
into a final commit. For fresh wrapper-launched training smokes, the wrapper
passes `--require-provenance` plus the expected mode/config to the verifier,
and the verifier checks the recorded top-k width and resolved reference knobs
against the smoke contract. In `--require-provenance` mode, it also requires
the commit/branch, runner commands, and source-tree fingerprints, and rejects
placeholder `unknown` / `unavailable` values for the git and hash fields. It
also recomputes the embedded untracked-file manifest hash and checks it against
the recorded `git_untracked_manifest_sha256`, and requires the bounded
`git status --short` section to include its end marker. Use the same flags for
manual acceptance checks of fresh outputs; omit them only when inspecting
artifact bundles that do not carry that provenance file. The provenance lookup
works from the run output directory and from nested `run_default`,
`token_exports`, `broadcasts`, step, or rank-file paths.

`run_sdpo_cuda_acceptance.sh` writes
`sdpo_cuda_acceptance_summary.txt` under the chosen output root only after both
the live-policy and EMA smoke checks pass. The summary is the combined acceptance
breadcrumb and points at each smoke's provenance file, verifier report,
token-export directory, the EMA broadcast directory, and the requested archive
path when one is provided. Each smoke verification also writes
`sdpo_smoke_verify_report.txt` beside its provenance file. After training, the
combined launcher re-runs the strict verifier for each smoke and overwrites those
reports with acceptance-owned verification output. It checks the reports for
`Verified SDPO smoke provenance:`, `Verified SDPO token exports:`, and the
EMA-only `Verified SDPO EMA broadcasts:` marker before it writes the summary or
archive. It also requires every mandatory proof file to be non-empty and every
mandatory proof directory to contain at least one non-empty file, so empty
directory scaffolding cannot produce a combined success marker.
It also writes
`sdpo_cuda_acceptance_manifest.txt` with SHA-256 hashes and byte sizes for the
proof files under the output root. Pass
`--archive /path/to/proof.tar.gz` after a successful training run or during
`--no-run` re-verification to bundle the summary, per-smoke provenance files,
the manifest, verifier reports, token exports, EMA broadcasts, and resolved
config/control metadata when present for download from a remote CUDA box. The
wrapper rejects an archive that is empty, cannot be listed, or fails the offline
archive verifier after writing. After downloading the tarball, run
`verify_sdpo_cuda_acceptance_archive.py` again to
check the required safe regular-file proof members, success markers,
summary/manifest acceptance mode agreement, non-placeholder summary git
identity, live/EMA provenance mode/config/top-k fields, resolved reference SDPO
knobs, non-empty env names on active SDPO records, provenance source-tree
fingerprints including the embedded untracked-file manifest hash, matching
summary/provenance commit and branch identity across both smoke runs, and every
manifest hash/size against the archived bytes. The verifier rejects archive
members other than regular files and directories, and rejects duplicate archive
member paths.

For a rented CUDA box, prefer `scripts/start_sdpo_cuda_acceptance_background.sh`
after syncing the branch. It starts the same combined acceptance command under
`nohup`, records a PID file, writes a log, and prints the remote monitor plus
download/verify commands. Run `--preflight-only` first on a fresh machine to
check basic host tooling, visible GPU count, and the combined acceptance config
without starting training; the helper runs those preflights by default before
`nohup` unless `--skip-host-preflight` or `--skip-config-preflight` is passed.
Re-run it with `--status` to report the
recorded PID, process state, archive state, log state, and the latest log lines
without starting another run. If a non-empty archive is present and the recorded
process is no longer running, status mode also runs the offline archive verifier
and returns nonzero when the completed proof tarball is invalid. If a recorded
process is stopped and no non-empty archive exists, status mode also returns
nonzero so failed or incomplete overnight runs are visible to automation. The default paths are
`outputs/sdpo-cuda-acceptance`, `outputs/sdpo-cuda-acceptance-proof.tar.gz`,
`outputs/sdpo-cuda-acceptance.log`, and `outputs/sdpo-cuda-acceptance.pid`.

For a broader local SDPO integration gate on macOS, use
`scripts/run_sdpo_local_validation.sh`. It adds the editable verifier environment
package roots plus the Python packages needed by the SDPO integration tests to
the ad hoc `uvx` runner, then runs the broad pytest slice plus shell/Python
syntax checks, ShellCheck over the SDPO shell wrappers, direct live/EMA smoke
config checks, the combined CUDA acceptance config check, targeted Ruff
lint/format checks, an explicit whitespace/final-newline scan over tracked and
untracked SDPO contribution files, and `git diff --check`. Full
`tests/unit` collection should be reserved for the normal Linux environment
because unrelated model/inference/SFT tests require Linux-side packages such as
`torchtitan` and `vllm`.

The preflight trainer pass itself is forward/export-only: it may write student
support token exports and mark the export step stable, but it must not compute a
loss, run backward, step the optimizer or scheduler, update the EMA teacher, or
advance trainer progress. Prime may still publish step-boundary broadcasts or
checkpoints before it receives the next batch; those artifacts reflect earlier
optimizer updates, not work caused by the preflight batch. When the same
`progress.step` re-enters after preflight export, that step-boundary
maintenance is reused rather than repeated. The final training batch is the
first pass that may consume the teacher-rescored support.

For `sdpo_huebotter_reference_ema_smoke.toml`, pass `--require-ema-teacher`. That
also requires stable `broadcasts/step_*/sdpo_teacher` artifacts. Because the
trainer does not publish filesystem broadcasts before the first optimizer
update, matching SDPO token-export evidence at `step_0` is allowed without an
EMA broadcast. Every later matching-support SDPO token-export step must have
the same-step EMA teacher broadcast.
