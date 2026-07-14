#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run a Hübotter-style SDPO debug smoke and verify its artifacts.

Usage:
  scripts/run_sdpo_smoke_and_verify.sh [--ema] [--output-dir DIR] [--clean-output-dir] [--check-config] [--no-run]

Options:
  --ema             Run the EMA-teacher smoke preset and require EMA broadcasts.
  --output-dir DIR  Write the run under DIR. Defaults to outputs/<preset>-<utc timestamp>.
  --clean-output-dir
                   Delete an existing output/checkpoint dir before training.
  --check-config   Resolve the selected RL config and print SDPO-critical settings, then exit.
  --no-run          Skip training and only verify artifacts under --output-dir (requires --output-dir).
  -h, --help        Show this help.

Examples:
  scripts/run_sdpo_smoke_and_verify.sh
  scripts/run_sdpo_smoke_and_verify.sh --ema --output-dir outputs/sdpo-ema-smoke
  scripts/run_sdpo_smoke_and_verify.sh --ema --check-config
  scripts/run_sdpo_smoke_and_verify.sh --no-run --output-dir outputs/sdpo-smoke
EOF
}

mode="live"
output_dir=""
run_training=1
clean_output_dir=0
check_config=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ema)
      mode="ema"
      shift
      ;;
    --output-dir)
      if [[ $# -lt 2 ]]; then
        echo "Error: --output-dir requires a value" >&2
        exit 2
      fi
      output_dir="$2"
      shift 2
      ;;
    --no-run)
      run_training=0
      shift
      ;;
    --clean-output-dir)
      clean_output_dir=1
      shift
      ;;
    --check-config)
      check_config=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

read -r -a python_runner <<< "${SDPO_SMOKE_PYTHON_RUNNER:-uv run --extra flash-attn --extra envs python}"
read -r -a rl_runner <<< "${SDPO_SMOKE_RL_RUNNER:-uv run --extra flash-attn --extra envs rl}"
config_pythonpath="src:packages/prime-rl-configs/src:deps/pydantic-config/src:deps/verifiers:deps/renderers:deps/research-environments"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_SILENT="${WANDB_SILENT:-true}"

hash_stdin_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum | awk '{print $1}'
    return
  fi
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 | awk '{print $1}'
    return
  fi
  echo "unavailable"
}

hash_file_sha256() {
  local file="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file" | awk '{print $1}'
    return
  fi
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$file" | awk '{print $1}'
    return
  fi
  echo "unavailable"
}

hash_git_diff_sha256() {
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "unavailable"
    return
  fi
  git diff --no-ext-diff --binary 2>/dev/null | hash_stdin_sha256
}

hash_git_cached_diff_sha256() {
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "unavailable"
    return
  fi
  git diff --cached --no-ext-diff --binary 2>/dev/null | hash_stdin_sha256
}

hash_git_untracked_manifest_sha256() {
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "unavailable"
    return
  fi
  {
    while IFS= read -r -d '' file; do
      if [[ -f "$file" ]]; then
        printf '%s  %s\n' "$(hash_file_sha256 "$file")" "$file"
      else
        printf 'non-file  %s\n' "$file"
      fi
    done < <(git ls-files --others --exclude-standard -z)
  } | LC_ALL=C sort | hash_stdin_sha256
}

write_git_untracked_manifest() {
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "unavailable"
    return
  fi
  {
    while IFS= read -r -d '' file; do
      if [[ -f "$file" ]]; then
        printf '%s  %s\n' "$(hash_file_sha256 "$file")" "$file"
      else
        printf 'non-file  %s\n' "$file"
      fi
    done < <(git ls-files --others --exclude-standard -z)
  } | LC_ALL=C sort
}

git_commit_sha() {
  local commit
  commit="$(git rev-parse HEAD 2>/dev/null || true)"
  if [[ -n "$commit" ]]; then
    echo "$commit"
    return
  fi
  echo "unknown"
}

git_branch_name() {
  local branch
  branch="$(git branch --show-current 2>/dev/null || true)"
  if [[ -n "$branch" ]]; then
    echo "$branch"
    return
  fi
  local short_commit
  short_commit="$(git rev-parse --short HEAD 2>/dev/null || true)"
  if [[ -n "$short_commit" ]]; then
    echo "detached-$short_commit"
    return
  fi
  echo "unknown"
}

if [[ "$mode" == "ema" ]]; then
  config="configs/debug/algorithms/sdpo_huebotter_reference_ema_smoke.toml"
  default_name="sdpo-huebotter-reference-ema-smoke"
else
  config="configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml"
  default_name="sdpo-huebotter-reference-smoke"
fi

if [[ "$run_training" -eq 0 && "$check_config" -eq 0 && -z "$output_dir" ]]; then
  echo "Error: --no-run requires --output-dir so the verifier knows which completed smoke artifacts to inspect." >&2
  exit 2
fi
if [[ "$run_training" -eq 0 && "$check_config" -eq 0 && "$clean_output_dir" -eq 1 ]]; then
  echo "Error: --clean-output-dir cannot be combined with --no-run; verification-only mode must preserve existing artifacts." >&2
  exit 2
fi

if [[ -z "$output_dir" ]]; then
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  output_dir="outputs/${default_name}-${timestamp}"
fi

resolve_config_value() {
  local field="$1"
  local -a python_args
  python_args=("@" "$config" "--output-dir" "$output_dir")
  if [[ "$clean_output_dir" -eq 1 && "$check_config" -eq 0 ]]; then
    python_args+=("--clean-output-dir")
  fi
  PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${config_pythonpath}" \
    "${python_runner[@]}" - "$field" "${python_args[@]}" <<'PY'
from prime_rl.configs.rl import RLConfig
from prime_rl.utils.config import cli
import sys

field = sys.argv[1]
cfg = cli(RLConfig, args=sys.argv[2:])
if field == "orchestrator.sdpo_teacher.base_url":
    teacher = cfg.orchestrator.sdpo_teacher
    print(None if teacher is None else teacher.client.base_url)
    raise SystemExit
if field == "orchestrator.train.env_ids":
    print([env.env_id for env in cfg.orchestrator.train.env])
    raise SystemExit
if field == "orchestrator.eval.env_ids":
    print(None if cfg.orchestrator.eval is None else [env.env_id for env in cfg.orchestrator.eval.env])
    raise SystemExit
value = cfg
for part in field.split("."):
    value = getattr(value, part)
print(value)
PY
}

write_smoke_provenance() {
  local provenance_file="$output_dir/sdpo_smoke_provenance.txt"
  mkdir -p "$output_dir"
  {
    echo "sdpo_smoke_provenance_version=1"
    echo "mode=$mode"
    echo "config=$config"
    echo "output_dir=$output_dir"
    echo "expected_topk=$expected_topk"
    echo "orchestrator.algo.distillation_topk=$algorithm_topk"
    echo "orchestrator.algo.distillation_topk_support=$algorithm_topk_support"
    echo "orchestrator.algo.teacher_regularization=$algorithm_teacher_regularization"
    echo "orchestrator.algo.teacher_update_rate=$algorithm_teacher_update_rate"
    echo "orchestrator.algo.success_reward_threshold=$success_reward_threshold"
    echo "orchestrator.algo.successful_demonstration_selection=$successful_demonstration_selection"
    echo "orchestrator.algo.dont_reprompt_on_self_success=$dont_reprompt_on_self_success"
    echo "orchestrator.algo.remove_thinking_from_demonstration=$remove_thinking_from_demonstration"
    echo "orchestrator.algo.include_environment_feedback=$include_environment_feedback"
    echo "orchestrator.algo.environment_feedback_only_without_solution=$environment_feedback_only_without_solution"
    echo "orchestrator.algo.max_reprompt_len=$max_reprompt_len"
    echo "orchestrator.algo.reprompt_truncation=$reprompt_truncation"
    echo "orchestrator.algo.assistant_prefix=$assistant_prefix"
    echo "orchestrator.algo.multi_turn=$multi_turn"
    echo "orchestrator.algo.template_target=$template_target"
    echo "trainer.sdpo_loss.full_logit_distillation=$full_logit_distillation"
    echo "trainer.sdpo_loss.distillation_topk=$expected_topk"
    echo "trainer.sdpo_loss.distillation_add_tail=$distillation_add_tail"
    echo "trainer.sdpo_loss.alpha=$sdpo_alpha"
    echo "trainer.sdpo_loss.is_clip=$is_clip"
    echo "trainer.sdpo_loss.rollout_is=$rollout_is"
    echo "trainer.sdpo_loss.rollout_is_threshold=$rollout_is_threshold"
    echo "trainer.sdpo_loss.rollout_is_batch_normalize=$rollout_is_batch_normalize"
    echo "trainer.sdpo_runtime.teacher_regularization=$teacher_regularization"
    echo "trainer.sdpo_runtime.teacher_update_rate=$teacher_update_rate"
    echo "git_commit=$(git_commit_sha)"
    echo "git_branch=$(git_branch_name)"
    echo "git_diff_sha256=$(hash_git_diff_sha256)"
    echo "git_cached_diff_sha256=$(hash_git_cached_diff_sha256)"
    echo "git_untracked_manifest_sha256=$(hash_git_untracked_manifest_sha256)"
    echo "python_runner=${python_runner[*]}"
    echo "rl_runner=${rl_runner[*]}"
    echo "git_untracked_manifest_begin"
    write_git_untracked_manifest
    echo "git_untracked_manifest_end"
    echo "git_status_short_begin"
    git status --short 2>/dev/null || true
    echo "git_status_short_end"
  } > "$provenance_file"
  echo "Wrote SDPO smoke provenance: $provenance_file"
}

if [[ "$check_config" -eq 1 ]]; then
  python_args=("@" "$config" "--output-dir" "$output_dir")
  PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${config_pythonpath}" \
    "${python_runner[@]}" - "${python_args[@]}" <<'PY'
from prime_rl.configs.rl import RLConfig
from prime_rl.utils.config import cli
import sys

cfg = cli(RLConfig, args=sys.argv[1:])
print(f"config_output_dir={cfg.output_dir}")
print(f"clean_output_dir={cfg.clean_output_dir}")
print(f"max_steps={cfg.max_steps}")
print(f"seq_len={cfg.seq_len}")
print(f"uses_sdpo_student_support={cfg.uses_sdpo_student_support}")
print(f"uses_sdpo_internal_teacher_regularization={cfg.uses_sdpo_internal_teacher_regularization}")
print(f"deployment.num_sdpo_teacher_gpus={cfg.deployment.num_sdpo_teacher_gpus}")
print(f"orchestrator.batch_size={cfg.orchestrator.batch_size}")
print(f"orchestrator.group_size={cfg.orchestrator.group_size}")
print(f"orchestrator.renderer.name={cfg.orchestrator.renderer.name}")
print(f"orchestrator.train.sampling.max_completion_tokens={cfg.orchestrator.train.sampling.max_completion_tokens}")
print(f"orchestrator.train.env_ids={[env.env_id for env in cfg.orchestrator.train.env]}")
print(f"orchestrator.eval.interval={None if cfg.orchestrator.eval is None else cfg.orchestrator.eval.interval}")
print(f"orchestrator.eval.num_examples={None if cfg.orchestrator.eval is None else cfg.orchestrator.eval.num_examples}")
print(
    "orchestrator.eval.sampling.max_completion_tokens="
    f"{None if cfg.orchestrator.eval is None else cfg.orchestrator.eval.sampling.max_completion_tokens}"
)
print(f"orchestrator.eval.env_ids={None if cfg.orchestrator.eval is None else [env.env_id for env in cfg.orchestrator.eval.env]}")
print(f"orchestrator.algo.distillation_topk={cfg.orchestrator.algo.distillation_topk}")
print(f"orchestrator.algo.distillation_topk_support={cfg.orchestrator.algo.distillation_topk_support}")
print(f"orchestrator.algo.model={cfg.orchestrator.algo.model}")
print(f"orchestrator.algo.preflight_export_timeout_s={cfg.orchestrator.algo.preflight_export_timeout_s}")
print(f"orchestrator.algo.teacher_regularization={cfg.orchestrator.algo.teacher_regularization}")
print(f"orchestrator.algo.success_reward_threshold={cfg.orchestrator.algo.success_reward_threshold}")
print(f"orchestrator.algo.successful_demonstration_selection={cfg.orchestrator.algo.successful_demonstration_selection}")
print(f"orchestrator.algo.dont_reprompt_on_self_success={cfg.orchestrator.algo.dont_reprompt_on_self_success}")
print(f"orchestrator.algo.remove_thinking_from_demonstration={cfg.orchestrator.algo.remove_thinking_from_demonstration}")
print(f"orchestrator.algo.include_environment_feedback={cfg.orchestrator.algo.include_environment_feedback}")
print(f"orchestrator.algo.environment_feedback_only_without_solution={cfg.orchestrator.algo.environment_feedback_only_without_solution}")
print(f"orchestrator.algo.max_reprompt_len={cfg.orchestrator.algo.max_reprompt_len}")
print(f"orchestrator.algo.reprompt_truncation={cfg.orchestrator.algo.reprompt_truncation}")
print(f"orchestrator.algo.assistant_prefix={cfg.orchestrator.algo.assistant_prefix}")
print(f"orchestrator.algo.multi_turn={cfg.orchestrator.algo.multi_turn}")
print(f"orchestrator.algo.template_target={cfg.orchestrator.algo.template_target}")
print(f"orchestrator.algo.template={cfg.orchestrator.algo.template!r}")
print(f"orchestrator.algo.solution_template={cfg.orchestrator.algo.solution_template!r}")
print(f"orchestrator.algo.feedback_template={cfg.orchestrator.algo.feedback_template!r}")
print(f"trainer.enable_token_export={cfg.trainer.enable_token_export}")
print(f"trainer.model.cp={cfg.trainer.model.cp}")
print(f"trainer.model.fused_lm_head_token_chunk_size={cfg.trainer.model.fused_lm_head_token_chunk_size}")
print(f"trainer.sdpo_loss.full_logit_distillation={cfg.trainer.sdpo_loss.full_logit_distillation}")
print(f"trainer.sdpo_loss.distillation_topk={cfg.trainer.sdpo_loss.distillation_topk}")
print(f"trainer.sdpo_loss.distillation_add_tail={cfg.trainer.sdpo_loss.distillation_add_tail}")
print(f"trainer.sdpo_loss.alpha={cfg.trainer.sdpo_loss.alpha}")
print(f"trainer.sdpo_loss.is_clip={cfg.trainer.sdpo_loss.is_clip}")
print(f"trainer.sdpo_loss.rollout_is={cfg.trainer.sdpo_loss.rollout_is}")
print(f"trainer.sdpo_loss.rollout_is_threshold={cfg.trainer.sdpo_loss.rollout_is_threshold}")
print(f"trainer.sdpo_loss.rollout_is_batch_normalize={cfg.trainer.sdpo_loss.rollout_is_batch_normalize}")
print(f"trainer.sdpo_runtime.teacher_regularization={cfg.trainer.sdpo_runtime.teacher_regularization}")
print(f"trainer.sdpo_runtime.teacher_update_rate={cfg.trainer.sdpo_runtime.teacher_update_rate}")
print(f"orchestrator.algo.teacher_update_rate={cfg.orchestrator.algo.teacher_update_rate}")
teacher = cfg.orchestrator.sdpo_teacher
print(f"orchestrator.sdpo_teacher.base_url={None if teacher is None else teacher.client.base_url}")
PY
fi

max_steps="$(resolve_config_value max_steps)"
if [[ "$max_steps" != "20" ]]; then
  echo "Error: SDPO reference smoke requires max_steps=20 so the run produces enough final/preflight/EMA evidence without becoming a long training job (got $max_steps)" >&2
  exit 2
fi
seq_len="$(resolve_config_value seq_len)"
if [[ "$seq_len" != "2048" ]]; then
  echo "Error: SDPO reference smoke requires seq_len=2048 (got $seq_len)" >&2
  exit 2
fi
batch_size="$(resolve_config_value orchestrator.batch_size)"
if [[ "$batch_size" != "32" ]]; then
  echo "Error: SDPO reference smoke requires orchestrator.batch_size=32 (got $batch_size)" >&2
  exit 2
fi
group_size="$(resolve_config_value orchestrator.group_size)"
if [[ "$group_size" != "8" ]]; then
  echo "Error: SDPO reference smoke requires orchestrator.group_size=8 for sibling rollout construction (got $group_size)" >&2
  exit 2
fi
renderer_name="$(resolve_config_value orchestrator.renderer.name)"
if [[ "$renderer_name" != "qwen3" ]]; then
  echo "Error: SDPO reference smoke requires orchestrator.renderer.name='qwen3' (got $renderer_name)" >&2
  exit 2
fi
train_max_completion_tokens="$(resolve_config_value orchestrator.train.sampling.max_completion_tokens)"
if [[ "$train_max_completion_tokens" != "128" ]]; then
  echo "Error: SDPO reference smoke requires orchestrator.train.sampling.max_completion_tokens=128 (got $train_max_completion_tokens)" >&2
  exit 2
fi
train_env_ids="$(resolve_config_value orchestrator.train.env_ids)"
if [[ "$train_env_ids" != "['reverse-text']" ]]; then
  echo "Error: SDPO reference smoke requires orchestrator.train.env_ids=['reverse-text'] (got $train_env_ids)" >&2
  exit 2
fi
eval_interval="$(resolve_config_value orchestrator.eval.interval)"
if [[ "$eval_interval" != "1" ]]; then
  echo "Error: SDPO reference smoke requires orchestrator.eval.interval=1 (got $eval_interval)" >&2
  exit 2
fi
eval_num_examples="$(resolve_config_value orchestrator.eval.num_examples)"
if [[ "$eval_num_examples" != "128" ]]; then
  echo "Error: SDPO reference smoke requires orchestrator.eval.num_examples=128 (got $eval_num_examples)" >&2
  exit 2
fi
eval_max_completion_tokens="$(resolve_config_value orchestrator.eval.sampling.max_completion_tokens)"
if [[ "$eval_max_completion_tokens" != "128" ]]; then
  echo "Error: SDPO reference smoke requires orchestrator.eval.sampling.max_completion_tokens=128 (got $eval_max_completion_tokens)" >&2
  exit 2
fi
eval_env_ids="$(resolve_config_value orchestrator.eval.env_ids)"
if [[ "$eval_env_ids" != "['reverse-text']" ]]; then
  echo "Error: SDPO reference smoke requires orchestrator.eval.env_ids=['reverse-text'] (got $eval_env_ids)" >&2
  exit 2
fi
expected_topk="$(resolve_config_value trainer.sdpo_loss.distillation_topk)"
if [[ -z "$expected_topk" || "$expected_topk" == "None" ]]; then
  echo "Error: SDPO smoke requires trainer.sdpo_loss.distillation_topk to be set" >&2
  exit 2
fi
algorithm_topk="$(resolve_config_value orchestrator.algo.distillation_topk)"
if [[ "$algorithm_topk" != "$expected_topk" ]]; then
  echo "Error: SDPO smoke requires orchestrator.algo.distillation_topk to match trainer.sdpo_loss.distillation_topk (got $algorithm_topk vs $expected_topk)" >&2
  exit 2
fi
algorithm_topk_support="$(resolve_config_value orchestrator.algo.distillation_topk_support)"
if [[ "$algorithm_topk_support" != "student" ]]; then
  echo "Error: SDPO smoke requires orchestrator.algo.distillation_topk_support='student' (got $algorithm_topk_support)" >&2
  exit 2
fi
algorithm_model="$(resolve_config_value orchestrator.algo.model)"
if [[ "$algorithm_model" != "policy" ]]; then
  echo "Error: SDPO reference smoke requires orchestrator.algo.model='policy' for self-distillation (got $algorithm_model)" >&2
  exit 2
fi
success_reward_threshold="$(resolve_config_value orchestrator.algo.success_reward_threshold)"
if [[ "$success_reward_threshold" != "0.5" ]]; then
  echo "Error: SDPO reference smoke requires orchestrator.algo.success_reward_threshold=0.5 (got $success_reward_threshold)" >&2
  exit 2
fi
successful_demonstration_selection="$(resolve_config_value orchestrator.algo.successful_demonstration_selection)"
if [[ "$successful_demonstration_selection" != "batch_order" ]]; then
  echo "Error: SDPO reference smoke requires orchestrator.algo.successful_demonstration_selection='batch_order' (got $successful_demonstration_selection)" >&2
  exit 2
fi
dont_reprompt_on_self_success="$(resolve_config_value orchestrator.algo.dont_reprompt_on_self_success)"
if [[ "$dont_reprompt_on_self_success" != "True" ]]; then
  echo "Error: SDPO reference smoke requires orchestrator.algo.dont_reprompt_on_self_success=true (got $dont_reprompt_on_self_success)" >&2
  exit 2
fi
remove_thinking_from_demonstration="$(resolve_config_value orchestrator.algo.remove_thinking_from_demonstration)"
if [[ "$remove_thinking_from_demonstration" != "True" ]]; then
  echo "Error: SDPO reference smoke requires orchestrator.algo.remove_thinking_from_demonstration=true (got $remove_thinking_from_demonstration)" >&2
  exit 2
fi
include_environment_feedback="$(resolve_config_value orchestrator.algo.include_environment_feedback)"
if [[ "$include_environment_feedback" != "True" ]]; then
  echo "Error: SDPO reference smoke requires orchestrator.algo.include_environment_feedback=true (got $include_environment_feedback)" >&2
  exit 2
fi
environment_feedback_only_without_solution="$(resolve_config_value orchestrator.algo.environment_feedback_only_without_solution)"
if [[ "$environment_feedback_only_without_solution" != "True" ]]; then
  echo "Error: SDPO reference smoke requires orchestrator.algo.environment_feedback_only_without_solution=true (got $environment_feedback_only_without_solution)" >&2
  exit 2
fi
max_reprompt_len="$(resolve_config_value orchestrator.algo.max_reprompt_len)"
if [[ "$max_reprompt_len" != "10240" ]]; then
  echo "Error: SDPO reference smoke requires orchestrator.algo.max_reprompt_len=10240 (got $max_reprompt_len)" >&2
  exit 2
fi
reprompt_truncation="$(resolve_config_value orchestrator.algo.reprompt_truncation)"
if [[ "$reprompt_truncation" != "right" ]]; then
  echo "Error: SDPO reference smoke requires orchestrator.algo.reprompt_truncation='right' (got $reprompt_truncation)" >&2
  exit 2
fi
assistant_prefix="$(resolve_config_value orchestrator.algo.assistant_prefix)"
if [[ -n "$assistant_prefix" ]]; then
  echo "Error: SDPO reference smoke requires orchestrator.algo.assistant_prefix='' (got $assistant_prefix)" >&2
  exit 2
fi
multi_turn="$(resolve_config_value orchestrator.algo.multi_turn)"
if [[ "$multi_turn" != "False" ]]; then
  echo "Error: SDPO reference smoke requires orchestrator.algo.multi_turn=false; use a separate smoke for Prime's multi-turn SDPO extension (got $multi_turn)" >&2
  exit 2
fi
template_target="$(resolve_config_value orchestrator.algo.template_target)"
if [[ "$template_target" != "first_user" ]]; then
  echo "Error: SDPO reference smoke requires orchestrator.algo.template_target='first_user' (got $template_target)" >&2
  exit 2
fi
template="$(resolve_config_value orchestrator.algo.template)"
expected_template=$'{question}{successful_solution_block}{feedback_block}\n\nCorrectly solve the original question.'
if [[ "$template" != "$expected_template" ]]; then
  echo "Error: SDPO reference smoke requires the Hübotter outer template (got $template)" >&2
  exit 2
fi
solution_template="$(resolve_config_value orchestrator.algo.solution_template)"
expected_solution_template=$'\nCorrect solution:\n\n{successful_previous_attempt}'
if [[ "$solution_template" != "$expected_solution_template" ]]; then
  echo "Error: SDPO reference smoke requires the Hübotter solution_template (got $solution_template)" >&2
  exit 2
fi
feedback_template="$(resolve_config_value orchestrator.algo.feedback_template)"
expected_feedback_template=$'\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}'
if [[ "$feedback_template" != "$expected_feedback_template" ]]; then
  echo "Error: SDPO reference smoke requires the Hübotter feedback_template (got $feedback_template)" >&2
  exit 2
fi
preflight_export_timeout_s="$(resolve_config_value orchestrator.algo.preflight_export_timeout_s)"
if [[ -z "$preflight_export_timeout_s" || "$preflight_export_timeout_s" == "None" ]]; then
  echo "Error: SDPO reference smoke requires orchestrator.algo.preflight_export_timeout_s to be set so student-support preflight export hangs fail diagnostically" >&2
  exit 2
fi
full_logit_distillation="$(resolve_config_value trainer.sdpo_loss.full_logit_distillation)"
if [[ "$full_logit_distillation" != "True" ]]; then
  echo "Error: SDPO smoke requires trainer.sdpo_loss.full_logit_distillation=true (got $full_logit_distillation)" >&2
  exit 2
fi
distillation_add_tail="$(resolve_config_value trainer.sdpo_loss.distillation_add_tail)"
if [[ "$distillation_add_tail" != "True" ]]; then
  echo "Error: SDPO reference smoke requires trainer.sdpo_loss.distillation_add_tail=true (got $distillation_add_tail)" >&2
  exit 2
fi
sdpo_alpha="$(resolve_config_value trainer.sdpo_loss.alpha)"
if [[ "$sdpo_alpha" != "0.5" ]]; then
  echo "Error: SDPO reference smoke requires trainer.sdpo_loss.alpha=0.5 (got $sdpo_alpha)" >&2
  exit 2
fi
is_clip="$(resolve_config_value trainer.sdpo_loss.is_clip)"
if [[ "$is_clip" != "2.0" ]]; then
  echo "Error: SDPO reference smoke requires trainer.sdpo_loss.is_clip=2.0 (got $is_clip)" >&2
  exit 2
fi
rollout_is="$(resolve_config_value trainer.sdpo_loss.rollout_is)"
if [[ "$rollout_is" != "token" ]]; then
  echo "Error: SDPO reference smoke requires trainer.sdpo_loss.rollout_is='token' (got $rollout_is)" >&2
  exit 2
fi
rollout_is_threshold="$(resolve_config_value trainer.sdpo_loss.rollout_is_threshold)"
if [[ "$rollout_is_threshold" != "2.0" ]]; then
  echo "Error: SDPO reference smoke requires trainer.sdpo_loss.rollout_is_threshold=2.0 (got $rollout_is_threshold)" >&2
  exit 2
fi
rollout_is_batch_normalize="$(resolve_config_value trainer.sdpo_loss.rollout_is_batch_normalize)"
if [[ "$rollout_is_batch_normalize" != "False" ]]; then
  echo "Error: SDPO reference smoke requires trainer.sdpo_loss.rollout_is_batch_normalize=false (got $rollout_is_batch_normalize)" >&2
  exit 2
fi
uses_student_support="$(resolve_config_value uses_sdpo_student_support)"
if [[ "$uses_student_support" != "True" ]]; then
  echo "Error: SDPO smoke requires distillation_topk_support='student' (uses_sdpo_student_support=$uses_student_support)" >&2
  exit 2
fi
uses_internal_teacher="$(resolve_config_value uses_sdpo_internal_teacher_regularization)"
if [[ "$mode" == "ema" && "$uses_internal_teacher" != "True" ]]; then
  echo "Error: --ema smoke requires an internal SDPO teacher runtime (uses_sdpo_internal_teacher_regularization=$uses_internal_teacher)" >&2
  exit 2
fi
if [[ "$mode" == "live" && "$uses_internal_teacher" != "False" ]]; then
  echo "Error: live SDPO smoke must not use an internal SDPO teacher runtime (uses_sdpo_internal_teacher_regularization=$uses_internal_teacher)" >&2
  exit 2
fi
num_sdpo_teacher_gpus="$(resolve_config_value deployment.num_sdpo_teacher_gpus)"
if [[ "$mode" == "ema" && "$num_sdpo_teacher_gpus" -le 0 ]]; then
  echo "Error: --ema reference smoke requires deployment.num_sdpo_teacher_gpus > 0 so the teacher inference pool is launched locally (got $num_sdpo_teacher_gpus)" >&2
  exit 2
fi
teacher_base_url="$(resolve_config_value orchestrator.sdpo_teacher.base_url)"
if [[ "$mode" == "ema" && "$teacher_base_url" != "['http://localhost:8001/v1']" ]]; then
  echo "Error: --ema reference smoke requires orchestrator.sdpo_teacher.base_url=['http://localhost:8001/v1'] so EMA top-k support is rescored by the local teacher server (got $teacher_base_url)" >&2
  exit 2
fi
if [[ "$mode" == "live" && "$teacher_base_url" != "None" ]]; then
  echo "Error: live SDPO smoke must not configure orchestrator.sdpo_teacher.base_url (got $teacher_base_url)" >&2
  exit 2
fi
token_export_enabled="$(resolve_config_value trainer.enable_token_export)"
if [[ "$token_export_enabled" != "True" ]]; then
  echo "Error: SDPO smoke requires trainer.enable_token_export=true" >&2
  exit 2
fi
context_parallelism="$(resolve_config_value trainer.model.cp)"
if [[ "$context_parallelism" != "1" ]]; then
  echo "Error: SDPO smoke requires trainer.model.cp=1 because top-k support export needs unsharded logits (got $context_parallelism)" >&2
  exit 2
fi
fused_head_mode="$(resolve_config_value trainer.model.fused_lm_head_token_chunk_size)"
if [[ "$fused_head_mode" != "disabled" ]]; then
  echo "Error: SDPO smoke requires trainer.model.fused_lm_head_token_chunk_size='disabled' (got $fused_head_mode)" >&2
  exit 2
fi
teacher_regularization="$(resolve_config_value trainer.sdpo_runtime.teacher_regularization)"
if [[ "$mode" == "ema" && "$teacher_regularization" != "ema" ]]; then
  echo "Error: --ema smoke requires trainer.sdpo_runtime.teacher_regularization='ema' (got $teacher_regularization)" >&2
  exit 2
fi
if [[ "$mode" == "live" && "$teacher_regularization" != "live-policy" ]]; then
  echo "Error: live SDPO smoke requires trainer.sdpo_runtime.teacher_regularization='live-policy' (got $teacher_regularization)" >&2
  exit 2
fi
algorithm_teacher_regularization="$(resolve_config_value orchestrator.algo.teacher_regularization)"
if [[ "$algorithm_teacher_regularization" != "$teacher_regularization" ]]; then
  echo "Error: SDPO smoke requires orchestrator.algo.teacher_regularization to match trainer.sdpo_runtime.teacher_regularization (got $algorithm_teacher_regularization vs $teacher_regularization)" >&2
  exit 2
fi
teacher_update_rate="$(resolve_config_value trainer.sdpo_runtime.teacher_update_rate)"
if [[ "$teacher_update_rate" != "0.05" ]]; then
  echo "Error: SDPO reference smoke requires trainer.sdpo_runtime.teacher_update_rate=0.05 (got $teacher_update_rate)" >&2
  exit 2
fi
algorithm_teacher_update_rate="$(resolve_config_value orchestrator.algo.teacher_update_rate)"
if [[ "$algorithm_teacher_update_rate" != "$teacher_update_rate" ]]; then
  echo "Error: SDPO smoke requires orchestrator.algo.teacher_update_rate to match trainer.sdpo_runtime.teacher_update_rate (got $algorithm_teacher_update_rate vs $teacher_update_rate)" >&2
  exit 2
fi

if [[ "$check_config" -eq 1 ]]; then
  echo "SDPO smoke config checks passed."
  exit 0
fi

if [[ "$run_training" -eq 1 ]]; then
  echo "Running SDPO smoke config: $config"
  echo "Output directory: $output_dir"
  train_cmd=("${rl_runner[@]}" @ "$config" --output-dir "$output_dir")
  if [[ "$clean_output_dir" -eq 1 ]]; then
    train_cmd+=(--clean-output-dir)
  fi
  "${train_cmd[@]}"
  write_smoke_provenance
else
  echo "Skipping training; verifying existing output directory: $output_dir"
fi

echo "Verifying SDPO smoke artifacts..."
verify_report_file="$output_dir/sdpo_smoke_verify_report.txt"
mkdir -p "$output_dir"
verify_cmd=("${python_runner[@]}" scripts/verify_sdpo_smoke_artifacts.py "$output_dir" --expected-topk "$expected_topk")
if [[ "$run_training" -eq 1 ]]; then
  verify_cmd+=(--require-provenance --expected-provenance-mode "$mode" --expected-provenance-config "$config")
fi
if [[ "$mode" == "ema" ]]; then
  verify_cmd+=(--require-ema-teacher)
fi
"${verify_cmd[@]}" | tee "$verify_report_file"
echo "Wrote SDPO smoke verifier report: $verify_report_file"
