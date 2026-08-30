#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
action=${1:-status}
session=q35-grpo-v3
state_dir=$root/experiments/qwen35-2b-self-bootstrap-dual-dense-v1/grpo-autonomous-v3
stop_file=$state_dir/STOP
events_file=$state_dir/events.jsonl
controller_log=$state_dir/controller.log
watchdog_log=$state_dir/watchdog.log
publisher_log=$state_dir/publisher.log
publication_events=$state_dir/hf-publications.jsonl
hf_secret=$state_dir/hf-token.json
coordinator_hf_repo=lentzl/rlm-prime-agent-qwen35-2b-spade-coordinator
child_hf_repo=lentzl/rlm-prime-agent-qwen35-2b-spade-child
coordinator=/home/ubuntu/rlm/outputs/q35-2b-self-bootstrap-dual-dense-grpo-v1/grpo-auto-000158-coordinator-e0d3-uncapped-yield-exact-child-9415800/weights/step_1
child=/home/ubuntu/rlm/outputs/q35-2b-recursive-coordinator-return-v1/c160-child-minimal-balanced-consolidation-v9/weights/step_2
initial_admission=/home/ubuntu/rlm/results/q35-2b-recursive-coordinator-return-v1/c158-v9-tight-production-v37-9427800-n6

controller() {
  if [[ -e "$stop_file" ]]; then
    echo "refusing to run while stop file exists: $stop_file" >&2
    exit 1
  fi
  export Q35_2B_ROLE_GRPO_LR=1e-6
  export PATH="$root/.venv/bin:$HOME/.local/bin:$PATH"
  exec "$root/.venv/bin/python" "$root/scripts/run_q35_2b_role_grpo_autonomous_v1.py" \
    --state-dir "$state_dir" \
    --coordinator-model "$coordinator" \
    --coordinator-label C158 \
    --child-model "$child" \
    --child-label V9 \
    --initial-admission-run "$initial_admission" \
    --child-phase e0c3_natural_child_minimal \
    --next-role coordinator \
    --start-cycle 163 \
    --start-index 9411700 \
    --admission-scaffold-profile tight_answer_free_child_reporting_v1 \
    --learned-designer \
    --stop-file "$stop_file" \
    --poll-seconds 30 \
    --prune-below-gib 120
}

publisher() {
  if [[ -e "$stop_file" ]]; then
    echo "refusing to publish while stop file exists: $stop_file" >&2
    exit 1
  fi
  if [[ ! -f "$hf_secret" ]]; then
    echo "HF publication secret is unavailable: $hf_secret" >&2
    exit 1
  fi
  export PATH="$root/.venv/bin:$HOME/.local/bin:$PATH"
  exec "$root/.venv/bin/python" "$root/scripts/sync_q35_2b_hf_promotions_v1.py" \
    --events-file "$events_file" \
    --publication-events "$publication_events" \
    --secret-file "$hf_secret" \
    --coordinator-repo "$coordinator_hf_repo" \
    --child-repo "$child_hf_repo" \
    --poll-seconds 60
}

start() {
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "autonomous tmux session already exists: $session" >&2
    exit 1
  fi
  if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "refusing to start while a GPU process is active" >&2
    exit 1
  fi
  for checkpoint in "$coordinator" "$child"; do
    if [[ ! -f "$checkpoint/STABLE" || ! -f "$checkpoint/model.safetensors" ]]; then
      echo "protected autonomous checkpoint is unavailable: $checkpoint" >&2
      exit 1
    fi
  done
  if [[ ! -f "$initial_admission/SUMMARY.json" || ! -f "$initial_admission/VERSIONS.txt" ]]; then
    echo "initial tight admission is unavailable: $initial_admission" >&2
    exit 1
  fi
  mkdir -p "$state_dir"
  rm -f -- "$stop_file"

  local script controller_shell controller_window_shell restart_command watchdog_shell publisher_shell
  script=$root/scripts/run_q35_2b_role_grpo_autonomous_v3.sh
  printf -v controller_shell '%q ' bash "$script" controller
  controller_shell+=">>$(printf '%q' "$controller_log") 2>&1"
  restart_command=$controller_shell
  controller_window_shell="$controller_shell; exec bash"
  tmux new-session -d -s "$session" -n controller "$controller_window_shell"

  printf -v watchdog_shell '%q ' \
    bash "$root/scripts/watch_q35_2b_spade_dual_dense_v1.sh" \
    "$session:controller.0" "grpo-autonomous-v3" "$stop_file" \
    "$events_file" "$watchdog_log" "$restart_command" 10 3 50331648
  watchdog_shell+=">>$(printf '%q' "$watchdog_log") 2>&1"
  tmux new-window -d -t "$session:" -n watchdog "$watchdog_shell"
  if [[ -f "$hf_secret" ]]; then
    printf -v publisher_shell '%q ' bash "$script" publisher
    publisher_shell+=">>$(printf '%q' "$publisher_log") 2>&1; exec bash"
    tmux new-window -d -t "$session:" -n publisher "$publisher_shell"
  else
    echo "warning: HF publisher not started because secret is absent: $hf_secret" >&2
  fi
  echo "started autonomous role-GRPO loop in tmux session $session"
}

status() {
  if tmux has-session -t "$session" 2>/dev/null; then
    tmux list-panes -a -F '#S:#I.#P #{pane_dead} #{pane_pid} #{pane_current_command}' \
      | grep "^$session:"
  else
    echo "tmux_session=absent"
  fi
  if [[ -s "$events_file" ]]; then
    PYTHONPATH="$root/scripts" "$root/.venv/bin/python" -c \
      'import json,sys; import run_q35_2b_role_grpo_autonomous_v1 as c; print(json.dumps(c.project(c.load_events(c.Path(sys.argv[1]))), sort_keys=True))' \
      "$events_file"
    tail -n 1 "$events_file"
  else
    echo "events=absent"
  fi
  df -h /home/ubuntu/rlm | tail -n 1
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
}

case "$action" in
  controller) controller ;;
  publisher) publisher ;;
  start) start ;;
  stop)
    mkdir -p "$state_dir"
    touch "$stop_file"
    echo "stop requested; the controller will stop at the next safe action boundary"
    ;;
  status) status ;;
  *) echo "usage: $0 {start|stop|status|controller|publisher}" >&2; exit 2 ;;
esac
