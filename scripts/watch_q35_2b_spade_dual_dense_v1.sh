#!/usr/bin/env bash
set -euo pipefail

tmux_target=${1:?tmux target required}
runner_pattern=${2:?runner process pattern required}
stop_file=${3:?stop file required}
events_file=${4:?controller events file required}
watchdog_log=${5:?watchdog log required}
restart_command=${6:?restart command required}
poll_seconds=${7:-60}
max_restarts_per_head=${8:-3}
max_env_server_rss_kib=${9:-50331648}

if [[ "$poll_seconds" -lt 1 || "$max_restarts_per_head" -lt 1 || "$max_env_server_rss_kib" -lt 1 ]]; then
  echo "poll interval, restart limit, and EnvServer RSS limit must be positive" >&2
  exit 2
fi

last_restart_head=
restarts_at_head=0
healthy_checks=0

record() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" >>"$watchdog_log"
}

runner_present() {
  local process pid proc_cmdline
  for process in /proc/[0-9]*; do
    pid=${process##*/}
    [[ "$pid" != "$$" ]] || continue
    if ! proc_cmdline=$(tr '\0' ' ' <"$process/cmdline" 2>/dev/null); then
      continue
    fi
    if [[ "$proc_cmdline" == *python* && "$proc_cmdline" =~ $runner_pattern ]]; then
      return 0
    fi
  done
  return 1
}

guard_env_server_memory() {
  local pid rss command
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    rss=$(ps -o rss= -p "$pid" | tr -d ' ')
    [[ "$rss" =~ ^[0-9]+$ ]] || continue
    if ((rss <= max_env_server_rss_kib)); then
      continue
    fi
    command="touch $stop_file && kill -KILL $pid"
    record "env_server_rss_guard pid=$pid rss_kib=$rss limit_kib=$max_env_server_rss_kib"
    if ! tmux send-keys -t "$tmux_target" -l "$command" \
      || ! tmux send-keys -t "$tmux_target" Enter; then
      record "env_server_rss_guard_dispatch_failed tmux_target=$tmux_target"
    fi
    return 1
  done < <(pgrep -f '^PRIME-RL::EnvServer' || true)
}

record "watchdog_started poll_seconds=$poll_seconds max_restarts_per_head=$max_restarts_per_head max_env_server_rss_kib=$max_env_server_rss_kib"
while [[ ! -e "$stop_file" ]]; do
  if runner_present; then
    if ! guard_env_server_memory; then
      exit 1
    fi
    healthy_checks=$((healthy_checks + 1))
    if ((healthy_checks % 60 == 0)); then
      record "runner_present"
    fi
    sleep "$poll_seconds"
    continue
  fi
  healthy_checks=0
  if ! gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); then
    record "runner_absent nvidia_smi_failed"
    sleep "$poll_seconds"
    continue
  fi
  if [[ -n "$gpu_pids" ]]; then
    record "runner_absent gpu_busy_wait"
    sleep "$poll_seconds"
    continue
  fi
  if [[ ! -s "$events_file" ]]; then
    record "runner_absent controller_events_missing"
    exit 1
  fi

  controller_head=$(tail -n 1 "$events_file" | sha256sum | awk '{print $1}')
  if [[ "$controller_head" == "$last_restart_head" ]]; then
    restarts_at_head=$((restarts_at_head + 1))
  else
    last_restart_head=$controller_head
    restarts_at_head=1
  fi
  if ((restarts_at_head > max_restarts_per_head)); then
    record "restart_fuse_open controller_head=$controller_head attempts=$max_restarts_per_head"
    exit 1
  fi

  record "restart_dispatched controller_head=$controller_head attempt=$restarts_at_head"
  if ! tmux send-keys -t "$tmux_target" -l "$restart_command" \
    || ! tmux send-keys -t "$tmux_target" Enter; then
    record "restart_dispatch_failed tmux_target=$tmux_target"
    exit 1
  fi
  sleep "$poll_seconds"
done

record "stop_file_present"
