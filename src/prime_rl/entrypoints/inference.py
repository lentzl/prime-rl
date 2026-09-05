import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from threading import Event, Thread
from typing import Any

from prime_rl.configs.inference import InferenceConfig
from prime_rl.utils.config import cli, dump_resolved_config
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pathing import format_log_message, get_config_dir, latest_log_dir
from prime_rl.utils.process import (
    DEFAULT_COMMON_ENV_VARS,
    DEFAULT_INFERENCE_ENV_VARS,
    cleanup_processes,
    set_proc_title,
)

INFERENCE_CONFIG = "inference.json"
INFERENCE_SBATCH = "inference.sbatch"


def vllm_overrides_fragment(overrides: dict[str, Any]) -> str:
    """Render per-role vLLM overrides as a JSON fragment for the ROLE_EXTRA bash string.

    Returns a leading-comma fragment with quotes escaped for the double-quoted assignment
    (e.g. `, \\"max_num_seqs\\": 256`), or an empty string when there are no overrides.
    """
    if not overrides:
        return ""
    return ", " + json.dumps(overrides)[1:-1].replace('"', '\\"')


def write_config(
    config: InferenceConfig, output_dir: Path, exclude: set[str] | None = None, engine_only: bool = False
) -> Path:
    """Write resolved config to disk.

    With ``engine_only``, the router is nulled so per-rank processes run bare
    engines — the sbatch script starts the single global router.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / INFERENCE_CONFIG
    config_dict = dump_resolved_config(config, exclude=exclude)
    if engine_only:
        config_dict["router"] = None
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    return config_path


def write_slurm_script(config: InferenceConfig, config_path: Path, script_path: Path) -> None:
    """Write the SLURM script to disk."""
    from jinja2 import Environment, FileSystemLoader

    assert config.slurm is not None
    assert config.slurm.template_path is not None

    env = Environment(loader=FileSystemLoader(config.slurm.template_path.parent), keep_trailing_newline=True)
    template = env.get_template(config.slurm.template_path.name)

    is_disaggregated = config.deployment.type == "disaggregated"
    dp_per_node = config.deployment.gpus_per_node // config.vllm.tensor_parallel_size

    offload = config.kv_cache_offload
    is_mooncake = offload is not None and offload.type == "mooncake"

    template_vars = dict(
        **config.slurm.template_vars,
        config_path=config_path,
        output_dir=config.output_dir,
        gpus_per_node=config.deployment.gpus_per_node,
        dp_per_node=dp_per_node,
        num_nodes=getattr(config.deployment, "num_nodes", 1),
        port=config.server.port,
        router=config.router,
        router_port=config.server.port,
        is_disaggregated=is_disaggregated,
        kv_offload=offload is not None,
        kv_offload_mooncake=is_mooncake,
        kv_offload_cpu_bytes=int(offload.cpu.num_bytes) if is_mooncake else 0,
        kv_offload_disk_path=str(offload.disk.path) if (is_mooncake and offload.disk is not None) else "",
        kv_offload_device_name=offload.device_name if is_mooncake else "",
        inference_env_vars={**DEFAULT_COMMON_ENV_VARS, **DEFAULT_INFERENCE_ENV_VARS, **config.env_vars},
    )

    is_multi_node = config.deployment.type == "multi_node"

    if is_disaggregated:
        template_vars.update(
            num_prefill_nodes=config.deployment.num_prefill_nodes,
            num_decode_nodes=config.deployment.num_decode_nodes,
            prefill_nodes_per_replica=config.deployment.prefill_nodes_per_replica,
            decode_nodes_per_replica=config.deployment.decode_nodes_per_replica,
            num_prefill_replicas=config.deployment.num_prefill_replicas,
            num_decode_replicas=config.deployment.num_decode_replicas,
            prefill_port=config.deployment.prefill_port,
            decode_port=config.deployment.decode_port,
            data_parallel_rpc_port=config.vllm.data_parallel_rpc_port,
            use_deep_gemm=config.use_deep_gemm,
            prefill_env_vars=config.deployment.prefill_env_vars,
            decode_env_vars=config.deployment.decode_env_vars,
            prefill_vllm_extra_json=vllm_overrides_fragment(config.deployment.prefill_vllm_overrides),
            decode_vllm_extra_json=vllm_overrides_fragment(config.deployment.decode_vllm_overrides),
        )
    elif is_multi_node:
        template_vars.update(
            backend_port=config.backend_port,
            data_parallel_rpc_port=config.vllm.data_parallel_rpc_port,
            enable_expert_parallel=config.vllm.enable_expert_parallel,
            infer_nodes_per_replica=config.deployment.num_nodes,
        )

    script = template.render(**template_vars)

    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)


def inference_slurm(config: InferenceConfig):
    """Run inference via SLURM."""
    assert config.slurm is not None

    logger = setup_logger(config.log.level, json_logging=config.log.json_logging)

    config_dir = get_config_dir(config.output_dir)
    is_multi_node = config.deployment.type in ("multi_node", "disaggregated")
    exclude = {"deployment", "slurm", "dry_run"} if is_multi_node else {"slurm", "dry_run"}
    config_path = write_config(config, config_dir, exclude=exclude, engine_only=is_multi_node)
    logger.info(f"Wrote config to {config_path}")

    script_path = config.output_dir / INFERENCE_SBATCH
    write_slurm_script(config, config_path, script_path)
    logger.info(f"Wrote SLURM script to {script_path}")

    log_dir = latest_log_dir(config.output_dir)
    num_nodes = getattr(config.deployment, "num_nodes", 1)
    log_message = format_log_message(log_dir=log_dir, inference=True, job_log=True, num_infer_nodes=num_nodes)

    if config.dry_run:
        logger.success(f"Dry run complete. To submit manually:\n\n  sbatch {script_path}\n\n{log_message}")
        return

    logger.info(f"Submitting: sbatch {script_path}")
    result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"sbatch failed: {result.stderr.strip()}")
        sys.exit(1)

    logger.success(f"{result.stdout.strip()}\n\n{log_message}")


def start_router(config: InferenceConfig) -> subprocess.Popen:
    """Start the vllm-router on ``server.port``, fronting the local engine at ``backend_port``."""
    assert config.router is not None and config.router.type == "vllm-router"
    host = config.server.host
    worker_host = "localhost" if host in (None, "0.0.0.0") else host
    cmd = [
        "vllm-router",
        "--policy",
        config.router.policy,
        "--host",
        host or "0.0.0.0",
        "--port",
        str(config.server.port),
        "--worker-urls",
        f"http://{worker_host}:{config.backend_port}",
        "--intra-node-data-parallel-size",
        str(config.vllm.data_parallel_size_local or config.vllm.data_parallel_size),
        "--request-id-headers",
        "x-session-id",
        "--worker-startup-timeout-secs",
        "4200",
        "--prometheus-port",
        str(config.server.port + 21000),
    ]
    return subprocess.Popen(cmd)


def start_role_router(config: InferenceConfig) -> list[subprocess.Popen]:
    """Start a frozen anchor engine and role-aware proxy beside the policy engine."""

    assert config.router is not None and config.router.type == "role-router"
    router = config.router
    router.state_dir.mkdir(parents=True, exist_ok=True)
    anchor_output = router.state_dir / "anchor"
    anchor_vllm = config.vllm.model_copy(
        update={
            "model": router.anchor_model,
            "gpu_memory_utilization": router.anchor_gpu_memory_utilization,
            "kv_cache_memory_bytes": router.anchor_kv_cache_memory_bytes,
            "data_parallel_rpc_port": router.anchor_data_parallel_rpc_port,
        }
    )
    anchor_config = config.model_copy(
        update={
            "server": config.server.model_copy(update={"port": router.anchor_backend_port}),
            "router": None,
            "backend_port": router.anchor_backend_port + 100,
            "vllm": anchor_vllm,
            "output_dir": anchor_output,
        }
    )
    anchor_config_path = write_config(anchor_config, anchor_output)
    anchor_log = open(router.state_dir / "anchor-inference.log", "w")
    anchor_process = subprocess.Popen(
        ["inference", "@", str(anchor_config_path)],
        stdout=anchor_log,
        stderr=anchor_log,
    )
    anchor_log.close()

    policy_url = f"http://127.0.0.1:{config.backend_port}/v1"
    anchor_url = f"http://127.0.0.1:{router.anchor_backend_port}/v1"
    if router.policy_role == "coordinator":
        coordinator_url, coordinator_model = policy_url, config.vllm.model
        child_url, child_model = anchor_url, router.anchor_model
    else:
        coordinator_url, coordinator_model = anchor_url, router.anchor_model
        child_url, child_model = policy_url, config.vllm.model

    repository_root = Path(__file__).resolve().parents[3]
    proxy_script = repository_root / "scripts" / "dual_policy_openai_proxy_v1.py"
    if not proxy_script.is_file():
        cleanup_processes([anchor_process])
        raise FileNotFoundError(f"role-router proxy is missing: {proxy_script}")
    proxy_log = open(router.state_dir / "proxy.log", "w")
    proxy_command = [
        sys.executable,
        str(proxy_script),
        "--host",
        config.server.host or "0.0.0.0",
        "--port",
        str(config.server.port),
        "--coordinator-url",
        coordinator_url,
        "--coordinator-model",
        coordinator_model,
        "--child-url",
        child_url,
        "--child-model",
        child_model,
        "--external-model",
        config.vllm.model,
        "--audit-log",
        str(router.audit_log),
    ]
    if router.leak_coordinator_exact_action:
        proxy_command.append("--leak-coordinator-exact-action")
    if router.leak_child_exact_action:
        proxy_command.append("--leak-child-exact-action")
    if router.strip_child_tool_choice:
        proxy_command.append("--strip-child-tool-choice")
    if router.strip_coordinator_tool_choice:
        proxy_command.append("--strip-coordinator-tool-choice")
    if router.specialist_fixed_expert is not None:
        proxy_command.extend(
            [
                "--specialist-worker-routing",
                "--specialist-route",
                router.specialist_fixed_expert,
                child_url,
                child_model,
                "--specialist-fixed-expert",
                router.specialist_fixed_expert,
            ]
        )
    if router.specialist_force_fixed_action:
        proxy_command.append("--specialist-force-fixed-action")
    proxy_process = subprocess.Popen(
        proxy_command,
        stdout=proxy_log,
        stderr=proxy_log,
    )
    proxy_log.close()
    return [proxy_process, anchor_process]


def inference_local(config: InferenceConfig):
    """Run inference locally: a router on ``server.port`` fronting the engine on ``backend_port``."""
    from prime_rl.inference.server import setup_vllm_env

    logger = setup_logger(config.log.level, json_logging=config.log.json_logging)

    if config.dry_run:
        logger.success("Dry run complete. To start inference locally, remove --dry-run from your command.")
        return

    host = config.server.host or "0.0.0.0"
    port = config.server.port

    # Apply the inference env (defaults + [inference.env_vars]) in-process so a standalone
    # `uv run inference` gets the same environment the rl/SLURM launchers inject into the
    # server subprocess. config.env_vars wins over the defaults; existing os.environ loses.
    os.environ.update({**DEFAULT_COMMON_ENV_VARS, **DEFAULT_INFERENCE_ENV_VARS, **config.env_vars})

    setup_vllm_env(config)

    router_processes: list[subprocess.Popen] = []
    router_stopping = Event()
    if config.router is not None:
        logger.info(
            f"Starting {config.router.type} on http://{host}:{port}/v1 "
            f"(policy engine on port {config.backend_port})\n"
        )
        if config.router.type == "role-router":
            router_processes = start_role_router(config)
        else:
            router_processes = [start_router(config)]
        # The router owns the client-facing port; the engine moves behind it.
        config.server.port = config.backend_port

        def watch_router(process: subprocess.Popen):
            process.wait()
            if not router_stopping.is_set():
                logger.error(f"Router process exited with code {process.returncode} - shutting down")
                os.kill(os.getpid(), signal.SIGTERM)

        for process in router_processes:
            Thread(target=watch_router, args=(process,), daemon=True).start()
    else:
        logger.info(f"Starting inference on http://{host}:{port}/v1\n")

    from prime_rl.inference.vllm.server import server  # pyright: ignore

    try:
        server(config)
    finally:
        if router_processes:
            router_stopping.set()
            cleanup_processes(router_processes)


def inference(config: InferenceConfig):
    if config.slurm is not None:
        inference_slurm(config)
    else:
        inference_local(config)


def main():
    set_proc_title("Inference")
    inference(cli(InferenceConfig))


if __name__ == "__main__":
    main()
