import warnings
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field, model_validator

from prime_rl.configs.algorithm import SDPO_MAX_STUDENT_SUPPORT_TOPK
from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.inference import WeightBroadcastConfig as InferenceWeightBroadcastConfig
from prime_rl.configs.orchestrator import (
    FileSystemWeightBroadcastConfig as OrchestratorFileSystemWeightBroadcastConfig,
)
from prime_rl.configs.orchestrator import (
    HostedModelConfig,
    OrchestratorConfig,
)
from prime_rl.configs.orchestrator import (
    NCCLWeightBroadcastConfig as OrchestratorNCCLWeightBroadcastConfig,
)
from prime_rl.configs.shared import (
    SlurmConfig,
    VLMConfig,
)
from prime_rl.configs.trainer import (
    BenchConfig,
    FakeDataLoaderConfig,
    TokenizerConfig,
    TrainerConfig,
)
from prime_rl.configs.trainer import (
    FileSystemWeightBroadcastConfig as TrainerFileSystemWeightBroadcastConfig,
)
from prime_rl.configs.trainer import (
    NCCLWeightBroadcastConfig as TrainerNCCLWeightBroadcastConfig,
)
from prime_rl.utils.config import BaseConfig, find_package_resource
from prime_rl.utils.validation import (
    propagate_shared_fields,
    validate_shared_ckpt_config,
    validate_shared_max_steps,
    validate_shared_model_name,
    validate_shared_output_dir,
    validate_shared_seq_len,
    validate_shared_tokenizer,
    validate_shared_wandb_config,
    validate_shared_weight_broadcast,
)

MAX_TCP_PORT = 65535


class RLExperimentalConfig(BaseConfig):
    pass


class SharedLogConfig(BaseConfig):
    level: str | None = None
    """Log level for trainer, orchestrator, and inference. When unset, each sub-config's own log level applies (defaults to ``$PRIME_LOG_LEVEL`` if set, else ``info``)."""

    json_logging: bool = False
    """Emit newline-delimited JSON logs for aggregation (Loki, Grafana, etc.). Propagated to trainer, orchestrator, and inference."""


class SharedWandbConfig(BaseConfig):
    project: str | None = "prime-rl"
    """W&B project."""

    entity: str | None = None
    """W&B entity."""

    name: str | None = None
    """W&B run name."""

    group: str | None = None
    """W&B group."""

    tags: list[str] | None = None
    """W&B tags attached to the run."""

    offline: bool | None = False
    """Run W&B in offline mode. Incompatible with shared mode, which is always on for the ``rl`` entrypoint."""

    @model_validator(mode="after")
    def validate_not_offline(self):
        if self.offline:
            raise ValueError(
                "W&B shared mode is always on for the rl entrypoint and requires server "
                "connectivity; wandb.offline = true is not supported. Use offline mode "
                "via the sub-config wandb blocks (trainer.wandb.offline, "
                "orchestrator.wandb.offline) if you really need it per-process."
            )
        return self


class SharedCheckpointConfig(BaseConfig):
    output_dir: Path | None = None
    """Override directory for checkpoints and weights. When set, checkpoints and weight snapshots are written here instead of under the trainer ``output_dir``."""

    interval: int | None = None
    """Interval at which to save checkpoints."""

    resume_step: int | None = None
    """Step to resume from. If None, does not resume from a checkpoint."""

    keep_last: int | None = Field(None, ge=1)
    """Keep at most this many recent step checkpoints on disk. If None, never clean old checkpoints based on recency."""

    keep_interval: int | None = Field(None, ge=1)
    """Keep checkpoints at every N steps permanently (e.g. ``keep_interval=100`` keeps step 100, 200, ...). If None, no interval-based keeping."""


class SharedModelConfig(BaseConfig):
    name: str = "Qwen/Qwen3-0.6B"
    """HF model name or local path."""

    vlm: "VLMConfig | None" = None
    """VLM configuration. Set this to enable vision-language model support."""


class SharedWeightBroadcastConfig(BaseConfig):
    type: Literal["nccl", "filesystem"] = "filesystem"
    """Weight broadcast transport."""

    port: int = 29501
    """Port for NCCL weight broadcast."""

    timeout: int = 1200
    """Timeout in seconds for NCCL weight broadcast."""

    quantize_in_weight_transfer: bool = False
    """Use kernel-format FP8 quantized NCCL transfer for weight updates. When disabled, uses default HF checkpoint-format transfer."""


class BaseDeploymentConfig(BaseConfig):
    gpus_per_node: int = 8
    """GPUs per node."""


class SingleNodeDeploymentConfig(BaseDeploymentConfig):
    type: Literal["single_node"] = "single_node"

    num_train_gpus: int = 1
    """GPUs allocated to the trainer."""

    num_infer_gpus: int = 1
    """GPUs allocated to inference."""

    num_sdpo_teacher_gpus: int = Field(0, ge=0)
    """Optional GPUs allocated to a separate SDPO teacher inference server for
    non-``live-policy`` teacher regularization. When 0, any
    ``orchestrator.sdpo_teacher`` endpoint is treated as user-managed."""

    @model_validator(mode="after")
    def validate_gpu_count(self):
        total = self.num_train_gpus + self.num_infer_gpus + self.num_sdpo_teacher_gpus
        if total > self.gpus_per_node:
            raise ValueError(
                f"Total GPU count ({total} = {self.num_train_gpus} train + {self.num_infer_gpus} infer + "
                f"{self.num_sdpo_teacher_gpus} sdpo teacher)"
                f" exceeds gpus_per_node ({self.gpus_per_node})."
            )
        return self


class MultiNodeDeploymentConfig(BaseDeploymentConfig):
    type: Literal["multi_node"] = "multi_node"

    num_train_nodes: int
    """Training nodes."""

    num_infer_nodes: int = Field(ge=0)
    """Inference nodes per replica. Set to 0 to skip inference and orchestrator (requires fake data)."""

    num_infer_replicas: int = Field(1, ge=1)
    """Independent inference replicas. Total inference nodes = ``num_infer_nodes * num_infer_replicas``."""

    nodes_per_fsdp_group: int | None = None
    """Training nodes per FSDP island. Auto-sets ``trainer.dp_replicate = num_train_nodes / nodes_per_fsdp_group``."""

    orchestrator_on_inference: bool = False
    """Run the orchestrator on the last inference node instead of trainer rank 0 (frees host RAM on the trainer node)."""

    @property
    def total_infer_nodes(self) -> int:
        return self.num_infer_nodes * self.num_infer_replicas


DeploymentConfig: TypeAlias = Annotated[
    SingleNodeDeploymentConfig | MultiNodeDeploymentConfig, Field(discriminator="type")
]


class RLConfig(BaseConfig):
    trainer: TrainerConfig

    orchestrator: OrchestratorConfig

    inference: InferenceConfig | None = None
    """Inference server configuration. If None, the rl entrypoint will not start an inference server (useful for elastic inference pools or manually started servers)."""

    output_dir: Path = Path("outputs")
    """Output directory. Should be unique per experiment."""

    clean_output_dir: bool = False
    """Delete the output directory before starting training. Required to overwrite an output directory that contains checkpoints from a previous run when not resuming."""

    env_server_base_port: int = 5000
    """Base TCP port for launcher-spawned env servers; the i-th launcher-managed env
    (train then eval) binds ``127.0.0.1:base_port + i``. Envs that set their own
    ``address`` are skipped. Pick a range clear of the inference/weight-broadcast ports."""

    ### Shared configurations

    log: SharedLogConfig = SharedLogConfig()
    """Shared log config. Propagated to trainer and orchestrator."""

    ckpt: SharedCheckpointConfig | None = None
    """Shared checkpoint config. If None, falls back to the sub-config checkpoint settings."""

    wandb: SharedWandbConfig | None = None
    """Shared W&B config. If None, falls back to the sub-config W&B settings."""

    model: SharedModelConfig | None = None
    """Shared model config. If None, falls back to the sub-config model settings."""

    tokenizer: TokenizerConfig | None = None
    """Shared tokenizer config. Propagated to trainer, orchestrator, and inference. If None, each component uses its own tokenizer config (defaulting to model name)."""

    max_steps: int | None = None
    """Shared maximum training steps. If None, falls back to the sub-config ``max_steps``."""

    seq_len: int | None = None
    """Shared sequence length. Propagates to ``trainer.model.seq_len`` and ``orchestrator.seq_len`` only when those values were not explicitly set; explicit per-component values always win."""

    weight_broadcast: SharedWeightBroadcastConfig | None = None

    bench: bool = False
    """Benchmark mode. Sets trainer and orchestrator to benchmark mode and, when set, suffixes the W&B project with ``-bench``."""

    deployment: DeploymentConfig = SingleNodeDeploymentConfig()

    slurm: SlurmConfig | None = None
    """SLURM configuration. If None, runs locally."""

    dry_run: bool = False
    """Only validate and dump resolved configs, then exit early."""

    experimental: RLExperimentalConfig = RLExperimentalConfig()

    ### Validate configs (e.g. raise for unsupported (combinations of) configs)

    @model_validator(mode="after")
    def validate_deployment(self):
        if self.deployment.type == "multi_node":
            if self.slurm is None:
                raise ValueError("Must use SLURM for multi-node deployment.")
            if self.deployment.num_infer_nodes > 0 and not self.inference:
                raise ValueError("Must configure inference when using multi-node deployment with inference nodes.")
            if self.deployment.num_infer_nodes == 0 and self.inference:
                raise ValueError(
                    "Cannot configure inference with num_infer_nodes = 0. "
                    "Either set num_infer_nodes > 0 or remove the inference config."
                )
            if self.deployment.num_infer_nodes == 0 and not self.trainer.data.fake and not self.bench:
                raise ValueError(
                    "Must use fake data (trainer.data.fake or bench = true) when num_infer_nodes = 0, "
                    "since no orchestrator or inference server will be running."
                )
        return self

    @model_validator(mode="after")
    def validate_enough_devices_for_nccl(self):
        if self.deployment.type == "single_node":
            if self.trainer.weight_broadcast.type == "nccl":
                if self.deployment.num_train_gpus + self.deployment.num_infer_gpus < 2:
                    raise ValueError(
                        "NCCL weight broadcast requires at least 2 GPUs to build the broadcast process group."
                    )
        return self

    @model_validator(mode="after")
    def validate_quantize_in_weight_transfer(self):
        if self.weight_broadcast is None or not self.weight_broadcast.quantize_in_weight_transfer:
            return self

        if self.weight_broadcast.type != "nccl":
            raise ValueError("weight_broadcast.quantize_in_weight_transfer requires weight_broadcast.type = 'nccl'.")

        if self.inference is None:
            raise ValueError("weight_broadcast.quantize_in_weight_transfer requires an inference config.")

        if self.trainer.model.impl != "custom":
            raise ValueError("weight_broadcast.quantize_in_weight_transfer requires trainer.model.impl = 'custom'.")

        return self

    @property
    def sdpo_algorithms(self):
        algorithms = [env.algo for env in self.orchestrator.train.env if getattr(env.algo, "type", None) == "sdpo"]
        if algorithms or self.orchestrator.train.env:
            return algorithms
        if getattr(self.orchestrator.algo, "type", None) == "sdpo":
            return [self.orchestrator.algo]
        return []

    @property
    def uses_sdpo_student_support(self) -> bool:
        return any(getattr(algo, "distillation_topk_support", None) == "student" for algo in self.sdpo_algorithms)

    @property
    def uses_sdpo_internal_teacher_regularization(self) -> bool:
        return any(getattr(algo, "teacher_regularization", None) != "live-policy" for algo in self.sdpo_algorithms)

    @model_validator(mode="after")
    def validate_sdpo_sampling_temperature(self):
        sdpo_envs = [env for env in self.orchestrator.train.env if getattr(env.algo, "type", None) == "sdpo"]
        if not sdpo_envs and not self.orchestrator.train.env:
            if getattr(self.orchestrator.algo, "type", None) != "sdpo":
                return self
            temperatures = [("orchestrator.train.sampling", self.orchestrator.train.sampling.temperature)]
        else:
            temperatures = [
                (f"orchestrator.train.env[{env.resolved_name!r}].sampling", env.sampling.temperature)
                for env in sdpo_envs
            ]

        for context, temperature in temperatures:
            if temperature != 1.0:
                raise ValueError(
                    f"sdpo requires {context}.temperature=1.0 because vLLM teacher logprobs are returned "
                    "before sampling-temperature transforms."
                )
        return self

    @model_validator(mode="after")
    def auto_setup_sdpo_runtime(self):
        algorithms = self.sdpo_algorithms
        if not algorithms:
            runtime = self.trainer.sdpo_runtime
            if "teacher_regularization" in runtime.model_fields_set and runtime.teacher_regularization != "live-policy":
                raise ValueError(
                    "trainer.sdpo_runtime.teacher_regularization != 'live-policy' requires an SDPO algorithm."
                )
            if "teacher_update_rate" in runtime.model_fields_set:
                raise ValueError("trainer.sdpo_runtime.teacher_update_rate requires an SDPO algorithm.")
            return self

        teacher_regularization = algorithms[0].teacher_regularization
        teacher_update_rate = algorithms[0].teacher_update_rate
        distillation_topk_support = algorithms[0].distillation_topk_support
        for algo in algorithms[1:]:
            if algo.teacher_regularization != teacher_regularization:
                raise ValueError("all sdpo algorithms in one RL run must use the same teacher_regularization.")
            if algo.teacher_update_rate != teacher_update_rate:
                raise ValueError("all sdpo algorithms in one RL run must use the same teacher_update_rate.")
            if algo.distillation_topk_support != distillation_topk_support:
                raise ValueError("all sdpo algorithms in one RL run must use the same distillation_topk_support.")

        runtime = self.trainer.sdpo_runtime
        if (
            "teacher_regularization" in runtime.model_fields_set
            and runtime.teacher_regularization != teacher_regularization
        ):
            raise ValueError(
                "trainer.sdpo_runtime.teacher_regularization conflicts with orchestrator SDPO teacher_regularization."
            )
        if "teacher_update_rate" in runtime.model_fields_set and runtime.teacher_update_rate != teacher_update_rate:
            raise ValueError(
                "trainer.sdpo_runtime.teacher_update_rate conflicts with orchestrator SDPO teacher_update_rate."
            )

        runtime.teacher_regularization = teacher_regularization
        runtime.teacher_update_rate = teacher_update_rate
        return self

    def _auto_setup_single_node_policy_base_url(self) -> None:
        if self.inference is None or self.deployment.type != "single_node":
            return
        client = self.orchestrator.model.client
        if "base_url" in client.model_fields_set:
            return
        host = self.inference.server.host or "localhost"
        if host in {"0.0.0.0", "::"}:
            host = "localhost"
        client.base_url = [f"http://{host}:{self.inference.server.port}/v1"]

    @model_validator(mode="after")
    def auto_setup_sdpo_teacher_endpoint(self):
        self._auto_setup_single_node_policy_base_url()
        if not self.uses_sdpo_internal_teacher_regularization:
            return self
        if self.orchestrator.sdpo_teacher is not None:
            return self
        if self.deployment.type != "single_node" or self.deployment.num_sdpo_teacher_gpus <= 0:
            return self
        if self.inference is None:
            return self

        host = self.inference.server.host or "localhost"
        if host in {"0.0.0.0", "::"}:
            host = "localhost"
        teacher_port = self.inference.server.port + 1
        if teacher_port > MAX_TCP_PORT:
            raise ValueError(
                "deployment.num_sdpo_teacher_gpus cannot auto-create orchestrator.sdpo_teacher because "
                f"inference.server.port + 1 exceeds the maximum TCP port: {self.inference.server.port} -> "
                f"{teacher_port}."
            )
        teacher = HostedModelConfig(
            name=self.orchestrator.model.name,
            trust_remote_code=self.orchestrator.model.trust_remote_code,
            client={"base_url": [f"http://{host}:{teacher_port}/v1"]},
        )
        teacher.client.extra_headers_from_state.setdefault("X-Session-ID", "trajectory_id")
        self.orchestrator.sdpo_teacher = teacher
        return self

    @model_validator(mode="after")
    def auto_setup_sdpo_inference_max_logprobs(self):
        if not self.sdpo_algorithms or self.inference is None:
            return self

        required_max_logprobs = max(algo.distillation_topk for algo in self.sdpo_algorithms)
        configured_max_logprobs = self.inference.vllm_extra.get("max_logprobs")
        if configured_max_logprobs is None:
            self.inference.vllm_extra["max_logprobs"] = required_max_logprobs
            return self
        if (
            isinstance(configured_max_logprobs, bool)
            or not isinstance(configured_max_logprobs, int)
            or configured_max_logprobs < required_max_logprobs
        ):
            raise ValueError(
                "sdpo algorithms require inference.vllm_extra.max_logprobs to be an integer at least as large "
                f"as the largest distillation_topk ({required_max_logprobs}); got {configured_max_logprobs!r}."
            )
        return self

    @model_validator(mode="after")
    def validate_sdpo_teacher_regularization_runtime_support(self):
        if not self.uses_sdpo_internal_teacher_regularization:
            if self.orchestrator.sdpo_teacher is not None:
                raise ValueError(
                    "orchestrator.sdpo_teacher requires an SDPO algorithm with teacher_regularization != 'live-policy'."
                )
            if self.deployment.type == "single_node" and self.deployment.num_sdpo_teacher_gpus > 0:
                raise ValueError(
                    "deployment.num_sdpo_teacher_gpus requires an SDPO algorithm with "
                    "teacher_regularization != 'live-policy'."
                )
            return self

        if any(getattr(algo, "teacher_regularization", None) == "trust-region" for algo in self.sdpo_algorithms):
            raise ValueError(
                "sdpo teacher_regularization='trust-region' is not launchable yet: the trainer has the "
                "trust-region scoring primitive, but the combined rl runtime does not define a reference "
                "module source to mix with the live student."
            )

        if self.orchestrator.sdpo_teacher is None:
            raise ValueError(
                "sdpo teacher_regularization != 'live-policy' requires "
                "orchestrator.sdpo_teacher to point at a separately updated teacher inference pool."
            )

        weight_broadcast_type = (
            self.weight_broadcast.type if self.weight_broadcast is not None else self.orchestrator.weight_broadcast.type
        )
        if weight_broadcast_type != "filesystem":
            raise ValueError(
                "sdpo teacher_regularization != 'live-policy' currently requires filesystem weight broadcast; "
                "NCCL extra live-inference targets are not wired yet."
            )
        if self.trainer.model.lora is not None:
            raise ValueError(
                "sdpo teacher_regularization != 'live-policy' is not supported with LoRA yet: "
                "filesystem weight broadcast cannot publish extra live teacher targets for adapter-only updates."
            )
        if self.trainer.max_concurrent_runs > 1:
            raise ValueError(
                "sdpo teacher_regularization != 'live-policy' is not supported with max_concurrent_runs > 1 yet: "
                "multi-run checkpointing does not persist the shared SDPO teacher extra state."
            )
        if self.trainer.ckpt is not None and self.trainer.ckpt.weights_only:
            raise ValueError(
                "sdpo teacher_regularization != 'live-policy' requires full trainer checkpoints; "
                "trainer.ckpt.weights_only skips the SDPO teacher extra state needed for resume."
            )

        if self.orchestrator.sdpo_teacher.name != self.orchestrator.model.name:
            raise ValueError(
                "orchestrator.sdpo_teacher.name must match orchestrator.model.name for SDPO "
                "teacher_regularization != 'live-policy' because the teacher endpoint receives "
                "policy-shaped broadcast weights."
            )
        if self.orchestrator.sdpo_teacher.trust_remote_code != self.orchestrator.model.trust_remote_code:
            raise ValueError(
                "orchestrator.sdpo_teacher.trust_remote_code must match orchestrator.model.trust_remote_code "
                "for SDPO teacher_regularization != 'live-policy' because the teacher endpoint receives "
                "policy-shaped broadcast weights."
            )

        policy_client = self.orchestrator.model.client
        teacher_client = self.orchestrator.sdpo_teacher.client
        policy_admin_urls = policy_client.admin_base_url or policy_client.base_url
        teacher_admin_urls = teacher_client.admin_base_url or teacher_client.base_url
        if policy_client.base_url == teacher_client.base_url or policy_admin_urls == teacher_admin_urls:
            raise ValueError(
                "orchestrator.sdpo_teacher must use a separately hosted inference endpoint, not the policy endpoint."
            )
        if self.deployment.type == "single_node" and self.deployment.num_sdpo_teacher_gpus > 0:
            if self.inference is None:
                raise ValueError("deployment.num_sdpo_teacher_gpus requires an [inference] config to clone.")
            if teacher_client.is_elastic:
                raise ValueError("deployment.num_sdpo_teacher_gpus does not support elastic sdpo_teacher clients yet.")
            if len(teacher_client.base_url) != 1:
                raise ValueError(
                    "deployment.num_sdpo_teacher_gpus requires exactly one orchestrator.sdpo_teacher.client.base_url."
                )
        return self

    @model_validator(mode="after")
    def auto_setup_sdpo_student_support_preflight(self):
        if self.uses_sdpo_student_support and "enable_token_export" not in self.trainer.model_fields_set:
            self.trainer.enable_token_export = True
        return self

    @model_validator(mode="after")
    def validate_sdpo_loss_requirements(self):
        if not self.sdpo_algorithms:
            return self

        if not self.trainer.sdpo_loss.full_logit_distillation or self.trainer.sdpo_loss.distillation_topk is None:
            raise ValueError(
                "sdpo algorithms require trainer.sdpo_loss full-logit top-k distillation; "
                "sampled-token and full-vocabulary SDPO are not wired in the combined rl runtime."
            )
        if self.trainer.model.fused_lm_head_token_chunk_size != "disabled":
            raise ValueError(
                "sdpo algorithms require trainer.model.fused_lm_head_token_chunk_size='disabled' "
                "so the trainer forward exposes logits for top-k distillation."
            )
        if self.trainer.model.cp > 1:
            raise ValueError(
                "sdpo algorithms do not support trainer.model.cp > 1 yet because top-k distillation "
                "requires full-sequence logits and support rows."
            )
        if self.trainer.sdpo_loss.rollout_is_batch_normalize:
            raise ValueError(
                "sdpo algorithms do not support trainer.sdpo_loss.rollout_is_batch_normalize=true yet: "
                "the combined rl trainer evaluates SDPO per packed sequence, so batch-normalized rollout-IS "
                "would be sequence-local rather than normalized over the global SDPO component batch."
            )

        for algo in self.sdpo_algorithms:
            if getattr(algo, "distillation_topk", None) != self.trainer.sdpo_loss.distillation_topk:
                raise ValueError(
                    "sdpo algorithms require orchestrator SDPO distillation_topk to match "
                    "trainer.sdpo_loss.distillation_topk."
                )

        return self

    @model_validator(mode="after")
    def validate_sdpo_student_support_preflight_requirements(self):
        if not self.uses_sdpo_student_support:
            return self

        if not self.trainer.enable_token_export:
            raise ValueError(
                "sdpo distillation_topk_support='student' requires trainer.enable_token_export=true "
                "so the preflight trainer pass can export student-selected top-k ids."
            )

        for algo in self.sdpo_algorithms:
            if algo.distillation_topk_support == "student" and algo.distillation_topk > SDPO_MAX_STUDENT_SUPPORT_TOPK:
                raise ValueError(
                    "sdpo distillation_topk_support='student' requires distillation_topk <= "
                    f"{SDPO_MAX_STUDENT_SUPPORT_TOPK}, vLLM's current candidate-token scoring limit."
                )

        return self

    ### Auto-setup shared configs (before sub-config construction)

    @model_validator(mode="before")
    @classmethod
    def auto_setup_shared_configs(cls, data: Any) -> Any:
        """Propagate shared top-level fields into sub-config dicts before sub-configs
        are constructed. See ``validation.propagate_shared_fields`` for the full
        propagation table, transforms, and the mutex rule.
        """
        return propagate_shared_fields(data)

    ### Validate shared configs (after sub-config construction)

    @model_validator(mode="after")
    def validate_shared_configs(self):
        """Validate consistency of shared configs across trainer, orchestrator, and inference."""
        validate_shared_output_dir(self.trainer, self.orchestrator)
        validate_shared_model_name(self.trainer, self.orchestrator, self.inference)
        validate_shared_tokenizer(self.trainer, self.orchestrator, self.inference)
        validate_shared_max_steps(self.trainer, self.orchestrator)
        validate_shared_seq_len(self.trainer, self.orchestrator)
        validate_shared_ckpt_config(self.trainer, self.orchestrator)
        validate_shared_wandb_config(self.trainer, self.orchestrator)
        return self

    @model_validator(mode="after")
    def auto_setup_weight_broadcast(self):
        """Auto-setup shared weight broadcast config for trainer, orchestrator, and inference."""
        if self.weight_broadcast is not None:
            if self.weight_broadcast.type == "nccl":
                inference_world_size = self.inference.parallel.dp * self.inference.parallel.tp if self.inference else 1
                self.trainer.weight_broadcast = TrainerNCCLWeightBroadcastConfig(
                    type=self.weight_broadcast.type,
                    inference_world_size=inference_world_size,
                    port=self.weight_broadcast.port,
                    timeout=self.weight_broadcast.timeout,
                    quantize_in_weight_transfer=self.weight_broadcast.quantize_in_weight_transfer,
                )
                self.orchestrator.weight_broadcast = OrchestratorNCCLWeightBroadcastConfig(
                    type=self.weight_broadcast.type,
                    port=self.weight_broadcast.port,
                    timeout=self.weight_broadcast.timeout,
                    inference_world_size=inference_world_size,
                    quantize_in_weight_transfer=self.weight_broadcast.quantize_in_weight_transfer,
                )
            elif self.weight_broadcast.type == "filesystem":
                self.trainer.weight_broadcast = TrainerFileSystemWeightBroadcastConfig()
                self.orchestrator.weight_broadcast = OrchestratorFileSystemWeightBroadcastConfig()
            if self.inference is not None:
                self.inference.weight_broadcast = InferenceWeightBroadcastConfig(type=self.weight_broadcast.type)

        validate_shared_weight_broadcast(self.trainer, self.orchestrator, self.inference)

        return self

    @model_validator(mode="after")
    def validate_eplb_requires_quantized_weight_transfer(self):
        if self.inference is None or not self.inference.enable_eplb:
            return self

        # TODO(matej): check if weight reloading works itself before supporting EPLB without quantized transfer.
        trainer_weight_broadcast = self.trainer.weight_broadcast
        if trainer_weight_broadcast.type != "nccl" or not trainer_weight_broadcast.quantize_in_weight_transfer:
            raise ValueError(
                "inference.enable_eplb requires weight_broadcast.type = 'nccl' and "
                "weight_broadcast.quantize_in_weight_transfer = true."
            )

        return self

    @model_validator(mode="after")
    def auto_setup_bench(self):
        if self.bench:
            self.trainer.bench = BenchConfig()
            self.orchestrator.bench = True
            self.trainer.data.fake = FakeDataLoaderConfig(
                batch_size=self.orchestrator.batch_size or 32,
            )

        trainer_bench_enabled = self.trainer.bench is not None
        if trainer_bench_enabled != self.orchestrator.bench:
            raise ValueError(
                f"Trainer benchmark mode ({self.trainer.bench}) and orchestrator benchmark mode "
                f"({self.orchestrator.bench}) must match. Use the top-level bench = true to set both."
            )

        return self

    @model_validator(mode="after")
    def auto_setup_lora(self):
        if self.trainer.model.lora is not None:
            if self.trainer.weight_broadcast.type == "nccl":
                raise ValueError("NCCL weight broadcast does not support LoRA yet.")

            if self.orchestrator.model.lora is None:
                from prime_rl.configs.orchestrator import LoRAConfig

                self.orchestrator.model.lora = LoRAConfig()

            if (
                self.orchestrator.model.lora.rank is not None
                and self.orchestrator.model.lora.rank != self.trainer.model.lora.rank
            ):
                raise ValueError(
                    f"orchestrator.model.lora.rank ({self.orchestrator.model.lora.rank}) conflicts with "
                    f"trainer.model.lora.rank ({self.trainer.model.lora.rank}). "
                    f"Remove orchestrator.model.lora.rank to inherit from trainer, or update trainer.model.lora.rank to match."
                )

            if (
                self.orchestrator.model.lora.alpha is not None
                and self.orchestrator.model.lora.alpha != self.trainer.model.lora.alpha
            ):
                raise ValueError(
                    f"orchestrator.model.lora.alpha ({self.orchestrator.model.lora.alpha}) conflicts with "
                    f"trainer.model.lora.alpha ({self.trainer.model.lora.alpha}). "
                    f"Remove orchestrator.model.lora.alpha to inherit from trainer, or update trainer.model.lora.alpha to match."
                )

            if self.orchestrator.model.lora.rank is None:
                self.orchestrator.model.lora.rank = self.trainer.model.lora.rank

            if self.orchestrator.model.lora.alpha is None:
                self.orchestrator.model.lora.alpha = self.trainer.model.lora.alpha

            if self.orchestrator.model.lora.name is None:
                self.orchestrator.model.lora.name = (
                    f"r{self.orchestrator.model.lora.rank}-a{self.orchestrator.model.lora.alpha}"
                )

            if self.inference is not None:
                self.inference.enable_lora = True
                self.inference.max_lora_rank = self.trainer.model.lora.rank
            else:
                warnings.warn(
                    "LoRA is enabled, but inference is not configured. When manually starting the inference server, "
                    "make sure to set --enable_lora and --max-lora-rank.",
                    stacklevel=2,
                )

        return self

    @model_validator(mode="after")
    def auto_setup_router_replay(self):
        if self.trainer.enable_router_replay:
            if self.inference is not None:
                if self.inference.enable_return_routed_experts is False:
                    warnings.warn(
                        "Router replay is enabled, but inference.enable_return_routed_experts is False. Setting to True.",
                        stacklevel=2,
                    )
                self.inference.enable_return_routed_experts = True
            else:
                warnings.warn(
                    "Router replay is enabled, but inference is not configured. When manually starting the inference server, make sure to pass `--enable-return-routed-experts` to the vLLM server.",
                    stacklevel=2,
                )
        return self

    @model_validator(mode="after")
    def validate_llmd_no_routed_experts(self):
        """Reject routed-expert return with the llm-d router (breaks P/D, unverified for multi-node).

        Runs after ``auto_setup_router_replay`` so it also catches the
        ``trainer.enable_router_replay`` path, which sets the inference flag here
        (after InferenceConfig's own validators, which therefore miss it).
        """
        if self.inference is not None and self.inference.enable_return_routed_experts:
            router = getattr(self.inference.deployment, "router", None)
            if router is not None and router.type == "llm-d":
                raise ValueError(
                    "The llm-d router backend does not support routed-expert return "
                    "(inference.enable_return_routed_experts / trainer.enable_router_replay): it "
                    "breaks P/D and is unverified for multi-node. Use router type 'vllm-router' "
                    "for router-replay runs."
                )
        return self

    @model_validator(mode="after")
    def validate_router_replay_without_kv_offload(self):
        if (
            self.trainer.enable_router_replay
            and self.inference is not None
            and self.inference.kv_cache_offload is not None
        ):
            raise ValueError(
                "Router replay with inference.kv_cache_offload is not supported. "
                "External KV cache hits do not carry routed-expert decisions."
            )
        return self

    @model_validator(mode="after")
    def validate_mooncake_offload_requires_slurm(self):
        if (
            self.slurm is None
            and self.inference is not None
            and self.inference.kv_cache_offload is not None
            and self.inference.kv_cache_offload.type == "mooncake"
        ):
            raise ValueError(
                "Mooncake KV offload requires SLURM — the per-node store is launched by the sbatch "
                "template. Use inference.kv_cache_offload.type='native' for local runs."
            )
        return self

    @model_validator(mode="after")
    def auto_setup_deployment(self):
        if self.deployment.type == "single_node":  # single-node
            # set num_train_workers to the number of data replicas
            non_data_parallel_size = self.trainer.model.cp
            if self.deployment.num_train_gpus > 1:
                self.orchestrator.num_train_workers = self.deployment.num_train_gpus // non_data_parallel_size

            # fill up inference capacity with dp ranks
            if self.inference is not None:
                num_infer_gpus = self.deployment.num_infer_gpus
                if num_infer_gpus != self.inference.parallel.dp * self.inference.parallel.tp:
                    if num_infer_gpus % self.inference.parallel.tp != 0:
                        raise ValueError(
                            "deployment.num_infer_gpus must be divisible by inference.parallel.tp "
                            f"({num_infer_gpus} % {self.inference.parallel.tp} != 0)."
                        )
                    self.inference.parallel.dp = num_infer_gpus // self.inference.parallel.tp
                # Ensure api_server_count matches DP so all workers are created.
                # Without this, the NCCL broadcast group expects dp*tp workers
                # but only api_server_count*tp exist, causing a deadlock.
                dp = self.inference.parallel.dp
                if self.inference.api_server_count < dp and not self.inference.enable_lora:
                    self.inference.api_server_count = dp

        elif self.deployment.type == "multi_node":  # multi-node
            self.orchestrator.num_train_workers = self.deployment.num_train_nodes * self.deployment.gpus_per_node

            if self.deployment.nodes_per_fsdp_group is not None:
                if self.deployment.num_train_nodes % self.deployment.nodes_per_fsdp_group != 0:
                    raise ValueError(
                        f"deployment.num_train_nodes ({self.deployment.num_train_nodes}) must be divisible by "
                        f"deployment.nodes_per_fsdp_group ({self.deployment.nodes_per_fsdp_group})"
                    )
                self.trainer.model.dp_replicate = (
                    self.deployment.num_train_nodes // self.deployment.nodes_per_fsdp_group
                )

            if (
                self.inference is not None
                and self.inference.enable_expert_parallel
                and self.inference.deployment.type != "disaggregated"
            ):
                inference_tp = self.inference.parallel.tp
                if self.deployment.gpus_per_node % inference_tp != 0:
                    raise ValueError(
                        "deployment.gpus_per_node must be divisible by inference.parallel.tp "
                        "when inference.enable_expert_parallel is enabled in multi-node deployment."
                    )

                inferred_dp_local = self.deployment.gpus_per_node // inference_tp
                total_infer_gpus = self.deployment.num_infer_nodes * self.deployment.gpus_per_node
                expected_global_world_size = self.inference.parallel.dp * inference_tp
                if expected_global_world_size != total_infer_gpus:
                    raise ValueError(
                        "For multi-node expert parallel inference, inference.parallel.dp * inference.parallel.tp "
                        f"must match total inference GPUs ({total_infer_gpus}), got {expected_global_world_size}."
                    )

                if self.inference.data_parallel_size_local is None:
                    self.inference.data_parallel_size_local = inferred_dp_local
                elif self.inference.data_parallel_size_local != inferred_dp_local:
                    raise ValueError(
                        "inference.data_parallel_size_local must equal deployment.gpus_per_node / inference.parallel.tp "
                        f"({inferred_dp_local}) when inference.enable_expert_parallel is enabled in multi-node deployment."
                    )

                if not self.inference.enable_lora and self.inference.api_server_count == self.inference.parallel.dp:
                    self.inference.api_server_count = inferred_dp_local

            # Auto-infer DP and api_server_count for standard multi-node inference.
            # Without EP, vLLM only creates api_server_count * tp workers per node,
            # not gpus_per_node workers. If DP isn't set, the broadcast group expects
            # more workers than exist, deadlocking NCCL init.
            if (
                self.inference is not None
                and not self.inference.enable_expert_parallel
                and self.inference.deployment.type != "disaggregated"
            ):
                dp_per_node = self.deployment.gpus_per_node // self.inference.parallel.tp
                if self.inference.parallel.dp == 1 and dp_per_node > 1:
                    self.inference.parallel.dp = dp_per_node
                if self.inference.data_parallel_size_local is None and dp_per_node > 1:
                    self.inference.data_parallel_size_local = dp_per_node
                if self.inference.api_server_count == 1 and dp_per_node > 1:
                    self.inference.api_server_count = dp_per_node

            if self.weight_broadcast is not None and self.weight_broadcast.type == "nccl":
                # Every allocated inference GPU is a NCCL rank in the weight broadcast.
                # The external-LB launcher starts dp_per_node (= gpus_per_node / tp)
                # TP-sharded servers per node, i.e. gpus_per_node workers per node, so use
                # the GPU count directly. Deriving it from api_server_count double-counts:
                # api_server_count can resolve to the *global* DP size, making the node
                # factor count twice and NCCL wait for ranks that never connect. Matches
                # the disaggregated path below.
                total_infer_workers = self.deployment.total_infer_nodes * self.deployment.gpus_per_node
                if self.trainer.weight_broadcast.type != "nccl":
                    raise ValueError("weight_broadcast.type='nccl' requires trainer.weight_broadcast.type='nccl'.")
                self.trainer.weight_broadcast.host = "0.0.0.0"
                self.trainer.weight_broadcast.inference_world_size = total_infer_workers
                if self.orchestrator.weight_broadcast.type != "nccl":
                    raise ValueError("weight_broadcast.type='nccl' requires orchestrator.weight_broadcast.type='nccl'.")
                self.orchestrator.weight_broadcast.inference_world_size = total_infer_workers

        return self

    @model_validator(mode="after")
    def auto_setup_disaggregated_inference(self):
        """Auto-setup for disaggregated P/D inference within a multi-node deployment."""
        if self.inference is None or self.inference.deployment.type != "disaggregated":
            return self
        if self.deployment.type != "multi_node":
            return self

        infer_deploy = self.inference.deployment
        expected_infer_nodes = infer_deploy.num_prefill_nodes + infer_deploy.num_decode_nodes
        if self.deployment.num_infer_nodes != expected_infer_nodes:
            raise ValueError(
                f"deployment.num_infer_nodes ({self.deployment.num_infer_nodes}) must equal "
                f"inference.deployment.num_prefill_nodes ({infer_deploy.num_prefill_nodes}) + "
                f"inference.deployment.num_decode_nodes ({infer_deploy.num_decode_nodes}) = {expected_infer_nodes}"
            )

        total_infer_gpus = self.deployment.total_infer_nodes * self.deployment.gpus_per_node
        if "inference_metrics_roles" not in self.orchestrator.model_fields_set:
            # External-LB: one admin client per DP rank, so roles expand per rank
            # (stride = dp_local = gpus_per_node / tp). ADMIN_URLS lists all prefill
            # ranks, then all decode ranks, per replica — match that order.
            stride = self.deployment.gpus_per_node // self.inference.parallel.tp
            role_order = ["prefill"] * (infer_deploy.num_prefill_nodes * stride) + ["decode"] * (
                infer_deploy.num_decode_nodes * stride
            )
            self.orchestrator.inference_metrics_roles = role_order * self.deployment.num_infer_replicas
        if self.weight_broadcast is not None and self.weight_broadcast.type == "nccl":
            if self.trainer.weight_broadcast.type != "nccl":
                raise ValueError("weight_broadcast.type='nccl' requires trainer.weight_broadcast.type='nccl'.")
            self.trainer.weight_broadcast.inference_world_size = total_infer_gpus
            if self.orchestrator.weight_broadcast.type != "nccl":
                raise ValueError("weight_broadcast.type='nccl' requires orchestrator.weight_broadcast.type='nccl'.")
            self.orchestrator.weight_broadcast.inference_world_size = total_infer_gpus

        return self

    @model_validator(mode="after")
    def auto_setup_inference_client(self):
        """Auto-configure the orchestrator policy client from the inference server config.

        Direct single-node runs expose all local DP ranks behind one base URL,
        so pin logical clients with ``X-data-parallel-rank``. Multi-node SLURM
        runs expose a vllm-router URL instead; the router balances across
        per-rank backend URLs and forwards request headers, so the orchestrator
        must not inject a DP-rank header there. Single-node runs always set the
        policy base URL from the launched inference server unless the user set
        it explicitly, so non-default local ports work for policy-sourced
        algorithms too.
        """
        if self.inference is None:
            return self
        client = self.orchestrator.model.client
        if "dp_rank_count" not in client.model_fields_set:
            if self.deployment.type == "multi_node":
                client.dp_rank_count = 1
            else:
                client.dp_rank_count = self.inference.data_parallel_size_local or self.inference.parallel.dp
        if self.deployment.type == "single_node":
            self._auto_setup_single_node_policy_base_url()
        elif not self.orchestrator.any_policy_sourced and "base_url" not in client.model_fields_set:
            host = self.inference.server.host or "localhost"
            port = self.inference.server.port
            client.base_url = [f"http://{host}:{port}/v1"]
        return self

    @model_validator(mode="after")
    def auto_setup_slurm_template(self):
        """Auto-setup the default single-node/multi-node SLURM template if no custom template is provided."""
        if self.slurm is not None and self.slurm.template_path is None:
            templates_dir = find_package_resource("templates")
            if templates_dir is not None:
                if self.deployment.type == "single_node":
                    self.slurm.template_path = templates_dir / "single_node_rl.sbatch.j2"
                else:
                    self.slurm.template_path = templates_dir / "multi_node_rl.sbatch.j2"
        return self

    ### Warnings
