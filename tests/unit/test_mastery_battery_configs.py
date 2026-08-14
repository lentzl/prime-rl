import tomllib
from pathlib import Path

from prime_rl.configs.rl import RLConfig
from prime_rl.utils.config import cli

ROOT = Path(__file__).parents[2]
CONFIG_ROOT = ROOT / "configs" / "debug" / "subagent-communication"


def load_config(name: str) -> dict:
    with (CONFIG_ROOT / name).open("rb") as stream:
        return tomllib.load(stream)


def test_ownership_battery_allows_prime_agent_to_finish() -> None:
    names = (
        "274-qwen35-27b-mastery-ownership-child-ood.toml",
        "275-qwen35-27b-mastery-ownership-coordinator-ood.toml",
    )
    configs = [load_config(name) for name in names]
    limits = [config["env"]["agent"] for config in configs]

    assert all(limit["max_turns"] >= 8 for limit in limits)
    assert all(limit["max_output_tokens"] >= 24_576 for limit in limits)
    assert all(limit["max_total_tokens"] >= 65_536 for limit in limits)


def test_oolong_battery_leaves_room_for_the_agent_feedback_loop() -> None:
    config = load_config("276-qwen35-27b-mastery-oolong-ood.toml")
    taskset = config["env"]["taskset"]
    limits = config["env"]["agent"]

    assert taskset["context_len"] <= 16_384
    assert limits["max_turns"] >= 24
    assert limits["max_input_tokens"] >= 196_608
    assert limits["max_output_tokens"] >= 24_576
    assert limits["max_total_tokens"] >= 221_184


def test_externalization_ramp_separates_aggregation_from_semantic_classification() -> None:
    labeled = load_config("332-qwen35-27b-oolong-labeled-admission.toml")
    recursive = load_config("333-qwen35-27b-oolong-recursive-admission.toml")

    assert labeled["env"]["taskset"] == {
        "id": "oolong-synth-v1",
        "split": "validation",
        "with_labels": True,
        "context_len": 2048,
        "example_offset": 0,
        "num_examples": 4,
    }
    assert recursive["env"]["taskset"] == {
        "id": "oolong-synth-v1",
        "split": "validation",
        "with_labels": False,
        "context_len": 1024,
        "example_offset": 0,
        "num_examples": 4,
    }
    assert recursive["env"]["agent"]["max_turns"] > labeled["env"]["agent"]["max_turns"]


def test_failure_sdpo_smoke_retains_prime_agent_actions_and_writes_portable_weights() -> None:
    config = load_config("344-qwen35-27b-memory-v2-failure-sdpo-smoke.toml")

    assert config["seq_len"] == config["trainer"]["model"]["seq_len"] == 12_288
    assert config["trainer"]["model"]["fused_lm_head_token_chunk_size"] == 512
    assert config["orchestrator"]["batch_size"] == config["deployment"]["num_train_gpus"] == 6
    assert config["orchestrator"]["max_inflight_episodes"] == 2 * config["orchestrator"]["batch_size"]
    assert config["orchestrator"]["train"]["source"][0]["env"]["agent"]["max_turns"] >= 12
    assert config["seq_len"] == (
        config["orchestrator"]["algo"]["max_reprompt_len"]
        + config["orchestrator"]["train"]["sampling"]["max_completion_tokens"]
    )
    assert config["trainer"]["ckpt"]["weights_only"] is True


def test_hybrid_memory_smoke_keeps_native_grpo_and_diagnostic_sdpo_disjoint() -> None:
    config_name = "345-qwen35-27b-memory-v2-hybrid-grpo-sdpo-smoke.toml"
    config = load_config(config_name)
    sources = {source["name"]: source for source in config["orchestrator"]["train"]["source"]}
    native = sources["programmatic-memory-v2-native-grpo"]
    diagnostic = sources["programmatic-memory-v2-diagnostic-sdpo"]

    assert config["orchestrator"]["algo"]["type"] == "grpo"
    assert native["group_size"] == 4
    assert "algo" not in native
    assert native["env"]["taskset"]["causal_feedback_retries"] == 0
    assert native["env"]["taskset"]["record_causal_feedback"] is False
    assert diagnostic["group_size"] == native["group_size"]
    assert config["orchestrator"]["batch_size"] % native["group_size"] == 0
    assert diagnostic["algo"]["type"] == "sdpo"
    assert diagnostic["algo"]["success_reward_threshold"] > 1.0
    assert diagnostic["algo"]["require_explicit_feedback"] is True
    assert diagnostic["algo"]["required_feedback_contract_schema"] == (
        "programmatic-episodic-memory-v2/causal-feedback/v1"
    )
    assert diagnostic["env"]["taskset"]["causal_feedback_retries"] == 0
    assert diagnostic["env"]["taskset"]["record_causal_feedback"] is True
    assert config["seq_len"] == (
        diagnostic["algo"]["max_reprompt_len"]
        + config["orchestrator"]["train"]["sampling"]["max_completion_tokens"]
    )

    resolved = cli(RLConfig, args=["@", str(CONFIG_ROOT / config_name), "--dry-run"])
    assert resolved.trainer.sdpo_loss.enabled is True

    launcher = (ROOT / "scripts" / "run_qwen35_27b_memory_v2_hybrid_grpo_sdpo_smoke.sh").read_text()
    assert "nvidia-smi --query-compute-apps=pid" in launcher
    assert 'rl @ "$config"' in launcher


def test_hybrid_memory_tranche_preserves_every_early_checkpoint() -> None:
    launcher = (
        ROOT / "scripts" / "run_qwen35_27b_memory_v2_hybrid_tranche_v1.sh"
    ).read_text()

    assert "345-qwen35-27b-memory-v2-hybrid-grpo-sdpo-smoke.toml" in launcher
    assert "--max-steps 8" in launcher
    assert "--ckpt.interval 1" in launcher
    assert "--ckpt.keep-last 8" in launcher
    assert "--ckpt.keep-interval 1" in launcher
    assert "nvidia-smi --query-compute-apps=pid" in launcher


def test_full_memory_eval_keeps_every_frozen_task_outside_training() -> None:
    familiar = load_config(
        "347-qwen35-27b-memory-v2-familiar-full-eval.toml"
    )
    ood = load_config("348-qwen35-27b-memory-v2-ood-full-eval.toml")

    assert familiar["num_tasks"] == 300
    assert familiar["num_rollouts"] == 1
    assert familiar["env"]["taskset"]["split"] == "familiar_heldout"
    assert ood["num_tasks"] == 96
    assert ood["num_rollouts"] == 1
    assert ood["env"]["taskset"]["split"] == "semantic_ood"
    for config in (familiar, ood):
        taskset = config["env"]["taskset"]
        assert taskset["condition_on_demonstration"] is False
        assert taskset["causal_feedback_retries"] == 0
        assert taskset["record_causal_feedback"] is True

    launcher = (
        ROOT / "scripts" / "run_qwen35_27b_memory_v2_full_eval.sh"
    ).read_text()
    assert "347-qwen35-27b-memory-v2-familiar-full-eval" in launcher
    assert "348-qwen35-27b-memory-v2-ood-full-eval" in launcher


def test_oolong_scale_admission_increases_decomposition_pressure() -> None:
    names = (
        "334-qwen35-27b-oolong-semantic-4k-admission.toml",
        "335-qwen35-27b-oolong-semantic-8k-admission.toml",
    )
    configs = [load_config(name) for name in names]

    assert [config["env"]["taskset"]["context_len"] for config in configs] == [4096, 8192]
    assert all(config["env"]["taskset"]["with_labels"] is False for config in configs)
    assert all(config["env"]["taskset"]["num_examples"] == 4 for config in configs)
    assert all(config["env"]["max_concurrent_agents"] == 4 for config in configs)
    assert configs[1]["env"]["agent"]["max_turns"] > configs[0]["env"]["agent"]["max_turns"]
    assert configs[1]["env"]["agent"]["max_total_tokens"] > configs[0]["env"]["agent"]["max_total_tokens"]

    launcher = (ROOT / "scripts" / "run_qwen35_27b_oolong_scale_admission.sh").read_text()
    assert all(name.removesuffix(".toml") in launcher for name in names)


def test_fast_mastery_screen_is_compact_frozen_and_disjoint() -> None:
    names = (
        "320-qwen35-27b-mastery-fast-foundations.toml",
        "321-qwen35-27b-mastery-fast-coordination.toml",
        "322-qwen35-27b-mastery-fast-ownership-child.toml",
        "323-qwen35-27b-mastery-fast-ownership-coordinator.toml",
        "324-qwen35-27b-mastery-fast-ownership-child-xml.toml",
        "325-qwen35-27b-mastery-fast-ownership-coordinator-xml.toml",
        "326-qwen35-27b-mastery-fast-oolong.toml",
    )
    configs = [load_config(name) for name in names]
    foundations, coordination, child_tsv, coordinator_tsv, child_xml, coordinator_xml, oolong = configs

    assert sum(config["num_tasks"] for config in configs) == 21
    assert foundations["env"]["taskset"]["instances_per_family"] == 1
    assert coordination["env"]["taskset"]["split"] == "eval"
    assert coordination["env"]["taskset"]["instances_per_template"] == 1
    assert set(coordination["env"]["taskset"]["families"]) == {
        "direct",
        "single",
        "parallel",
        "followup",
        "handshake",
    }
    ownership_pairs = ((child_tsv, coordinator_tsv), (child_xml, coordinator_xml))
    assert all(child["env"]["taskset"]["ownership"] == "child" for child, _ in ownership_pairs)
    assert all(coordinator["env"]["taskset"]["ownership"] == "coordinator" for _, coordinator in ownership_pairs)
    assert all(
        child["env"]["taskset"]["families"] == coordinator["env"]["taskset"]["families"]
        for child, coordinator in ownership_pairs
    )
    assert {
        child["env"]["taskset"]["families"][0] for child, _ in ownership_pairs
    } == {"tsv_score_total", "xml_item_count"}
    assert oolong["env"]["taskset"]["example_offset"] == 32
    assert oolong["env"]["taskset"]["num_examples"] == 2
    assert all(config["env"]["agent"]["harness"]["thinking"] == "high" for config in configs)

    launcher = (ROOT / "scripts" / "run_qwen35_27b_mastery_fast_screen_v1.sh").read_text()
    assert all(name.removesuffix(".toml") in launcher for name in names)
    assert "EVAL_CLIENT_BASE_URL" in launcher
    assert 'args+=(--client.base-url "$client_base_url")' in launcher

    model_launcher = (ROOT / "scripts" / "run_qwen35_27b_mastery_fast_screen_model_v1.sh").read_text()
    assert "refusing to launch while another GPU process is active" in model_launcher
    assert 'for device in "${eval_devices[@]}"' in model_launcher
    assert 'nvidia-smi --id="$device" --query-compute-apps=pid' in model_launcher
    assert '[[ ! -f "$model/STABLE" ]]' in model_launcher
    assert 'max_model_len = 65536' in model_launcher
    assert 'tool_call_parser = "qwen3_coder"' in model_launcher
    assert 'reasoning_parser = "qwen3"' in model_launcher
    assert "run_qwen35_27b_mastery_fast_screen_v1.sh" in model_launcher
    assert "MASTERY_EVAL_DRIVER" in model_launcher
    assert '"$eval_driver" "$model" "$label"' in model_launcher
    assert 'kill "$eval_pid"' in model_launcher
    assert 'wait -n -p completed_pid "$inference_pid" "$eval_pid"' in model_launcher
    assert "inference exited before the mastery evaluation completed" in model_launcher
    assert "EVAL_BACKEND_PORT" in model_launcher
    assert "EVAL_ROUTER_PORT" in model_launcher
    assert "EVAL_DATA_PARALLEL_RPC_PORT" in model_launcher
    assert "data_parallel_rpc_port = $data_parallel_rpc_port" in model_launcher
    assert 'EVAL_CLIENT_BASE_URL="http://127.0.0.1:$backend_port/v1"' in model_launcher

    battery_launcher = (ROOT / "scripts" / "run_qwen35_27b_mastery_battery_v1.sh").read_text()
    assert "EVAL_CLIENT_BASE_URL" in battery_launcher
    assert 'args+=(--client.base-url "$client_base_url")' in battery_launcher
    assert '[[ -n "${MASTERY_CONFIGS:-}" ]]' in battery_launcher


def test_teacher_collections_are_disjoint_and_keep_thinking_enabled() -> None:
    names = (
        "278-qwen35-27b-mastery-child-teacher-collection.toml",
        "279-qwen35-27b-mastery-coordinator-teacher-collection.toml",
    )
    configs = [load_config(name) for name in names]

    assert {config["env"]["taskset"]["ownership"] for config in configs} == {
        "child",
        "coordinator",
    }
    assert all(config["env"]["taskset"]["instance_offset"] >= 20_000 for config in configs)
    assert all(config["env"]["agent"]["harness"]["thinking"] == "high" for config in configs)
    assert all(config["sampling"]["temperature"] == 1.0 for config in configs)
    assert all(config["num_rollouts"] >= 16 for config in configs)


def test_guided_collection_is_complete_thinking_data_not_a_frozen_gate() -> None:
    config = load_config("280-qwen35-27b-mastery-guided-communication-collection.toml")
    taskset = config["env"]["taskset"]

    assert taskset["instruction_level"] == "guided"
    assert taskset["instance_offset"] >= 22_000
    assert set(taskset["families"]) == {
        "direct",
        "single",
        "parallel",
        "followup",
        "handshake",
    }
    assert config["env"]["agent"]["harness"]["thinking"] == "high"
    assert config["env"]["agent"]["harness"]["autonomous"] is True
    assert config["num_rollouts"] >= 4


def test_hard_communication_supplement_targets_fresh_sparse_families() -> None:
    broad = load_config("280-qwen35-27b-mastery-guided-communication-collection.toml")
    supplement = load_config("284-qwen35-27b-mastery-hard-communication-supplement.toml")
    taskset = supplement["env"]["taskset"]

    assert set(taskset["families"]) == {"parallel", "followup", "handshake"}
    assert taskset["instance_offset"] != broad["env"]["taskset"]["instance_offset"]
    assert taskset["seed"] != broad["env"]["taskset"]["seed"]
    assert supplement["num_tasks"] == 12
    assert supplement["num_rollouts"] == 8
    assert supplement["env"]["agent"]["harness"]["thinking"] == "high"
    assert supplement["env"]["agent"]["harness"]["autonomous"] is True


def test_corrected_bidirectional_supplements_are_disjoint_and_conditioned() -> None:
    original = load_config("302-qwen35-27b-mastery-teacher-conditioned-bidirectional.toml")
    handshake = load_config("304-qwen35-27b-mastery-corrected-handshake-supplement.toml")
    followup = load_config("306-qwen35-27b-mastery-corrected-followup-supplement.toml")
    evidence_gated_handshake = load_config("308-qwen35-27b-mastery-evidence-gated-handshake-supplement.toml")
    strict_output_followup = load_config("310-qwen35-27b-mastery-strict-output-followup-supplement.toml")
    quiescent_followup = load_config("312-qwen35-27b-mastery-quiescent-followup-supplement.toml")
    literal_safe_followup = load_config("314-qwen35-27b-mastery-literal-safe-followup-supplement.toml")
    supplements = (
        handshake,
        followup,
        evidence_gated_handshake,
        strict_output_followup,
        quiescent_followup,
        literal_safe_followup,
    )

    assert handshake["env"]["taskset"]["families"] == ["handshake"]
    assert followup["env"]["taskset"]["families"] == ["followup"]
    assert evidence_gated_handshake["env"]["taskset"]["families"] == ["handshake"]
    assert strict_output_followup["env"]["taskset"]["families"] == ["followup"]
    assert quiescent_followup["env"]["taskset"]["families"] == ["followup"]
    assert literal_safe_followup["env"]["taskset"]["families"] == ["followup"]
    assert {config["env"]["taskset"]["instance_offset"] for config in supplements}.isdisjoint(
        {original["env"]["taskset"]["instance_offset"]}
    )
    assert len({config["env"]["taskset"]["seed"] for config in supplements}) == len(supplements)
    assert all(config["env"]["taskset"]["teacher_conditioned"] is True for config in supplements)
    assert all(config["env"]["taskset"]["prompt_contract"] == "explicit_bidirectional_v2" for config in supplements)
    assert all(config["env"]["agent"]["harness"]["thinking"] == "high" for config in supplements)
    assert all(config["env"]["agent"]["harness"]["autonomous"] is True for config in supplements)


def test_child_ownership_supplement_is_targeted_and_disjoint() -> None:
    broad = load_config("278-qwen35-27b-mastery-child-teacher-collection.toml")
    supplement = load_config("286-qwen35-27b-mastery-child-ownership-supplement.toml")
    taskset = supplement["env"]["taskset"]

    assert taskset["ownership"] == "child"
    assert set(taskset["families"]) == {"csv_amount_total", "json_max_value", "sha256_prefix"}
    assert taskset["instance_offset"] != broad["env"]["taskset"]["instance_offset"]
    assert taskset["seed"] != broad["env"]["taskset"]["seed"]
    assert supplement["num_tasks"] == 3
    assert supplement["num_rollouts"] == 16
    assert supplement["env"]["agent"]["harness"]["thinking"] == "high"


def test_guided_coordinator_supplement_is_balanced_and_fresh() -> None:
    native = load_config("279-qwen35-27b-mastery-coordinator-teacher-collection.toml")
    supplement = load_config("288-qwen35-27b-mastery-guided-coordinator-ownership-supplement.toml")
    taskset = supplement["env"]["taskset"]

    assert taskset["ownership"] == "coordinator"
    assert taskset["instruction_level"] == "guided"
    assert taskset["instance_offset"] != native["env"]["taskset"]["instance_offset"]
    assert taskset["seed"] != native["env"]["taskset"]["seed"]
    assert supplement["num_tasks"] == 8
    assert supplement["num_rollouts"] == 8
    assert supplement["env"]["agent"]["harness"]["thinking"] == "high"


def test_teacher_bootstraps_adapt_qwen35_linear_attention_and_preserve_thinking() -> None:
    names = (
        "281-qwen35-27b-prime-agent-teacher-bootstrap-r64.toml",
        "282-qwen35-27b-prime-agent-teacher-bootstrap-r128.toml",
    )
    configs = [load_config(name) for name in names]

    assert [config["model"]["lora"]["rank"] for config in configs] == [64, 128]
    assert all(config["deployment"]["num_gpus"] == 4 for config in configs)
    assert all(config["renderer"]["enable_thinking"] is True for config in configs)
    assert all(config["ckpt"]["interval"] <= 2 for config in configs)
    assert all(
        any("linear_attn" in target for target in config["model"]["lora"]["target_modules"]) for config in configs
    )


def test_online_teacher_bootstrap_reserves_inference_and_uses_frozen_gates() -> None:
    config = load_config("283-qwen35-27b-prime-agent-teacher-bootstrap-online.toml")
    sources = {source["name"]: source for source in config["eval"]["source"]}

    assert config["deployment"] == {
        "gpus_per_node": 4,
        "num_gpus": 2,
        "num_infer_gpus": 2,
    }
    assert config["renderer"]["enable_thinking"] is True
    assert config["eval"]["skip_first_step"] is False
    assert set(sources) == {
        "ownership-child-heldout",
        "ownership-coordinator-heldout",
        "communication-heldout",
        "prime-agent-foundations",
        "oolong-externalization",
    }
    assert sources["ownership-child-heldout"]["interval"] == 4
    assert sources["ownership-coordinator-heldout"]["interval"] == 4
    assert sources["communication-heldout"]["interval"] == 8
    assert sources["prime-agent-foundations"]["interval"] == config["max_steps"]
    assert sources["oolong-externalization"]["interval"] == config["max_steps"]
    assert config["eval"]["client"]["base_url"].endswith(":8100/v1")


def test_broad_mastery_grpo_is_on_policy_disjoint_and_strictly_gated() -> None:
    config = load_config("316-qwen35-27b-prime-agent-mastery-grpo.toml")
    train = {source["name"]: source for source in config["orchestrator"]["train"]["source"]}
    evaluation = {source["name"]: source for source in config["orchestrator"]["eval"]["source"]}

    assert config["max_steps"] == 8
    assert config["deployment"] == {
        "gpus_per_node": 4,
        "num_train_gpus": 2,
        "num_infer_gpus": 2,
    }
    assert config["trainer"]["model"]["lora"]["rank"] == 64
    assert config["orchestrator"]["max_off_policy_steps"] == 0
    assert config["inference"]["router"] == "None"
    assert set(train) == {
        "mastery-foundations-train",
        "mastery-ownership-child-train",
        "mastery-ownership-coordinator-train",
        "mastery-communication-train",
        "mastery-externalization-train",
    }
    assert set(evaluation) == {
        "ownership-child-heldout",
        "ownership-coordinator-heldout",
        "communication-heldout",
        "prime-agent-foundations",
        "oolong-externalization",
    }
    for name in ("mastery-ownership-child-train", "mastery-ownership-coordinator-train"):
        assert train[name]["env"]["taskset"]["task"]["reward_shape"] == "dense"
        assert train[name]["env"]["taskset"]["instance_offset"] != evaluation[
            name.removeprefix("mastery-").removesuffix("-train") + "-heldout"
        ]["env"]["taskset"]["instance_offset"]
    assert "reward_shape" not in evaluation["ownership-child-heldout"]["env"]["taskset"]["task"]
    assert train["mastery-foundations-train"]["env"]["taskset"]["instance_offset"] > 0
    externalization = train["mastery-externalization-train"]["env"]["taskset"]
    assert externalization["example_offset"] == 8
    assert externalization["num_examples"] == 24
    assert evaluation["oolong-externalization"]["num_examples"] == 8


def test_broad_mastery_grpo_resolves_to_bare_vllm() -> None:
    config = cli(
        RLConfig,
        args=[
            "@",
            str(CONFIG_ROOT / "316-qwen35-27b-prime-agent-mastery-grpo.toml"),
            "--dry-run",
        ],
    )

    assert config.inference is not None
    assert config.inference.router is None
    assert config.inference.server.port == 8000
    assert config.orchestrator.model.client.base_url == "http://localhost:8000/v1"


def test_mastery_grpo_launcher_uses_native_rl_entrypoint() -> None:
    launcher = (ROOT / "scripts" / "run_prime_agent_mastery_grpo.sh").read_text()

    assert "nvidia-smi --query-compute-apps=pid" in launcher
    assert "source .env" in launcher
    assert "backup_prime_agent_adapters.py" in launcher
    assert 'rl @ "$config"' in launcher


def test_full_weight_smoke_reserves_dense_reload_headroom() -> None:
    launcher = (ROOT / "scripts" / "run_prime_agent_full_weight_smoke.sh").read_text()

    assert "nvidia-smi --query-compute-apps=pid" in launcher
    assert "export HF_TOKEN=${HF_TOKEN:-${HF_KEY:-}}" in launcher
    assert "--deployment.gpus-per-node 8" in launcher
    assert "--deployment.num-train-gpus 6" in launcher
    assert "--deployment.num-infer-gpus 2" in launcher
    assert "--inference.vllm.tensor-parallel-size 2" in launcher
    assert "--inference.vllm.gpu-memory-utilization 0.80" in launcher
    assert "--inference.vllm.max-num-seqs 4" in launcher
    assert "--inference.vllm.enforce-eager true" in launcher
    assert "--trainer.model.lora None" in launcher
    assert "--trainer.ckpt.weights.save-adapter-separately false" in launcher
    assert "--orchestrator.eval None" in launcher


def test_balanced_full_weight_launcher_restores_multi_group_updates() -> None:
    launcher = (ROOT / "scripts" / "run_prime_agent_full_weight_balanced_mastery.sh").read_text()

    assert "328-qwen35-27b-full-weight-balanced-r2" in launcher
    assert "refusing to overwrite balanced-mastery output" in launcher
    assert "--deployment.num-train-gpus 6" in launcher
    assert "--deployment.num-infer-gpus 2" in launcher
    assert "--trainer.model.lora None" in launcher
    assert "--orchestrator.batch-size 24" in launcher
    assert "--orchestrator.oversampling-factor None" in launcher
    assert "--orchestrator.max-inflight-episodes 8" in launcher
    assert "--ckpt.interval 2" in launcher


def test_balanced_full_weight_resume_restores_optimizer_and_policy_version() -> None:
    launcher = (ROOT / "scripts" / "resume_prime_agent_full_weight_balanced_mastery.sh").read_text()

    assert "checkpoints/step_${resume_step}/trainer/.metadata" in launcher
    assert "checkpoints/step_${resume_step}/orchestrator/progress.pt" in launcher
    assert "max_steps <= resume_step" in launcher
    assert '--ckpt.resume-step "$resume_step"' in launcher
    assert '--max-steps "$max_steps"' in launcher
    assert "--orchestrator.max-inflight-episodes 8" in launcher
    assert "refusing to resume while another GPU process is active" in launcher


def test_qwen35_9b_opd_smoke_uses_direct_frozen_teacher_distillation() -> None:
    config = load_config("329-qwen35-9b-prime-agent-mastery-opd.toml")
    sources = {source["name"]: source for source in config["orchestrator"]["train"]["source"]}

    assert config["model"] == {
        "name": "Qwen/Qwen3.5-9B",
        "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "vlm": {
            "vision_encoder_attr": "model.visual",
            "language_model_attr": "model.language_model",
            "freeze_vision_encoder": True,
        },
    }
    assert config["deployment"] == {
        "gpus_per_node": 6,
        "num_train_gpus": 5,
        "num_infer_gpus": 1,
    }
    assert config["orchestrator"]["algo"]["type"] == "opd"
    assert config["orchestrator"]["algo"]["teacher"]["base_url"] == "http://localhost:8001/v1"
    assert config["orchestrator"]["group_size"] == 1
    assert config["orchestrator"]["renderer"]["enable_thinking"] is True
    assert set(sources) == {
        "mastery-foundations-opd",
        "mastery-ownership-child-opd",
        "mastery-ownership-coordinator-opd",
        "mastery-communication-opd",
        "mastery-externalization-opd",
    }
    assert all(source["group_size"] == 1 for source in sources.values())
    assert "pre_batch_filters" not in config["orchestrator"]

    launcher = (ROOT / "scripts" / "run_qwen35_9b_prime_agent_opd_smoke.sh").read_text()
    assert "PRIME_AGENT_OPD_TEACHER:?" in launcher
    assert '[[ ! -f "$teacher/STABLE" ]]' in launcher
    assert "CUDA_VISIBLE_DEVICES=6,7 inference" in launcher
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 rl" in launcher
    assert '--orchestrator.algo.teacher.name "$teacher"' in launcher
