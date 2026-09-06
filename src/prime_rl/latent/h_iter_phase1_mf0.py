from __future__ import annotations

import hashlib
import json
import math
from typing import Any

MECHANISM = "q35-2b-h-iter-phase1-train-calibration-v1"
RUN_ID = "h-iter-phase1-train-calibration-prereg-run1"
OUTPUT_ROOT = "/home/ubuntu/rlm/outputs/q35-2b-h-iter-phase1-train-calibration-prereg-run1"
ARTIFACT_DIR = "experiments/qwen35-2b-latent-workspace-v1/h-iter-phase1-train-calibration-v1"
PHASE0_DIR = "experiments/qwen35-2b-latent-workspace-v1/h-iter-phase0-generator-locality-v1"
TRAIN_BANK_PATH = f"{PHASE0_DIR}/train-bank.json"
TRAIN_BANK_FILE_SHA256 = "12f6f9a000c1fa13380b7d58d302ad9d2f75ebc5eeb1922d1a088f5fec4bdbfd"
TRAIN_BANK_INTERNAL_SHA256 = "1dd675f276cbe3164ed03901c6036761cee12f39c0571d3edc40d0d37fa4aca2"
INIT_PAYLOAD = f"{MECHANISM}:init"
INIT_SHA256 = "f42df2afcbe75e6962c9abd4aac781c61f202f61274935ba1761f89f8492034d"
INIT_SEED = 17594986156060532329
SCHEDULE_PAYLOAD = f"{MECHANISM}:schedule"
SCHEDULE_SHA256 = "f5c75e0a2ce0067d2c4ecc421e3350746ab384237a704d3a99aa370a8d7d149e"
SCHEDULE_SEED = 17710227457453327997
SYNTHETIC_PAYLOAD = f"{MECHANISM}:mf0-synthetic-gradient"
SYNTHETIC_SHA256 = "466d7ebafc8568cc1d136921f152c8b2021f0f4b9df66422ed1b80966470e2bf"
SYNTHETIC_FEATURE_SHA256 = "6dc26d44d36f9bac7401bdfb073899a6cc87f04d344098ef6ac121f13fb4967f"
ARMS = ["STATIC", "FFN", "FIXED_T4", "RESET_K", "REC_K"]
ACTIONS = ["ACT_Z1", "ACT_K4", "ACT_M7", "ACT_Q9"]
ASSET_NAMES = [
    "phase0-evidence-binding.json",
    "train-partition.json",
    "cap0-probe-selection.json",
    "training-schedule.json",
    "candidate-module-contract.json",
    "capture-contract.json",
    "metric-gate-contract.json",
    "threshold-builder-contract.json",
]
MEMORY_LABELS = [
    "runtime_verified", "full_freeze_preflight_verified", "phase0_evidence_binding_validated",
    "train_bank_validated", "train_partition_validated", "cap0_probe_selection_validated",
    "training_schedule_validated", "candidate_contract_validated",
    "candidate_cpu_instantiation_validated", "candidate_synthetic_gradients_validated",
    "capture_contract_validated", "metric_gate_contract_validated",
    "threshold_builder_contract_validated", "safety_resource_contract_validated",
    "tamper_audit_validated", "full_freeze_postflight_validated", "proof_prewrite_ready",
]
TAMPERS = [
    "source_train_bank_file_hash_changed", "source_train_bank_internal_hash_changed",
    "nontrain_split_added", "fit_replicate_rule_changed", "calibration_replicate_rule_changed",
    "fit_cal_overlap", "train_row_missing", "train_row_duplicated", "fit_order_changed",
    "calibration_order_changed", "cell_balance_changed", "cap_probe_depth_missing",
    "cap_probe_not_fit", "cap_probe_not_lexicographic_minimum", "cap_probe_order_changed",
    "epoch_count_changed", "depth_batch_order_changed", "arm_update_order_changed",
    "batch_row_order_changed", "optimizer_update_budget_changed", "init_payload_changed",
    "module_dimension_changed", "module_arm_semantics_changed", "forbidden_graph_input_added",
    "capture_render_contract_changed", "cache_guard_weakened", "protected_model_update_allowed",
    "t0_gate_changed", "threshold_formula_changed", "validation_path_added", "heldout_path_added",
    "resource_or_timeout_changed", "proof_status_changed", "proof_self_hash_changed",
]


class MF0ContractError(RuntimeError):
    pass


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def finish(value: dict[str, Any], field: str) -> dict[str, Any]:
    value[field] = sha256_bytes(canonical_json({key: item for key, item in value.items() if key != field}))
    return value


def row_ref(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in (
        "depth", "action_index", "replicate", "row_id", "order_key_sha256",
        "row_sha256", "receiver_input_sha256",
    )}


def build_phase0_binding() -> dict[str, Any]:
    return finish({
        "schema_version": "prime-rl/latent-h-iter-phase1-phase0-binding/v1",
        "status": "phase0_evidence_bound",
        "phase0_evidence_freeze_commit": "e3485efc33c815beb2691d4d6149e580e9e81a7f",
        "phase0_evidence_commit": "7ee3e68554225e83fbae500e7bd5d17ff6c67735",
        "phase0_manifest_path": f"{PHASE0_DIR}/deterministic-terminal-recovery-run1-evidence-manifest.json",
        "phase0_manifest_file_sha256": "cdc5af8cd0fc1eea3f7031d8321c0ca9a7833d77c2e541a576d08bf263e03faa",
        "phase0_manifest_internal_sha256": "8d62f1a199e7e3d4195ec985c0743212f568485a4d36968a321422d2ebc1f903",
        "phase0_proof_file_sha256": "cf2a51decbb1e050f54896de82813d149f726f7c7bc88f5e41626e9a7fb5018d",
        "phase0_proof_internal_sha256": "bc76d98d9cb4e2881eb76a055032a696d75b21e4da76868da5be24975089b88c",
        "claim_boundary": {
            "phase0_generator_locality_validated": True,
            "deterministic_recovery_not_independent_scientific_replication": True,
            "phase0_metrics_for_phase1_gate": False,
            "phase1_authorized": False,
        },
        "binding_sha256": "",
    }, "binding_sha256")


def build_partition(bank: dict[str, Any]) -> dict[str, Any]:
    fit = sorted((row_ref(row) for row in bank["rows"] if row["replicate"] <= 3), key=lambda r: (r["depth"], r["action_index"], r["replicate"], r["row_id"]))
    cal = sorted((row_ref(row) for row in bank["rows"] if row["replicate"] >= 4), key=lambda r: (r["depth"], r["action_index"], r["replicate"], r["row_id"]))
    return finish({
        "schema_version": "prime-rl/latent-h-iter-phase1-train-partition/v1",
        "status": "train_partition_preregistered", "mechanism": MECHANISM,
        "source_train_bank_path": TRAIN_BANK_PATH,
        "source_train_bank_file_sha256": TRAIN_BANK_FILE_SHA256,
        "source_train_bank_internal_sha256": TRAIN_BANK_INTERNAL_SHA256,
        "source_row_count": 96,
        "fit_rule": "split=train and replicate in [0,1,2,3]",
        "calibration_rule": "split=train and replicate in [4,5]",
        "fit_order": ["depth", "action_index", "replicate", "row_id"],
        "calibration_order": ["depth", "action_index", "replicate", "row_id"],
        "fit_rows": fit, "calibration_rows": cal,
        "counts": {"fit": 64, "calibration": 32, "fit_per_depth": 16, "calibration_per_depth": 8, "fit_per_action": 16, "calibration_per_action": 8},
        "disjointness": {"row_id_intersection": [], "complete_train_union": True},
        "partition_sha256": "",
    }, "partition_sha256")


def build_cap0(partition: dict[str, Any]) -> dict[str, Any]:
    probes = []
    for depth in range(1, 5):
        candidates = [row for row in partition["fit_rows"] if row["depth"] == depth]
        chosen = min(candidates, key=lambda row: (row["order_key_sha256"], row["row_id"]))
        probes.append({"probe_index": depth, **chosen})
    return finish({
        "schema_version": "prime-rl/latent-h-iter-phase1-cap0-probe-selection/v1",
        "status": "cap0_probe_selection_preregistered", "mechanism": MECHANISM,
        "source_partition_sha256": partition["partition_sha256"],
        "selection_rule": "for each depth 1..4 choose lexicographic minimum (order_key_sha256,row_id) among fit rows",
        "ordered_probes": probes,
        "counts": {"probes": 4, "per_depth": 1, "tokenizer_calls": 4, "model_forwards": 8, "sequences": 192, "ordered_cases": [{"case_index": 2 * (depth - 1) + repeat - 1, "depth": depth, "repeat": repeat} for depth in range(1, 5) for repeat in (1, 2)]},
        "selection_sha256": "",
    }, "selection_sha256")


def build_schedule(partition: dict[str, Any]) -> dict[str, Any]:
    fit_by_depth = {depth: [row["row_id"] for row in partition["fit_rows"] if row["depth"] == depth] for depth in range(1, 5)}
    cal_by_depth = {depth: [row["row_id"] for row in partition["calibration_rows"] if row["depth"] == depth] for depth in range(1, 5)}
    probe_row = next(row["row_id"] for row in partition["fit_rows"] if row["depth"] == 4 and row["action_index"] == 0 and row["replicate"] == 0)
    preconnect = [{"operation_index": arm_index, "phase": "PRECONNECT", "arm": arm, "depth": 4, "row_ids": [probe_row], "gradient": True, "optimizer_step": False} for arm_index, arm in enumerate(ARMS)]
    precal = []
    for arm_index, arm in enumerate(ARMS):
        for depth in range(1, 5):
            precal.append({"operation_index": 5 + 4 * arm_index + depth - 1, "phase": "PRECAL", "arm": arm, "depth": depth, "row_ids": cal_by_depth[depth], "gradient": False, "optimizer_step": False})
    train = []
    for epoch in range(16):
        for depth in range(1, 5):
            for arm_index, arm in enumerate(ARMS):
                operation_index = 25 + 20 * epoch + 5 * (depth - 1) + arm_index
                train.append({"operation_index": operation_index, "update_index": operation_index - 24, "phase": "TRAIN", "epoch": epoch, "depth": depth, "arm": arm, "row_ids": fit_by_depth[depth], "gradient": True, "optimizer_step": True})
    def evaluation(phase: str, offset: int, rows: dict[int, list[str]]) -> list[dict[str, Any]]:
        return [{"operation_index": offset + 4 * arm_index + depth - 1, "phase": phase, "arm": arm, "depth": depth, "row_ids": rows[depth], "gradient": False, "optimizer_step": False} for arm_index, arm in enumerate(ARMS) for depth in range(1, 5)]
    postcal = evaluation("POSTCAL", 345, cal_by_depth)
    postfit = evaluation("POSTFIT", 365, fit_by_depth)
    return finish({
        "schema_version": "prime-rl/latent-h-iter-phase1-training-schedule/v1",
        "status": "training_schedule_preregistered", "mechanism": MECHANISM,
        "source_partition_sha256": partition["partition_sha256"], "epochs": 16,
        "depth_order": [1, 2, 3, 4], "arm_order": ARMS,
        "batches": {"preconnect": preconnect, "precal": precal, "train": train, "postcal": postcal, "postfit": postfit},
        "optimizer_updates_per_arm": 64, "total_optimizer_updates": 320,
        "expected_call_counts": {
            "cap0_separate": {"tokenizer_calls": 4, "model_forwards": 8, "sequences": 192},
            "preconnect": {"forwards": 5, "backwards": 5, "optimizer_steps": 0, "cell_calls": 13},
            "precal": {"forwards": 20, "backwards": 0, "optimizer_steps": 0, "cell_calls": 40, "row_presentations": 160},
            "train": {"forwards": 320, "backwards": 320, "optimizer_steps": 320, "cell_calls": 640, "row_presentations": 5120},
            "postcal": {"forwards": 20, "backwards": 0, "optimizer_steps": 0, "cell_calls": 40, "row_presentations": 160},
            "postfit": {"forwards": 20, "backwards": 0, "optimizer_steps": 0, "cell_calls": 40, "row_presentations": 320},
            "sidecar_total": {"forwards": 385, "backwards": 325, "optimizer_steps": 320, "cell_calls": 773},
            "per_arm": {"forwards": 77, "backwards": 65, "optimizer_steps": 64, "cell_calls": {"STATIC": 0, "FFN": 77, "FIXED_T4": 308, "RESET_K": 194, "REC_K": 194}},
        },
        "schedule_sha256": "",
    }, "schedule_sha256")


def build_candidate_contract() -> dict[str, Any]:
    return finish({
        "schema_version": "prime-rl/latent-h-iter-phase1-candidate-contract/v1",
        "status": "candidate_contract_preregistered", "mechanism": MECHANISM,
        "input_dim": 2048, "state_dim": 128, "action_order": ACTIONS,
        "module_spec": {
            "codec": ["LayerNorm(2048,eps=1e-5,elementwise_affine=True)", "Linear(2048,128,bias=True)", "GELU(approximate=none)"],
            "cell": ["LayerNorm_self(128,eps=1e-5,elementwise_affine=True)", "LayerNorm_message(128,eps=1e-5,elementwise_affine=True)", "Linear(256,256,bias=True)", "GELU(approximate=none)", "Linear(256,128,bias=True)", "residual_post_LayerNorm(128,eps=1e-5,elementwise_affine=True)"],
            "readout": "Linear(128,4,bias=True) on indexed start vector only",
        },
        "arm_order": ARMS,
        "arm_semantics": {"STATIC": "z", "FFN": "C(z,zeros) once without adjacency", "FIXED_T4": "persistent C exactly 4 times", "RESET_K": "K calls each on original z and z[successor], discard prior", "REC_K": "persistent C exactly K times"},
        "initialization": {"payload": INIT_PAYLOAD, "payload_sha256": INIT_SHA256, "seed_u64_be": INIT_SEED, "one_global_cpu_seed": True, "identical_initial_state_all_arms": True},
        "precision": {"parameters": "torch.float32_cpu", "dropout": False, "tf32": False},
        "determinism": {
            "shuffle": False,
            "sampling": False,
            "other_seeds": 0,
            "tf32": False,
            "objective": "mean 4-class cross entropy on final FP32 start-node logits",
            "optimizer": {"type": "AdamW", "lr": 0.001, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.01, "gradient_clip_global_norm": 1.0, "zero_grad_set_to_none": True},
            "training_order": "epoch 0..15, depth 1..4, arm order; one forward/backward/clip/step per batch",
        },
        "parameter_parity": {"names_shapes_counts_bytes_equal": True},
        "gradient_probe": {
            "mf0_synthetic": {"synthetic_payload": SYNTHETIC_PAYLOAD, "payload_sha256": SYNTHETIC_SHA256, "feature_sha256": SYNTHETIC_FEATURE_SHA256, "forwards": 5, "backwards": 5, "optimizer_constructed": False, "target_action_index": 0, "depth": 4},
            "t0_preconnect": {"row_rule": "unique fit row depth=4,action_index=0,replicate=0", "arms": 5, "forwards": 5, "backwards": 5, "optimizer_steps": 0, "loss": "actual 4-class mean cross entropy", "codec_readout_gradient_l2_min": 1e-8, "nonstatic_cell_gradient_l2_min": 1e-8, "static_cell_gradients": None, "state_unchanged": True, "gradients_cleared": True},
        },
        "forbidden_inputs": ["depth_or_K_scalar", "global_pool", "dense_adjacency", "answer", "supervision", "auxiliary_targets", "intermediate_teacher"],
        "contract_sha256": "",
    }, "contract_sha256")


def build_capture_contract(cap0: dict[str, Any]) -> dict[str, Any]:
    return finish({
        "schema_version": "prime-rl/latent-h-iter-phase1-capture-contract/v1",
        "status": "capture_contract_preregistered", "mechanism": MECHANISM,
        "source_model": {"model": "e33", "checkpoint_sha256": "e33bd4cdbfd92eb22844dbbde2764aa7fa00e1cd25ca7045f91ce22210499e47", "frozen": True, "dtype": "bfloat16", "attention": "eager", "worker_h176_checkpoint_sha256": "77980e247bbccd6463ddda02cd42d2c357e15f8ec1ad0ea84627e008a8674a1e", "worker_h176_loaded": False},
        "tokenizer_contract": {"input": "exact 68-byte local_text only", "add_special_tokens": True, "padding": "max_length", "padding_side": "left", "max_length": 128, "truncation": False, "return_tensors": "pt", "unpadded_range": [1, 128]},
        "render_contract": {"graph_batch": 24, "node_order": "bank receiver_input.nodes order", "answer_or_supervision_in_input": False},
        "model_call_contract": {"eval": True, "inference_mode": True, "use_cache": False, "output_hidden_states": True, "return_dict": True, "logits_to_keep": 1, "generation": False, "loss": False, "full_vocab_persisted": False},
        "hidden_selection": {"tensor": "hidden_states[-1][:,-1,:]", "shape": [24, 2048], "detached_cpu_dtype": "bfloat16", "sidecar_cast": "float32", "pkv": None},
        "cap0_schedule": {"selection_sha256": cap0["selection_sha256"], "graphs": 4, "tokenizer_calls": 4, "model_forwards": 8, "sequences": 192, "repeat_input_and_hidden_bitwise": True, "all_finite": True, "not_all_node_identical": True},
        "t0_schedule": {"graphs": 96, "tokenizer_calls": 96, "model_forwards": 96, "sequences": 2304, "capture_once": True, "features_memory_only": True},
        "cache_guard": {
            "cap0_checks": 18,
            "t0_checks": 194,
            "dynamic_cache_negative_trips": 1,
            "actual_allocations": 0,
            "pkv_none": True,
            "configuration_use_cache_false_restored": True,
            "class_closure": [
                {"fqcn": name, "module_path": path, "module_sha256": digest, "distribution": distribution}
                for name, path, digest, distribution in [
                    ("fla.models.utils.Cache", "/home/ubuntu/rlm/prime-rl/.venv/lib/python3.12/site-packages/fla/models/utils.py", "3785d027727370b6eb8da96050109a249254c90caa12a3531a4034cc79f256a1", "flash-linear-attention==0.5.2"),
                    ("fla.models.utils.FLACache", "/home/ubuntu/rlm/prime-rl/.venv/lib/python3.12/site-packages/fla/models/utils.py", "3785d027727370b6eb8da96050109a249254c90caa12a3531a4034cc79f256a1", "flash-linear-attention==0.5.2"),
                    ("fla.models.utils.LegacyFLACache", "/home/ubuntu/rlm/prime-rl/.venv/lib/python3.12/site-packages/fla/models/utils.py", "3785d027727370b6eb8da96050109a249254c90caa12a3531a4034cc79f256a1", "flash-linear-attention==0.5.2"),
                    *[(f"transformers.cache_utils.{name}", "/home/ubuntu/rlm/prime-rl/.venv/lib/python3.12/site-packages/transformers/cache_utils.py", "a51d2cd525f4458941a20cb5b3b8a8d989d8ca5bcd451855a27bb41743fed586", "transformers==5.6.2") for name in ("Cache", "DynamicCache", "EncoderDecoderCache", "QuantizedCache", "StaticCache")],
                ]
            ],
        },
        "protected_state": {"e33_disk_config_tokenizer_state_metadata_exact": True, "e33_grads_none": True, "h176_unloaded": True, "no_update": True},
        "resource_bounds": {"gpu": "one_A6000", "minimum_free_device_gib": 44, "maximum_allocated_or_reserved_gib": 40, "minimum_ram_gib": 64, "minimum_disk_gib": 16, "artifact_mib": {"cap0": 16, "t0": 32}, "cap0_seconds": {"compute": 3000, "audit": 240, "failure": 180, "terminal": 60, "outer": 3600}, "t0_seconds": {"compute": 18000, "audit": 1200, "failure": 1200, "terminal": 300, "startup": 600, "postexit": 300, "outer": 21600}, "alarm_safety_margin_seconds": 1},
        "exposure_boundary": {"train_only": True, "validation_opens": 0, "heldout_opens": 0, "no_candidate_reuse": True},
        "contract_sha256": "",
    }, "contract_sha256")


def build_metric_contract() -> dict[str, Any]:
    return finish({
        "schema_version": "prime-rl/latent-h-iter-phase1-metric-gate-contract/v1",
        "status": "metric_gate_contract_preregistered", "mechanism": MECHANISM,
        "metric_formulas": {"nll": "-log_softmax(logits.double(),dim=-1)[action_index]", "prediction": "argmax with lowest-index tie", "mean": "math.fsum in frozen order divided by count"},
        "aggregation_order": ["depth", "action_index", "replicate", "row_id"],
        "pre_post_schedule": ["PRE_cal_before_connectivity_and_training", "POST_cal_after_64_updates_per_arm", "POST_fit_after_training"],
        "t0_go_gates": {"rec_postfit_accuracy_min": [48, 64], "rec_postcal_accuracy_min": [20, 32], "rec_each_depth_min": [4, 8], "rec_each_action_recall_min": [3, 8], "rec_postcal_nll_le_precal_factor": 0.75, "rec_minus_reset_accuracy_min": [4, 32], "reset_nll_minus_rec_min": 0.05, "rec_minus_max_static_ffn_accuracy_min": [2, 32], "min_static_ffn_nll_minus_rec_min": 0.02},
        "complete_counts": {"calibration": 32, "fit": 64, "arms": 5},
        "stop_semantics": {"valid_miss": "h_iter_phase1_train_calibration_stop", "retry": False, "more_steps": False, "new_seed": False, "gate_relaxation": False, "validation_open": False},
        "descriptive_metrics": ["FIXED_T4 comparisons", "overall", "by_depth", "by_action"],
        "contract_sha256": "",
    }, "contract_sha256")


def build_threshold_contract() -> dict[str, Any]:
    return finish({
        "schema_version": "prime-rl/latent-h-iter-phase1-threshold-builder-contract/v1",
        "status": "threshold_builder_contract_preregistered", "mechanism": MECHANISM,
        "builder_identity": {"payload": f"{MECHANISM}:threshold-builder", "payload_sha256": "084d0518c95b123348383c8c513778a9ff667ff3d28dcd6e366685d8ef2fcfa1"},
        "input_contract": {"requires_t0_go": True, "source": "POST_cal train-only aggregates", "variables": ["AR", "min_depth_AR", "A_RESET", "N_RESET", "NR"]},
        "formula_primitives": {"floor_n": "floor(n*x)/n", "floor_1e6": "floor(1e6*x)/1e6"},
        "validation_threshold_formulas": {"rec_accuracy_min": "min(36/48,max(24/48,floor_48(AR-4/32)))", "per_depth_min": "min(18/24,max(10/24,floor_24(min_depth_AR-1/8)))", "rec_reset_accuracy_min": "min(8/48,max(2/48,floor_48(.5*(AR-A_RESET))))", "reset_rec_nll_min": "min(.15,max(.02,floor_1e6(.5*(N_RESET-NR))))", "rec_fixed_accuracy_min": "2/48", "fixed_rec_nll_min": ".01"},
        "heldout_threshold_binding": "same frozen thresholds; heldout only after validation passes and a separate final contract",
        "go_only_materialization": True,
        "no_exposure_boundary": {"validation_or_heldout_paths_opened_before_freeze": False, "validation_or_heldout_model_calls": 0, "candidate_selection_uses": 0},
        "contract_sha256": "",
    }, "contract_sha256")


def build_assets(bank: dict[str, Any]) -> dict[str, dict[str, Any]]:
    partition = build_partition(bank); cap0 = build_cap0(partition)
    return {
        "phase0-evidence-binding.json": build_phase0_binding(),
        "train-partition.json": partition,
        "cap0-probe-selection.json": cap0,
        "training-schedule.json": build_schedule(partition),
        "candidate-module-contract.json": build_candidate_contract(),
        "capture-contract.json": build_capture_contract(cap0),
        "metric-gate-contract.json": build_metric_contract(),
        "threshold-builder-contract.json": build_threshold_contract(),
    }


def validate_assets(assets: dict[str, dict[str, Any]], bank: dict[str, Any]) -> None:
    if list(assets) != ASSET_NAMES or assets != build_assets(bank):
        raise MF0ContractError("MF0 canonical assets differ")
    partition = assets["train-partition.json"]
    fit_ids = [row["row_id"] for row in partition["fit_rows"]]
    cal_ids = [row["row_id"] for row in partition["calibration_rows"]]
    if len(fit_ids) != 64 or len(set(fit_ids)) != 64 or len(cal_ids) != 32 or len(set(cal_ids)) != 32:
        raise MF0ContractError("MF0 partition count differs")
    if set(fit_ids) & set(cal_ids) or set(fit_ids) | set(cal_ids) != {row["row_id"] for row in bank["rows"]}:
        raise MF0ContractError("MF0 partition disjointness differs")
    for depth in range(1, 5):
        for action in range(4):
            if sum(row["depth"] == depth and row["action_index"] == action for row in partition["fit_rows"]) != 4:
                raise MF0ContractError("MF0 fit depth/action balance differs")
            if sum(row["depth"] == depth and row["action_index"] == action for row in partition["calibration_rows"]) != 2:
                raise MF0ContractError("MF0 calibration depth/action balance differs")
    schedule = assets["training-schedule.json"]
    if len(schedule["batches"]["train"]) != 320 or [row["update_index"] for row in schedule["batches"]["train"]] != list(range(1, 321)):
        raise MF0ContractError("MF0 update schedule differs")
    operations = [*schedule["batches"]["preconnect"], *schedule["batches"]["precal"], *schedule["batches"]["train"], *schedule["batches"]["postcal"], *schedule["batches"]["postfit"]]
    if [row["operation_index"] for row in operations] != list(range(385)):
        raise MF0ContractError("MF0 sidecar operation index differs")
    if sum(len(row["row_ids"]) for row in schedule["batches"]["train"]) != 5120:
        raise MF0ContractError("MF0 fit presentation count differs")
    if sum(len(row["row_ids"]) for row in schedule["batches"]["precal"]) != 160:
        raise MF0ContractError("MF0 PRE-cal presentation count differs")
    if sum(len(row["row_ids"]) for row in schedule["batches"]["postcal"]) != 160:
        raise MF0ContractError("MF0 POST-cal presentation count differs")
    if sum(len(row["row_ids"]) for row in schedule["batches"]["postfit"]) != 320:
        raise MF0ContractError("MF0 POST-fit presentation count differs")


def finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def run_candidate_synthetic(torch: Any) -> dict[str, Any]:
    nn = torch.nn

    class Candidate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.codec_ln = nn.LayerNorm(2048, eps=1e-5, elementwise_affine=True)
            self.codec_projection = nn.Linear(2048, 128, bias=True)
            self.self_norm = nn.LayerNorm(128, eps=1e-5, elementwise_affine=True)
            self.message_norm = nn.LayerNorm(128, eps=1e-5, elementwise_affine=True)
            self.cell_in = nn.Linear(256, 256, bias=True)
            self.cell_out = nn.Linear(256, 128, bias=True)
            self.post_norm = nn.LayerNorm(128, eps=1e-5, elementwise_affine=True)
            self.readout = nn.Linear(128, 4, bias=True)

        def cell(self, state: Any, message: Any) -> Any:
            joined = torch.cat((self.self_norm(state), self.message_norm(message)), dim=-1)
            delta = self.cell_out(torch.nn.functional.gelu(self.cell_in(joined), approximate="none"))
            return self.post_norm(state + delta)

        def forward(self, features: Any, successor: Any, arm: str, depth: int) -> Any:
            z = torch.nn.functional.gelu(self.codec_projection(self.codec_ln(features)), approximate="none")
            if arm == "STATIC":
                state = z
            elif arm == "FFN":
                state = self.cell(z, torch.zeros_like(z))
            elif arm == "FIXED_T4":
                state = z
                for _ in range(4):
                    state = self.cell(state, state.index_select(0, successor))
            elif arm == "RESET_K":
                state = z
                for _ in range(depth):
                    state = self.cell(z, z.index_select(0, successor))
            elif arm == "REC_K":
                state = z
                for _ in range(depth):
                    state = self.cell(state, state.index_select(0, successor))
            else:
                raise MF0ContractError("unknown MF0 arm")
            return self.readout(state[0])

    if sha256_bytes(SYNTHETIC_PAYLOAD.encode()) != SYNTHETIC_SHA256:
        raise MF0ContractError("MF0 synthetic payload differs")
    values = []
    for node in range(24):
        for dimension in range(2048):
            digest = hashlib.sha256(f"{SYNTHETIC_PAYLOAD}:node:{node:02d}:dim:{dimension:04d}".encode()).digest()
            unsigned = int.from_bytes(digest[:4], "big")
            values.append(2.0 * (unsigned / 4294967296.0) - 1.0)
    features = torch.tensor(values, dtype=torch.float32, device="cpu").reshape(24, 2048)
    successor = torch.tensor([*range(1, 24), 0], dtype=torch.int64, device="cpu")
    torch.manual_seed(INIT_SEED)
    first = Candidate().to(dtype=torch.float32, device="cpu")
    initial = {name: tensor.detach().clone() for name, tensor in first.state_dict().items()}
    candidates = {ARMS[0]: first}
    for arm in ARMS[1:]:
        candidate = Candidate().to(dtype=torch.float32, device="cpu")
        candidate.load_state_dict(initial, strict=True)
        candidates[arm] = candidate
    names = list(initial)
    if any(list(candidate.state_dict()) != names for candidate in candidates.values()):
        raise MF0ContractError("MF0 candidate parameter names differ")
    def tensor_hash(tensor: Any) -> str:
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return sha256_bytes(raw)

    def tree_hash(state: dict[str, Any]) -> str:
        rows = [
            {
                "name": name,
                "dtype": str(state[name].dtype),
                "shape": list(state[name].shape),
                "sha256": tensor_hash(state[name]),
            }
            for name in sorted(state)
        ]
        return sha256_bytes(canonical_json(rows))

    initial_tree_sha256 = tree_hash(initial)
    parameter_count = sum(parameter.numel() for parameter in first.parameters())
    if any(tree_hash(dict(candidate.state_dict())) != initial_tree_sha256 for candidate in candidates.values()):
        raise MF0ContractError("MF0 candidate initial bytes differ")
    results = []
    for arm in ARMS:
        candidate = candidates[arm]
        clone = features.detach().clone()
        logits = candidate(clone, successor, arm, 4)
        if logits.shape != (4,) or not torch.isfinite(logits).all():
            raise MF0ContractError("MF0 synthetic output differs")
        loss = torch.nn.functional.cross_entropy(logits.reshape(1, 4), torch.tensor([0]))
        loss.backward()
        codec = [parameter.grad for name, parameter in candidate.named_parameters() if name.startswith("codec_")]
        cell = [parameter.grad for name, parameter in candidate.named_parameters() if name.startswith(("self_norm", "message_norm", "cell_", "post_norm"))]
        readout = [parameter.grad for name, parameter in candidate.named_parameters() if name.startswith("readout")]
        def group_ok(group: list[Any]) -> bool:
            return all(gradient is not None and torch.isfinite(gradient).all() for gradient in group) and sum(float(gradient.double().square().sum()) for gradient in group) > 1e-8
        if not group_ok(codec) or not group_ok(readout):
            raise MF0ContractError("MF0 codec/readout connectivity differs")
        if arm == "STATIC":
            if any(gradient is not None for gradient in cell):
                raise MF0ContractError("MF0 STATIC cell received gradients")
            cell_connected = False
        else:
            if not group_ok(cell):
                raise MF0ContractError("MF0 cell connectivity differs")
            cell_connected = True
        unchanged = all(torch.equal(candidate.state_dict()[name], initial[name]) for name in names)
        if not unchanged:
            raise MF0ContractError("MF0 backward changed candidate state")
        results.append({"arm": arm, "output_shape": [4], "codec_gradient_nonzero": True, "readout_gradient_nonzero": True, "cell_gradient_nonzero": cell_connected, "state_unchanged": unchanged, "initial_tree_sha256": initial_tree_sha256})
        candidate.zero_grad(set_to_none=True)
    feature_sha256 = tensor_hash(features)
    if feature_sha256 != SYNTHETIC_FEATURE_SHA256:
        raise MF0ContractError("MF0 synthetic feature bytes differ")
    del candidates, first, initial, features, successor
    return {
        "arms": results,
        "forwards": 5,
        "backwards": 5,
        "optimizer_objects": 0,
        "optimizer_steps": 0,
        "parameter_names": names,
        "parameter_count_per_arm": parameter_count,
        "initial_tree_sha256": initial_tree_sha256,
        "all_initial_trees_equal": True,
        "synthetic_feature_shape": [24, 2048],
        "synthetic_feature_sha256": feature_sha256,
    }
