from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
from collections.abc import Iterable
from typing import Any

MECHANISM = "q35-2b-h-iter-phase0-generator-locality-v1"
RUN_IDENTITY = "h-iter-phase0-generator-locality-run1"
BANK_SCHEMA = "q35-2b-h-iter-bank/v1"
RECEIVER_SCHEMA = "q35-2b-h-iter-receiver-input/v1"
SUPERVISION_SCHEMA = "q35-2b-h-iter-supervision/v1"
PLAN_SCHEMA = "prime-rl/latent-h-iter-phase0-generator-locality-plan/v1"
PROOF_SCHEMA = "prime-rl/latent-h-iter-phase0-generator-locality-proof/v1"
FAILURE_SCHEMA = "prime-rl/latent-h-iter-phase0-generator-locality-failure/v1"
PROBE_SCHEMA = "prime-rl/latent-h-iter-phase0-locality-probes/v1"
OVERLAP_SCHEMA = "prime-rl/latent-h-iter-phase0-overlap-evidence/v1"
OPERATION_SCHEMA = "prime-rl/latent-h-iter-phase0-operation-schedule/v1"
TAMPER_SCHEMA = "prime-rl/latent-h-iter-phase0-tamper-schedule/v1"
ARTIFACT_DIR_REL = "experiments/qwen35-2b-latent-workspace-v1/h-iter-phase0-generator-locality-v1"

NODE_COUNT = 24
FEATURE_DIM = 4
MARKERS = ["mk_3f8a", "mk_6c1d", "mk_9b4e", "mk_c2a7"]
ACTIONS = ["ACT_Z1", "ACT_K4", "ACT_M7", "ACT_Q9"]
SPLITS = {"train": [1, 2, 3, 4], "validation": [5, 6], "heldout": [7, 8]}
BANK_PAYLOADS = {split: f"{MECHANISM}:bank:{split}" for split in SPLITS}
ORDER_PAYLOADS = {split: f"{MECHANISM}:order:{split}" for split in SPLITS}
EXPECTED_PAYLOAD_SHA256 = {
    "train": "0ab70d15a87d316292c13457d5d0851b1cc73069df3d445fe3342f0a47ea7950",
    "validation": "1d34b0b7581f41828886f2c50e00c72a5d85fb1dac01b94c1b1cdca271bf5321",
    "heldout": "dbeb56d6520f5359f0f10d3b32c4c238288f3db4731db2a52ed3615ad21fcf8a",
}
EXPECTED_PAYLOAD_SEEDS = {
    "train": 772100247789580642,
    "validation": 2104501227392811394,
    "heldout": 15846855192332948313,
}
EXPECTED_ORDER_SHA256 = {
    "train": "aa02e1f474c5921a1975b59163ede3a9958787bfedae0f473531460196e8ac2a",
    "validation": "18bf4beb762a3decd5f7ba946b01a1e48c8e23a54bbfa9af00e548a0bff4d4d7",
    "heldout": "1e32f5dc00263950a1cfb23ac60a5c7fc3f3839a2b1ea6a3d7b32c9b9d84f530",
}
EXPECTED_ORDER_SEEDS = {
    "train": 12250602376448545306,
    "validation": 1783227452133883372,
    "heldout": 2176071895217486160,
}
PROBE_PAYLOAD = f"{MECHANISM}:locality-probes"
PROBE_PAYLOAD_SHA256 = "dc5fb76dc85a9cf48230d9038e6c6c8990572d242bc6d5c0a9eb11356a6a338a"
PROBE_SEED = 15879612493272358132
PERTURB_VECTOR = [2**-20, 2**-21, 2**-22, 2**-23]

ARM_NAMES = ["STATIC", "FFN", "FIXED_T4", "RESET_K", "REC_K"]
MECHANISM_TAMPERS = [
    "train_row_seed_changed",
    "row_order_changed",
    "row_deleted",
    "row_duplicated",
    "cross_split_row_id_reused",
    "node_id_reused",
    "node_id_in_local_text",
    "answer_action_in_receiver_input",
    "node_count_changed",
    "edge_count_changed",
    "ring_broken",
    "target_distance_changed",
    "marker_balance_changed",
    "action_balance_changed",
    "distractor_count_changed",
    "endpoint_swap_donor_inside_radius",
    "encoder_given_graph_context",
    "two_hop_message_access",
    "dense_global_aggregate_access",
    "readout_given_all_nodes",
    "reset_arm_persists_state",
    "recurrent_arm_resets_state",
    "endpoint_swap_changes_k_minus_1",
    "endpoint_swap_unchanged_at_k",
    "autograd_outside_radius_nonzero",
    "autograd_inside_radius_zero",
]
RECEIPT_TAMPERS = [
    "missing_top_key",
    "extra_top_key",
    "status_changed",
    "bank_hash_changed",
    "full_freeze_truncated",
    "model_call_nonzero",
    "cuda_initialized_true",
    "optimizer_step_nonzero",
    "validation_phase1_learning_exposure_nonzero",
    "heldout_phase1_learning_exposure_nonzero",
    "thresholds_present",
    "receipt_sha_stale",
]

_ROW_KEYS = {
    "ordinal",
    "row_id",
    "split",
    "depth",
    "action_index",
    "replicate",
    "generation_payload",
    "generation_sha256",
    "generation_seed_u64_be",
    "order_key_sha256",
    "receiver_input",
    "supervision",
    "receiver_input_sha256",
    "row_sha256",
}
_BANK_KEYS = {
    "schema_version",
    "split",
    "generator_namespace",
    "generator_payload",
    "generator_payload_sha256",
    "generator_seed_u64_be",
    "depths",
    "actions",
    "markers",
    "rows",
    "bank_sha256",
}
_TOKEN = re.compile(r"^(?:hi_|n_|z_)[0-9a-f]{16}$")
_IDENTITY_IN_TEXT = re.compile(r"(?<![A-Za-z0-9_])((?:hi|n|z)_[0-9a-f]{16})(?![A-Za-z0-9_])")


class ContractError(ValueError):
    pass


def strict_json_loads(data: str | bytes) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ContractError(f"nonfinite JSON constant: {value}")

    return json.loads(data, object_pairs_hook=pairs_hook, parse_constant=reject_constant)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any, *, omit: str | None = None) -> str:
    if omit is not None:
        if not isinstance(value, dict):
            raise ContractError("self hash applies only to a mapping")
        value = {key: item for key, item in value.items() if key != omit}
    return sha256_bytes(canonical_json(value))


def seed_u64(payload: str) -> int:
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def _digest(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _shuffle(values: Iterable[int] | Iterable[str], seed: int) -> list[Any]:
    result = list(values)
    random.Random(seed).shuffle(result)
    return result


def _node_id(payload: str, logical_index: int) -> str:
    return "n_" + _digest(f"{payload}:node:{logical_index}")[:16]


def _nonce(payload: str, logical_index: int) -> str:
    return "z_" + _digest(f"{payload}:nonce:{logical_index}")[:16]


def _local_text(marker: str, nonce: str) -> str:
    return f"<hiter_local>\nmarker={marker}\nnonce={nonce}\n</hiter_local>"


def generate_row(split: str, depth: int, action_index: int, replicate: int) -> dict[str, Any]:
    if split not in SPLITS or depth not in SPLITS[split]:
        raise ContractError("row split/depth differs from contract")
    if action_index not in range(4) or replicate not in range(6):
        raise ContractError("row action/replicate differs from contract")
    action = ACTIONS[action_index]
    payload = f"{BANK_PAYLOADS[split]}:K={depth}:action={action}:rep={replicate}"
    digest = _digest(payload)
    component_seed = lambda component: seed_u64(f"{payload}:{component}")

    ring = _shuffle(range(NODE_COUNT), component_seed("ring"))
    start = ring[0]
    target = ring[depth]
    successor = {ring[index]: ring[(index + 1) % NODE_COUNT] for index in range(NODE_COUNT)}

    residual = [MARKERS[action_index]] * 5
    for index, marker in enumerate(MARKERS):
        if index != action_index:
            residual.extend([marker] * 6)
    residual = _shuffle(residual, component_seed("markers"))
    marker_by_node = {target: MARKERS[action_index]}
    for logical_index, marker in zip(
        (index for index in range(NODE_COUNT) if index != target), residual, strict=True
    ):
        marker_by_node[logical_index] = marker

    node_order = _shuffle(range(NODE_COUNT), component_seed("node-order"))
    edge_order = _shuffle(range(NODE_COUNT), component_seed("edge-order"))
    node_ids = {index: _node_id(payload, index) for index in range(NODE_COUNT)}
    nonces = {index: _nonce(payload, index) for index in range(NODE_COUNT)}
    nodes = [
        {
            "node_id": node_ids[index],
            "local_text": _local_text(marker_by_node[index], nonces[index]),
        }
        for index in node_order
    ]
    edges = [
        {
            "source_node_id": node_ids[index],
            "successor_node_id": node_ids[successor[index]],
        }
        for index in edge_order
    ]
    receiver = {
        "schema_version": RECEIVER_SCHEMA,
        "start_node_id": node_ids[start],
        "depth": depth,
        "nodes": nodes,
        "edges": edges,
    }
    supervision = {
        "schema_version": SUPERVISION_SCHEMA,
        "target_node_id": node_ids[target],
        "target_marker": MARKERS[action_index],
        "answer_action": action,
    }
    order_key = _digest(f"{ORDER_PAYLOADS[split]}:row_id=hi_{digest[:16]}")
    row = {
        "ordinal": -1,
        "row_id": "hi_" + digest[:16],
        "split": split,
        "depth": depth,
        "action_index": action_index,
        "replicate": replicate,
        "generation_payload": payload,
        "generation_sha256": digest,
        "generation_seed_u64_be": int(digest[:16], 16),
        "order_key_sha256": order_key,
        "receiver_input": receiver,
        "supervision": supervision,
        "receiver_input_sha256": canonical_sha256(receiver),
        "row_sha256": "",
    }
    return row


def _finish_row(row: dict[str, Any], ordinal: int) -> dict[str, Any]:
    row = copy.deepcopy(row)
    row["ordinal"] = ordinal
    row["row_sha256"] = canonical_sha256(row, omit="row_sha256")
    return row


def generate_bank(split: str) -> dict[str, Any]:
    rows = [
        generate_row(split, depth, action_index, replicate)
        for depth in SPLITS[split]
        for action_index in range(4)
        for replicate in range(6)
    ]
    rows.sort(key=lambda row: (row["order_key_sha256"], row["row_id"]))
    rows = [_finish_row(row, ordinal) for ordinal, row in enumerate(rows)]
    bank = {
        "schema_version": BANK_SCHEMA,
        "split": split,
        "generator_namespace": MECHANISM,
        "generator_payload": BANK_PAYLOADS[split],
        "generator_payload_sha256": _digest(BANK_PAYLOADS[split]),
        "generator_seed_u64_be": seed_u64(BANK_PAYLOADS[split]),
        "depths": SPLITS[split],
        "actions": ACTIONS,
        "markers": MARKERS,
        "rows": rows,
        "bank_sha256": "",
    }
    bank["bank_sha256"] = canonical_sha256(bank, omit="bank_sha256")
    return bank


def row_ring(row: dict[str, Any]) -> tuple[list[str], dict[str, str], dict[str, str]]:
    receiver = row["receiver_input"]
    successor = {edge["source_node_id"]: edge["successor_node_id"] for edge in receiver["edges"]}
    local = {node["node_id"]: node["local_text"] for node in receiver["nodes"]}
    ring = [receiver["start_node_id"]]
    for _ in range(1, NODE_COUNT):
        ring.append(successor[ring[-1]])
    return ring, successor, local


def marker_from_local_text(text: str) -> str:
    match = re.fullmatch(r"<hiter_local>\nmarker=([^\n]+)\nnonce=(z_[0-9a-f]{16})\n</hiter_local>", text)
    if match is None or match.group(1) not in MARKERS:
        raise ContractError("local_text differs from exact contract")
    return match.group(1)


def nonce_from_local_text(text: str) -> str:
    match = re.fullmatch(r"<hiter_local>\nmarker=[^\n]+\nnonce=(z_[0-9a-f]{16})\n</hiter_local>", text)
    if match is None:
        raise ContractError("local_text nonce differs from exact contract")
    return match.group(1)


def donor_for_row(row: dict[str, Any]) -> tuple[str, int, str]:
    ring, _, local = row_ring(row)
    target_marker = row["supervision"]["target_marker"]
    for distance in range(row["depth"] + 1, NODE_COUNT):
        node_id = ring[distance]
        marker = marker_from_local_text(local[node_id])
        if marker != target_marker:
            return node_id, distance, marker
    raise ContractError("no eligible donor exists")


def _assert_answer_free(receiver: dict[str, Any]) -> None:
    encoded = canonical_json(receiver).decode("utf-8")
    forbidden = [*ACTIONS, "answer_action", "target_node_id", "target_marker", "supervision", "donor", "ring_position"]
    if any(item in encoded for item in forbidden):
        raise ContractError("receiver input contains a forbidden answer/audit field")


def validate_bank(bank: dict[str, Any], split: str) -> dict[str, Any]:
    if set(bank) != _BANK_KEYS or bank.get("schema_version") != BANK_SCHEMA or bank.get("split") != split:
        raise ContractError("bank schema/split differs")
    expected = generate_bank(split)
    if canonical_json(bank) != canonical_json(expected):
        raise ContractError("bank does not regenerate byte-identically")
    if bank["generator_payload_sha256"] != EXPECTED_PAYLOAD_SHA256[split]:
        raise ContractError("bank payload digest differs")
    if bank["generator_seed_u64_be"] != EXPECTED_PAYLOAD_SEEDS[split]:
        raise ContractError("bank payload seed differs")
    if _digest(ORDER_PAYLOADS[split]) != EXPECTED_ORDER_SHA256[split]:
        raise ContractError("order payload digest differs")
    if seed_u64(ORDER_PAYLOADS[split]) != EXPECTED_ORDER_SEEDS[split]:
        raise ContractError("order payload seed differs")
    expected_count = len(SPLITS[split]) * 24
    if len(bank["rows"]) != expected_count:
        raise ContractError("bank row count differs")
    depth_action_counts: dict[tuple[int, int], int] = {}
    for ordinal, row in enumerate(bank["rows"]):
        if set(row) != _ROW_KEYS or row["ordinal"] != ordinal:
            raise ContractError("row schema/ordinal differs")
        if row["row_sha256"] != canonical_sha256(row, omit="row_sha256"):
            raise ContractError("row self hash differs")
        if row["receiver_input_sha256"] != canonical_sha256(row["receiver_input"]):
            raise ContractError("receiver hash differs")
        receiver = row["receiver_input"]
        if set(receiver) != {"schema_version", "start_node_id", "depth", "nodes", "edges"}:
            raise ContractError("receiver schema fields differ")
        if receiver["schema_version"] != RECEIVER_SCHEMA or receiver["depth"] != row["depth"]:
            raise ContractError("receiver schema/depth differs")
        if len(receiver["nodes"]) != NODE_COUNT or len(receiver["edges"]) != NODE_COUNT:
            raise ContractError("topology size differs")
        if any(set(node) != {"node_id", "local_text"} for node in receiver["nodes"]):
            raise ContractError("node fields differ")
        if any(set(edge) != {"source_node_id", "successor_node_id"} for edge in receiver["edges"]):
            raise ContractError("edge fields differ")
        ids = [node["node_id"] for node in receiver["nodes"]]
        if len(set(ids)) != NODE_COUNT:
            raise ContractError("row node identifiers are not unique")
        ring, successor, local = row_ring(row)
        if len(set(ring)) != NODE_COUNT or successor[ring[-1]] != ring[0]:
            raise ContractError("topology is not one Hamiltonian cycle")
        if set(successor) != set(ids) or set(successor.values()) != set(ids):
            raise ContractError("topology indegree/outdegree differs")
        supervision = row["supervision"]
        if set(supervision) != {"schema_version", "target_node_id", "target_marker", "answer_action"}:
            raise ContractError("supervision fields differ")
        if supervision["schema_version"] != SUPERVISION_SCHEMA:
            raise ContractError("supervision schema differs")
        if supervision["target_node_id"] != ring[row["depth"]]:
            raise ContractError("target distance differs")
        if len(set(ring[:9])) != 9:
            raise ContractError("p0..p8 are not distinct")
        markers = [marker_from_local_text(local[node_id]) for node_id in ring]
        if any(markers.count(marker) != 6 for marker in MARKERS):
            raise ContractError("marker balance differs")
        if supervision["target_marker"] != markers[row["depth"]]:
            raise ContractError("target marker differs")
        if supervision["answer_action"] != ACTIONS[row["action_index"]]:
            raise ContractError("answer/action index differs")
        donor_id, donor_distance, donor_marker = donor_for_row(row)
        if donor_distance <= row["depth"] or donor_marker == supervision["target_marker"]:
            raise ContractError("donor locality differs")
        if donor_id not in ids or NODE_COUNT - row["depth"] - 1 != len(ring[row["depth"] + 1 :]):
            raise ContractError("distractor count differs")
        _assert_answer_free(receiver)
        depth_action_counts[(row["depth"], row["action_index"])] = (
            depth_action_counts.get((row["depth"], row["action_index"]), 0) + 1
        )
    if set(depth_action_counts.values()) != {6} or len(depth_action_counts) != len(SPLITS[split]) * 4:
        raise ContractError("depth/action balance differs")
    return {
        "split": split,
        "row_count": len(bank["rows"]),
        "structural_rows_valid": len(bank["rows"]),
        "row_order_sha256": canonical_sha256([row["row_id"] for row in bank["rows"]]),
        "node_serialization_order_sha256": canonical_sha256(
            [
                {"row_id": row["row_id"], "node_ids": [node["node_id"] for node in row["receiver_input"]["nodes"]]}
                for row in bank["rows"]
            ]
        ),
        "edge_serialization_order_sha256": canonical_sha256(
            [
                {
                    "row_id": row["row_id"],
                    "edges": [
                        [edge["source_node_id"], edge["successor_node_id"]]
                        for edge in row["receiver_input"]["edges"]
                    ],
                }
                for row in bank["rows"]
            ]
        ),
    }


def validate_banks(banks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(banks) != set(SPLITS):
        raise ContractError("bank split set differs")
    summaries = [validate_bank(banks[split], split) for split in SPLITS]
    row_ids: set[str] = set()
    node_ids: set[str] = set()
    nonces: set[str] = set()
    receiver_hashes: set[str] = set()
    for split in SPLITS:
        for row in banks[split]["rows"]:
            if row["row_id"] in row_ids or row["receiver_input_sha256"] in receiver_hashes:
                raise ContractError("cross-split row/receiver identity reused")
            row_ids.add(row["row_id"])
            receiver_hashes.add(row["receiver_input_sha256"])
            for node in row["receiver_input"]["nodes"]:
                nonce = nonce_from_local_text(node["local_text"])
                if node["node_id"] in node_ids or nonce in nonces:
                    raise ContractError("global node/nonce identity reused")
                node_ids.add(node["node_id"])
                nonces.add(nonce)
    return {
        "split_summaries": summaries,
        "total_rows": len(row_ids),
        "unique_row_ids": len(row_ids),
        "unique_node_ids": len(node_ids),
        "unique_nonces": len(nonces),
        "unique_receiver_input_sha256": len(receiver_hashes),
    }


def build_probe_selection(banks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    probes = []
    for depth in range(1, 9):
        split = "train" if depth <= 4 else "validation" if depth <= 6 else "heldout"
        matches = [
            row
            for row in banks[split]["rows"]
            if row["depth"] == depth and row["action_index"] == 0 and row["replicate"] == 0
        ]
        if len(matches) != 1:
            raise ContractError("probe selector is not singular")
        row = matches[0]
        donor_id, donor_distance, donor_marker = donor_for_row(row)
        probes.append(
            {
                "probe_index": depth,
                "split": split,
                "depth": depth,
                "action_index": 0,
                "answer_action": "ACT_Z1",
                "replicate": 0,
                "row_id": row["row_id"],
                "receiver_input_sha256": row["receiver_input_sha256"],
                "target_node_id": row["supervision"]["target_node_id"],
                "target_marker": row["supervision"]["target_marker"],
                "donor_node_id": donor_id,
                "donor_distance": donor_distance,
                "donor_marker": donor_marker,
            }
        )
    result = {
        "schema_version": PROBE_SCHEMA,
        "generator_namespace": MECHANISM,
        "probe_payload": PROBE_PAYLOAD,
        "probe_payload_sha256": _digest(PROBE_PAYLOAD),
        "probe_seed_u64_be": seed_u64(PROBE_PAYLOAD),
        "probes": probes,
        "selection_sha256": "",
    }
    result["selection_sha256"] = canonical_sha256(result, omit="selection_sha256")
    return result


def build_operation_schedule(selection: dict[str, Any]) -> dict[str, Any]:
    rows = []
    index = 1
    for probe in selection["probes"]:
        depth = probe["depth"]
        specs = []
        for arm in ARM_NAMES:
            calls = 0 if arm in {"STATIC", "FFN"} else 4 if arm == "FIXED_T4" else depth
            readouts = 2 if arm == "REC_K" else 1
            specs.append(("BASELINE", arm, calls, readouts, True))
        specs.append(("SWAP_REC_K", "REC_K", depth, 2, False))
        for arm in ARM_NAMES:
            calls = 0 if arm in {"STATIC", "FFN"} else 4 if arm == "FIXED_T4" else depth
            specs.extend(
                [
                    ("PERTURB_INSIDE", arm, calls, 1, False),
                    ("PERTURB_OUTSIDE", arm, calls, 1, False),
                ]
            )
        for kind, arm, calls, readouts, backward in specs:
            encode_passes = 1 if (kind == "BASELINE" and arm == "STATIC") or kind == "SWAP_REC_K" else 0
            rows.append(
                {
                    "operation_index": index,
                    "probe_index": probe["probe_index"],
                    "row_id": probe["row_id"],
                    "depth": depth,
                    "kind": kind,
                    "arm": arm,
                    "transition_calls": calls,
                    "readout_calls": readouts,
                    "graph_encode_passes": encode_passes,
                    "local_node_encode_calls": NODE_COUNT * encode_passes,
                    "synthetic_backward_calls": 1 if backward else 0,
                }
            )
            index += 1
    result = {
        "schema_version": OPERATION_SCHEMA,
        "operations": rows,
        "expected_counts": {
            "arm_executions": 128,
            "transition_calls": 348,
            "readout_calls": 144,
            "graph_encode_passes": 16,
            "local_node_encode_calls": 384,
            "synthetic_backward_calls": 40,
            "perturb_runs": 80,
            "endpoint_swap_runs": 8,
            "model_or_transformer_forwards": 0,
            "optimizer_steps": 0,
        },
        "schedule_sha256": "",
    }
    result["schedule_sha256"] = canonical_sha256(result, omit="schedule_sha256")
    return result


def build_tamper_schedule() -> dict[str, Any]:
    result = {
        "schema_version": TAMPER_SCHEMA,
        "mechanism_tampers": [
            {"tamper_index": index, "name": name} for index, name in enumerate(MECHANISM_TAMPERS, 1)
        ],
        "receipt_tampers": [
            {"tamper_index": index, "name": name} for index, name in enumerate(RECEIPT_TAMPERS, 1)
        ],
        "tamper_schedule_sha256": "",
    }
    result["tamper_schedule_sha256"] = canonical_sha256(result, omit="tamper_schedule_sha256")
    return result


def iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_strings(item)


def _receiver_candidates(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if {"start_node_id", "depth", "nodes", "edges"} <= set(value):
            yield value
        for item in value.values():
            yield from _receiver_candidates(item)
    elif isinstance(value, list):
        for item in value:
            yield from _receiver_candidates(item)


def extract_prior_source(path: str, data: bytes, identities: dict[str, set[str]]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    if path.endswith(".json"):
        value = strict_json_loads(data)
        container = {"container": "strict_json", "container_schema_sha256": canonical_sha256(value)}
    elif path.endswith(".sha256"):
        text = data.decode("utf-8")
        lines = text.splitlines()
        if not lines or any(not re.fullmatch(r"[0-9a-f]{64}(?:  .+)?", line) for line in lines):
            raise ContractError(f"invalid prior SHA manifest: {path}")
        value = lines
        container = {
            "container": "sha256_manifest",
            "container_schema_sha256": sha256_bytes(b"sha256[ path]"),
        }
    elif path.endswith(".parquet"):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pq.read_table(pa.BufferReader(data))
        value = table.to_pylist()
        container = {
            "container": "parquet_pyarrow_24",
            "container_schema_sha256": sha256_bytes(str(table.schema).encode("utf-8")),
            "parquet_row_count": table.num_rows,
            "parquet_column_count": table.num_columns,
        }
    else:
        raise ContractError(f"unsupported prior source: {path}")
    strings = list(iter_strings(value))
    tokens = {match.group(1) for string in strings for match in _IDENTITY_IN_TEXT.finditer(string)}
    source_sets = {
        "row_ids": {token for token in tokens if token.startswith("hi_")},
        "node_ids": {token for token in tokens if token.startswith("n_")},
        "nonces": {token for token in tokens if token.startswith("z_")},
        "receiver_hashes": {canonical_sha256(candidate) for candidate in _receiver_candidates(value)},
        "local_texts": set(strings),
    }
    extraction_payload = {
        "row_ids": sorted(source_sets["row_ids"]),
        "node_ids": sorted(source_sets["node_ids"]),
        "nonces": sorted(source_sets["nonces"]),
        "receiver_input_sha256": sorted(source_sets["receiver_hashes"]),
        "complete_local_text_sha256": sorted(
            sha256_bytes(item.encode("utf-8")) for item in source_sets["local_texts"]
        ),
    }
    observed = {
        **container,
        "string_scalar_count": len(strings),
        "distinct_string_scalar_count": len(set(strings)),
        "row_id_token_count": len(source_sets["row_ids"]),
        "node_id_token_count": len(source_sets["node_ids"]),
        "nonce_token_count": len(source_sets["nonces"]),
        "receiver_input_candidate_count": len(source_sets["receiver_hashes"]),
        "extraction_sha256": canonical_sha256(extraction_payload),
    }
    intersection = {
        "row_ids": sorted(identities["row_ids"] & source_sets["row_ids"]),
        "node_ids": sorted(identities["node_ids"] & source_sets["node_ids"]),
        "nonces": sorted(identities["nonces"] & source_sets["nonces"]),
        "receiver_input_sha256": sorted(identities["receiver_hashes"] & source_sets["receiver_hashes"]),
        "complete_local_text": sorted(identities["local_texts"] & source_sets["local_texts"]),
    }
    return observed, intersection


def new_identity_sets(banks: dict[str, dict[str, Any]]) -> dict[str, set[str]]:
    result = {"row_ids": set(), "node_ids": set(), "nonces": set(), "receiver_hashes": set(), "local_texts": set()}
    for bank in banks.values():
        for row in bank["rows"]:
            result["row_ids"].add(row["row_id"])
            result["receiver_hashes"].add(row["receiver_input_sha256"])
            for node in row["receiver_input"]["nodes"]:
                result["node_ids"].add(node["node_id"])
                result["nonces"].add(nonce_from_local_text(node["local_text"]))
                result["local_texts"].add(node["local_text"])
    return result


def validate_no_threshold_fields(value: Any) -> None:
    allowed_false_sentinels = {
        "validation_learning_threshold_use",
        "heldout_learning_threshold_use",
        "phase1_thresholds_present",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            if "threshold" in key.lower():
                if key not in allowed_false_sentinels or item is not False:
                    raise ContractError("Phase-0 artifact contains a learning-threshold field")
                continue
            validate_no_threshold_fields(item)
    elif isinstance(value, list):
        for item in value:
            validate_no_threshold_fields(item)


def validate_schedule(schedule: dict[str, Any], selection: dict[str, Any]) -> None:
    if schedule != build_operation_schedule(selection):
        raise ContractError("operation schedule differs from frozen builder")
    totals = {
        key: sum(row[key] for row in schedule["operations"])
        for key in (
            "transition_calls",
            "readout_calls",
            "graph_encode_passes",
            "local_node_encode_calls",
            "synthetic_backward_calls",
        )
    }
    for key, value in totals.items():
        if value != schedule["expected_counts"][key]:
            raise ContractError(f"operation total differs: {key}")
    if sum(row["kind"].startswith("PERTURB_") for row in schedule["operations"]) != 80:
        raise ContractError("perturb operation count differs")
    if sum(row["kind"] == "SWAP_REC_K" for row in schedule["operations"]) != 8:
        raise ContractError("endpoint-swap operation count differs")


def expected_symbolic_counts() -> dict[str, int]:
    return {"final_arm_checks": 960, "recurrent_timestep_checks": 1056}


def finite_float(value: Any) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def validate_probe_selection(selection: dict[str, Any], banks: dict[str, dict[str, Any]]) -> None:
    if selection != build_probe_selection(banks):
        raise ContractError("locality probe selection differs from the frozen builder")
    if selection["probe_payload_sha256"] != PROBE_PAYLOAD_SHA256 or selection["probe_seed_u64_be"] != PROBE_SEED:
        raise ContractError("locality probe payload identity differs")


def memory_labels() -> list[str]:
    labels = [
        "runtime_verified",
        "full_freeze_preflight_verified",
        "banks_structurally_validated",
        "overlap_closure_validated",
        "operation_schedule_validated",
    ]
    for probe_index in range(1, 9):
        labels.extend([f"pre_locality_probe_{probe_index:02d}", f"post_locality_probe_{probe_index:02d}"])
    labels.extend(
        [
            "symbolic_dependency_validated",
            "mechanism_tampers_validated",
            "safety_postflight_validated",
            "full_freeze_postflight_validated",
            "proof_prewrite_ready",
        ]
    )
    return labels


def _tensor_sha256(tensor: Any) -> str:
    import torch

    value = tensor.detach().cpu().contiguous()
    if value.dtype != torch.float64:
        raise ContractError("locality tensor is not CPU float64")
    return sha256_bytes(bytes(value.reshape(-1).view(torch.uint8).tolist()))


def _encode_graph(row: dict[str, Any]):
    import torch

    nodes = row["receiver_input"]["nodes"]
    node_index = {node["node_id"]: index for index, node in enumerate(nodes)}
    features = []
    for node in nodes:
        marker = marker_from_local_text(node["local_text"])
        nonce_from_local_text(node["local_text"])
        feature = [0.0] * FEATURE_DIM
        feature[MARKERS.index(marker)] = 1.0
        features.append(feature)
    hidden = torch.tensor(features, dtype=torch.float64, device="cpu")
    successor = {edge["source_node_id"]: edge["successor_node_id"] for edge in row["receiver_input"]["edges"]}
    successor_index = torch.tensor(
        [node_index[successor[node["node_id"]]] for node in nodes], dtype=torch.int64, device="cpu"
    )
    start_index = node_index[row["receiver_input"]["start_node_id"]]
    return hidden, successor_index, start_index, node_index


def _transition(hidden: Any, successor_index: Any):
    return 0.75 * hidden + 0.25 * hidden.index_select(0, successor_index)


def _arm_state(arm: str, hidden: Any, successor_index: Any, depth: int, *, steps: int | None = None):
    if arm == "STATIC":
        return hidden
    if arm == "FFN":
        return 1.125 * hidden
    if arm == "FIXED_T4":
        state = hidden
        for _ in range(4):
            state = _transition(state, successor_index)
        return state
    if arm == "RESET_K":
        state = hidden
        for _ in range(depth):
            state = _transition(hidden, successor_index)
        return state
    if arm == "REC_K":
        state = hidden
        for _ in range(depth if steps is None else steps):
            state = _transition(state, successor_index)
        return state
    raise ContractError(f"unknown locality arm: {arm}")


def _readout(vector: Any):
    if list(vector.shape) != [FEATURE_DIM]:
        raise ContractError("readout received more than the indexed start vector")
    return vector, vector.sum()


def _radius(arm: str, depth: int) -> int:
    return {"STATIC": 0, "FFN": 0, "FIXED_T4": 4, "RESET_K": 1, "REC_K": depth}[arm]


def _distance_indices(row: dict[str, Any], node_index: dict[str, int]) -> list[int]:
    ring, _, _ = row_ring(row)
    return [node_index[node_id] for node_id in ring]


def run_locality_probe(row: dict[str, Any], probe: dict[str, Any]) -> dict[str, Any]:
    import torch

    depth = row["depth"]
    if depth != probe["depth"]:
        raise ContractError("probe depth differs from selected row")
    baseline = []
    operation_counts = {
        "arm_executions": 0,
        "transition_calls": 0,
        "readout_calls": 0,
        "graph_encode_passes": 0,
        "local_node_encode_calls": 0,
        "synthetic_backward_calls": 0,
        "perturb_runs": 0,
        "endpoint_swap_runs": 0,
    }

    def encoded(source_row: dict[str, Any] = row):
        operation_counts["graph_encode_passes"] += 1
        operation_counts["local_node_encode_calls"] += NODE_COUNT
        return _encode_graph(source_row)

    def transitions_for(arm: str) -> int:
        return 0 if arm in {"STATIC", "FFN"} else 4 if arm == "FIXED_T4" else depth

    original_hidden, original_successors, original_start_index, original_node_index = encoded()
    rec_k_minus_1 = None
    rec_k = None
    baseline_vectors: dict[str, Any] = {}
    for arm in ARM_NAMES:
        hidden = original_hidden.clone().detach().requires_grad_(True)
        successors = original_successors
        start_index = original_start_index
        node_index = original_node_index
        if arm == "REC_K":
            state = hidden
            for timestep in range(depth):
                if timestep == depth - 1:
                    rec_k_minus_1, _ = _readout(state[start_index])
                state = _transition(state, successors)
            vector, scalar = _readout(state[start_index])
            rec_k = vector.detach().clone()
        elif arm == "RESET_K":
            reset_first = None
            state = hidden
            for _ in range(depth):
                state = _transition(hidden, successors)
                if reset_first is None:
                    reset_first = state
            if reset_first is None:
                raise ContractError("reset arm performed no transition")
            vector, scalar = _readout(state[start_index])
        else:
            state = _arm_state(arm, hidden, successors, depth)
            vector, scalar = _readout(state[start_index])
        baseline_vectors[arm] = vector.detach().clone()
        scalar.backward()
        distances = _distance_indices(row, node_index)
        radius = _radius(arm, depth)
        inside = hidden.grad[distances[: radius + 1]]
        outside = hidden.grad[distances[radius + 1 :]]
        if not torch.isfinite(inside).all() or not bool((inside.abs() > 1e-12).all()):
            raise ContractError("autograd inside radius is zero/nonfinite")
        if outside.numel() and not torch.equal(outside, torch.zeros_like(outside)):
            raise ContractError("autograd outside radius is nonzero")
        if arm == "RESET_K":
            if not torch.equal(state, reset_first):
                raise ContractError("reset arm does not equal one transition")
        baseline.append(
            {
                "arm": arm,
                "radius": radius,
                "transition_calls": transitions_for(arm),
                "readout_sha256": _tensor_sha256(vector),
                "inside_gradient_components": (radius + 1) * FEATURE_DIM,
                "outside_gradient_components": (NODE_COUNT - radius - 1) * FEATURE_DIM,
                "inside_gradient_finite_nonzero": True,
                "outside_gradient_bitwise_positive_zero": True,
                "reset_equals_single_transition": True if arm == "RESET_K" else None,
            }
        )
        operation_counts["arm_executions"] += 1
        operation_counts["transition_calls"] += transitions_for(arm)
        operation_counts["readout_calls"] += 1 if arm != "REC_K" else 2
        operation_counts["synthetic_backward_calls"] += 1

    hidden = original_hidden
    successors = original_successors
    start_index = original_start_index
    node_index = original_node_index
    ring, _, local = row_ring(row)
    target_id = row["supervision"]["target_node_id"]
    donor_id = probe["donor_node_id"]
    swapped_row = copy.deepcopy(row)
    swapped_local = {node["node_id"]: node for node in swapped_row["receiver_input"]["nodes"]}
    swapped_local[target_id]["local_text"], swapped_local[donor_id]["local_text"] = (
        swapped_local[donor_id]["local_text"],
        swapped_local[target_id]["local_text"],
    )
    swapped, swapped_successors, swapped_start_index, swapped_node_index = encoded(swapped_row)
    if not torch.equal(successors, swapped_successors) or start_index != swapped_start_index or node_index != swapped_node_index:
        raise ContractError("endpoint swap changed topology/indexing")
    swapped_state = swapped
    swapped_k_minus_1 = None
    for timestep in range(depth):
        if timestep == depth - 1:
            swapped_k_minus_1, _ = _readout(swapped_state[start_index])
        swapped_state = _transition(swapped_state, successors)
    swapped_k, _ = _readout(swapped_state[start_index])
    if rec_k_minus_1 is None or rec_k is None or swapped_k_minus_1 is None:
        raise ContractError("recurrent endpoint readout was not captured")
    if not torch.equal(rec_k_minus_1, swapped_k_minus_1):
        raise ContractError("endpoint swap changes REC readout at K-1")
    swap_delta = float((rec_k - swapped_k).abs().max().item())
    if torch.equal(rec_k, swapped_k) or not math.isfinite(swap_delta) or swap_delta <= 1e-12:
        raise ContractError("endpoint swap is unchanged at K")
    if marker_from_local_text(local[target_id]) == marker_from_local_text(local[donor_id]):
        raise ContractError("endpoint swap donor marker equals target")
    operation_counts["arm_executions"] += 1
    operation_counts["transition_calls"] += depth
    operation_counts["readout_calls"] += 2
    operation_counts["endpoint_swap_runs"] += 1

    perturb = []
    perturb_vector = torch.tensor(PERTURB_VECTOR, dtype=torch.float64, device="cpu")
    for arm in ARM_NAMES:
        radius = _radius(arm, depth)
        for location, distance in (("inside", radius), ("outside", radius + 1)):
            hidden = original_hidden
            successors = original_successors
            start_index = original_start_index
            node_index = original_node_index
            distance_indices = _distance_indices(row, node_index)
            if distance >= NODE_COUNT:
                raise ContractError("perturb location exceeds the ring")
            reference = baseline_vectors[arm]
            changed = hidden.clone()
            changed[distance_indices[distance]] += perturb_vector
            observed, _ = _readout(_arm_state(arm, changed, successors, depth)[start_index])
            delta = float((observed - reference).abs().max().item())
            if location == "inside" and (torch.equal(observed, reference) or delta <= 1e-12):
                raise ContractError("inside-radius perturbation did not change readout")
            if location == "outside" and not torch.equal(observed, reference):
                raise ContractError("outside-radius perturbation changed readout")
            perturb.append(
                {
                    "arm": arm,
                    "location": location,
                    "distance": distance,
                    "bitwise_equal": torch.equal(observed, reference),
                    "max_abs_delta": delta,
                }
            )
            operation_counts["arm_executions"] += 1
            operation_counts["transition_calls"] += transitions_for(arm)
            operation_counts["readout_calls"] += 1
            operation_counts["perturb_runs"] += 1

    expected = {
        "arm_executions": 16,
        "transition_calls": 12 + 7 * depth,
        "readout_calls": 18,
        "graph_encode_passes": 2,
        "local_node_encode_calls": 48,
        "synthetic_backward_calls": 5,
        "perturb_runs": 10,
        "endpoint_swap_runs": 1,
    }
    if operation_counts != expected:
        raise ContractError("numeric locality operation counts differ")
    return {
        "probe_index": probe["probe_index"],
        "row_id": row["row_id"],
        "depth": depth,
        "baseline_arms": baseline,
        "endpoint_swap": {
            "target_node_id": target_id,
            "donor_node_id": donor_id,
            "donor_distance": probe["donor_distance"],
            "k_minus_1_bitwise_equal": True,
            "k_non_bitwise": True,
            "k_max_abs_delta": swap_delta,
        },
        "perturbations": perturb,
        "counts": operation_counts,
    }


def run_symbolic_dependency_audit(banks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    final_checks = 0
    recurrent_checks = 0
    for split in SPLITS:
        for row in banks[split]["rows"]:
            ring, successor, _ = row_ring(row)
            dependencies = {node_id: {node_id} for node_id in ring}

            def transition_deps(current: dict[str, set[str]]) -> dict[str, set[str]]:
                return {node_id: current[node_id] | current[successor[node_id]] for node_id in ring}

            expected_radius = {
                "STATIC": 0,
                "FFN": 0,
                "FIXED_T4": 4,
                "RESET_K": 1,
                "REC_K": row["depth"],
            }
            for arm, radius in expected_radius.items():
                state = dependencies
                steps = 0 if arm in {"STATIC", "FFN"} else 4 if arm == "FIXED_T4" else 1 if arm == "RESET_K" else row["depth"]
                for _ in range(steps):
                    state = transition_deps(state)
                if state[ring[0]] != set(ring[: radius + 1]):
                    raise ContractError(f"symbolic dependency radius differs for {arm}")
                final_checks += 1
            state = dependencies
            for timestep in range(row["depth"] + 1):
                if state[ring[0]] != set(ring[: timestep + 1]):
                    raise ContractError("recurrent timestep dependency differs")
                recurrent_checks += 1
                if timestep < row["depth"]:
                    state = transition_deps(state)
    result = {"final_arm_checks": final_checks, "recurrent_timestep_checks": recurrent_checks}
    if result != expected_symbolic_counts():
        raise ContractError("symbolic dependency check counts differ")
    return result


def run_all_locality_probes(
    banks: dict[str, dict[str, Any]], selection: dict[str, Any]
) -> dict[str, Any]:
    probes = []
    for probe in selection["probes"]:
        rows = [row for row in banks[probe["split"]]["rows"] if row["row_id"] == probe["row_id"]]
        if len(rows) != 1:
            raise ContractError("selected probe row is absent/ambiguous")
        probes.append(run_locality_probe(rows[0], probe))
    totals = {
        key: sum(probe["counts"][key] for probe in probes)
        for key in probes[0]["counts"]
    }
    expected = {
        "arm_executions": 128,
        "transition_calls": 348,
        "readout_calls": 144,
        "graph_encode_passes": 16,
        "local_node_encode_calls": 384,
        "synthetic_backward_calls": 40,
        "perturb_runs": 80,
        "endpoint_swap_runs": 8,
    }
    if totals != expected:
        raise ContractError("aggregate locality operation counts differ")
    return {"probes": probes, "probe_count": len(probes), "counts": totals}


def validate_locality_policy(policy: dict[str, Any]) -> None:
    expected = {
        "encoder_input": "one_local_text_only",
        "encoder_graph_context": False,
        "transition_access": "one_immediate_successor_sparse_index_select",
        "two_hop_message_access": False,
        "dense_global_aggregate_access": False,
        "readout_input": "indexed_start_vector_shape_4_only",
        "reset_persists_state": False,
        "recurrent_persists_state": True,
        "trainable_parameters": 0,
    }
    if policy != expected:
        raise ContractError("locality policy exposes a forbidden information path")


def locality_policy() -> dict[str, Any]:
    return {
        "encoder_input": "one_local_text_only",
        "encoder_graph_context": False,
        "transition_access": "one_immediate_successor_sparse_index_select",
        "two_hop_message_access": False,
        "dense_global_aggregate_access": False,
        "readout_input": "indexed_start_vector_shape_4_only",
        "reset_persists_state": False,
        "recurrent_persists_state": True,
        "trainable_parameters": 0,
    }


def validate_locality_evidence(evidence: dict[str, Any], selection: dict[str, Any]) -> None:
    if set(evidence) != {"probes", "probe_count", "counts", "policy", "symbolic_dependencies"}:
        raise ContractError("locality evidence fields differ")
    validate_locality_policy(evidence["policy"])
    if evidence["probe_count"] != 8 or len(evidence["probes"]) != 8:
        raise ContractError("locality probe completion differs")
    if evidence["counts"] != {
        "arm_executions": 128,
        "transition_calls": 348,
        "readout_calls": 144,
        "graph_encode_passes": 16,
        "local_node_encode_calls": 384,
        "synthetic_backward_calls": 40,
        "perturb_runs": 80,
        "endpoint_swap_runs": 8,
    }:
        raise ContractError("locality aggregate counts differ")
    if evidence["symbolic_dependencies"] != expected_symbolic_counts():
        raise ContractError("symbolic locality evidence differs")
    for probe, selected in zip(evidence["probes"], selection["probes"], strict=True):
        if probe["probe_index"] != selected["probe_index"] or probe["row_id"] != selected["row_id"]:
            raise ContractError("locality evidence probe identity differs")
        if probe["counts"] != {
            "arm_executions": 16,
            "transition_calls": 12 + 7 * selected["depth"],
            "readout_calls": 18,
            "graph_encode_passes": 2,
            "local_node_encode_calls": 48,
            "synthetic_backward_calls": 5,
            "perturb_runs": 10,
            "endpoint_swap_runs": 1,
        }:
            raise ContractError("per-probe locality counts differ")
        if [item["arm"] for item in probe["baseline_arms"]] != ARM_NAMES:
            raise ContractError("baseline arm order differs")
        for arm in probe["baseline_arms"]:
            if not arm["inside_gradient_finite_nonzero"] or not arm["outside_gradient_bitwise_positive_zero"]:
                raise ContractError("autograd locality evidence rejected")
            if arm["arm"] == "RESET_K" and arm["reset_equals_single_transition"] is not True:
                raise ContractError("reset equality evidence rejected")
        swap = probe["endpoint_swap"]
        if swap["donor_distance"] <= probe["depth"] or not swap["k_minus_1_bitwise_equal"] or not swap["k_non_bitwise"]:
            raise ContractError("endpoint-swap locality evidence rejected")
        if not finite_float(swap["k_max_abs_delta"]) or swap["k_max_abs_delta"] <= 1e-12:
            raise ContractError("endpoint-swap delta evidence rejected")
        if len(probe["perturbations"]) != 10:
            raise ContractError("perturbation evidence count differs")
        for perturb in probe["perturbations"]:
            if not finite_float(perturb["max_abs_delta"]):
                raise ContractError("perturbation delta is nonfinite")
            if perturb["location"] == "inside":
                if perturb["bitwise_equal"] or perturb["max_abs_delta"] <= 1e-12:
                    raise ContractError("inside perturbation locality rejected")
            elif perturb["location"] == "outside":
                if not perturb["bitwise_equal"] or perturb["max_abs_delta"] != 0.0:
                    raise ContractError("outside perturbation locality rejected")
            else:
                raise ContractError("perturbation location differs")


def _rehash_row(row: dict[str, Any]) -> None:
    row["receiver_input_sha256"] = canonical_sha256(row["receiver_input"])
    row["row_sha256"] = canonical_sha256(row, omit="row_sha256")


def _rehash_bank(bank: dict[str, Any]) -> None:
    for ordinal, row in enumerate(bank["rows"]):
        row["ordinal"] = ordinal
        _rehash_row(row)
    bank["bank_sha256"] = canonical_sha256(bank, omit="bank_sha256")


def run_mechanism_tamper_audit(
    banks: dict[str, dict[str, Any]], selection: dict[str, Any], locality: dict[str, Any]
) -> dict[str, Any]:
    results = []

    def rejected(name: str, operation: Any) -> None:
        try:
            operation()
        except (ContractError, KeyError, IndexError, TypeError):
            results.append({"name": name, "rejected": True})
            return
        raise ContractError(f"mechanism tamper was accepted: {name}")

    def bank_mutation(name: str, mutate: Any) -> None:
        altered = copy.deepcopy(banks)
        mutate(altered)
        for bank in altered.values():
            _rehash_bank(bank)
        rejected(name, lambda: validate_banks(altered))

    bank_mutation("train_row_seed_changed", lambda value: value["train"]["rows"][0].__setitem__("generation_seed_u64_be", 0))
    bank_mutation("row_order_changed", lambda value: value["train"]["rows"].__setitem__(slice(0, 2), list(reversed(value["train"]["rows"][:2]))))
    bank_mutation("row_deleted", lambda value: value["train"]["rows"].pop())
    bank_mutation("row_duplicated", lambda value: value["train"]["rows"].append(copy.deepcopy(value["train"]["rows"][0])))
    bank_mutation("cross_split_row_id_reused", lambda value: value["validation"]["rows"][0].__setitem__("row_id", value["train"]["rows"][0]["row_id"]))
    bank_mutation(
        "node_id_reused",
        lambda value: value["train"]["rows"][1]["receiver_input"]["nodes"][0].__setitem__(
            "node_id", value["train"]["rows"][0]["receiver_input"]["nodes"][0]["node_id"]
        ),
    )
    bank_mutation(
        "node_id_in_local_text",
        lambda value: value["train"]["rows"][0]["receiver_input"]["nodes"][0].__setitem__(
            "local_text",
            value["train"]["rows"][0]["receiver_input"]["nodes"][0]["local_text"]
            + value["train"]["rows"][0]["receiver_input"]["nodes"][0]["node_id"],
        ),
    )
    bank_mutation(
        "answer_action_in_receiver_input",
        lambda value: value["train"]["rows"][0]["receiver_input"].__setitem__("answer_action", "ACT_Z1"),
    )
    bank_mutation("node_count_changed", lambda value: value["train"]["rows"][0]["receiver_input"]["nodes"].pop())
    bank_mutation("edge_count_changed", lambda value: value["train"]["rows"][0]["receiver_input"]["edges"].pop())
    bank_mutation(
        "ring_broken",
        lambda value: value["train"]["rows"][0]["receiver_input"]["edges"][0].__setitem__(
            "successor_node_id", value["train"]["rows"][0]["receiver_input"]["edges"][0]["source_node_id"]
        ),
    )
    bank_mutation(
        "target_distance_changed",
        lambda value: value["train"]["rows"][0]["supervision"].__setitem__(
            "target_node_id", value["train"]["rows"][0]["receiver_input"]["start_node_id"]
        ),
    )
    bank_mutation(
        "marker_balance_changed",
        lambda value: value["train"]["rows"][0]["receiver_input"]["nodes"][0].__setitem__(
            "local_text", value["train"]["rows"][0]["receiver_input"]["nodes"][1]["local_text"]
        ),
    )
    bank_mutation("action_balance_changed", lambda value: value["train"]["rows"][0].__setitem__("action_index", 1))
    bank_mutation("distractor_count_changed", lambda value: value["train"]["rows"][0].__setitem__("depth", 2))

    altered_selection = copy.deepcopy(selection)
    altered_selection["probes"][0]["donor_distance"] = altered_selection["probes"][0]["depth"]
    altered_selection["selection_sha256"] = canonical_sha256(altered_selection, omit="selection_sha256")
    rejected("endpoint_swap_donor_inside_radius", lambda: validate_probe_selection(altered_selection, banks))

    policy_mutations = {
        "encoder_given_graph_context": ("encoder_graph_context", True),
        "two_hop_message_access": ("two_hop_message_access", True),
        "dense_global_aggregate_access": ("dense_global_aggregate_access", True),
        "readout_given_all_nodes": ("readout_input", "whole_graph"),
        "reset_arm_persists_state": ("reset_persists_state", True),
        "recurrent_arm_resets_state": ("recurrent_persists_state", False),
    }
    for name, (key, value) in policy_mutations.items():
        altered = locality_policy()
        altered[key] = value
        rejected(name, lambda altered=altered: validate_locality_policy(altered))

    evidence_mutations = {
        "endpoint_swap_changes_k_minus_1": ("endpoint_swap", "k_minus_1_bitwise_equal", False),
        "endpoint_swap_unchanged_at_k": ("endpoint_swap", "k_non_bitwise", False),
        "autograd_outside_radius_nonzero": ("baseline_arms", 0, "outside_gradient_bitwise_positive_zero", False),
        "autograd_inside_radius_zero": ("baseline_arms", 0, "inside_gradient_finite_nonzero", False),
    }
    for name, path in evidence_mutations.items():
        altered = copy.deepcopy(locality)
        if path[0] == "endpoint_swap":
            altered["probes"][0][path[0]][path[1]] = path[2]
        else:
            altered["probes"][0][path[0]][path[1]][path[2]] = path[3]
        rejected(name, lambda altered=altered: validate_locality_evidence(altered, selection))
    if [row["name"] for row in results] != MECHANISM_TAMPERS or not all(row["rejected"] for row in results):
        raise ContractError("mechanism tamper audit order/completeness differs")
    return {"results": results, "rejected_count": len(results)}


EXPECTED_RUNTIME = {
    "python": "3.12.14",
    "torch_distribution": "2.11.0+cu128",
    "torch_runtime": "2.11.0+cu128",
    "transformers_distribution": "5.6.2",
    "tokenizers_distribution": "0.22.2",
    "pyarrow_distribution": "24.0.0",
    "cuda_visible_devices": "",
    "cuda_initialized_required": False,
    "sys_executable": "/home/ubuntu/rlm/prime-rl/.venv/bin/python3",
    "sys_prefix": "/home/ubuntu/rlm/prime-rl/.venv",
    "shared_project_pyproject_sha256": "504907808f992f1e6883f54c2695a4814ae77d6b80814239cbfc98d81a543656",
    "shared_project_uv_lock_sha256": "fca5fa6183345b5b68974078c38d58e0320f79eef13a695af11ceab12fdf36d5",
}
RESOURCE_BOUNDS = {
    "minimum_host_ram_gib": 8,
    "minimum_free_disk_gib": 8,
    "maximum_artifact_bytes": 32 * 2**20,
    "outer_timeout_seconds": 1200,
    "compute_timeout_seconds": 600,
    "audit_timeout_seconds": 180,
    "failure_audit_timeout_seconds": 180,
    "terminal_timeout_seconds": 60,
    "success_inner_maximum_seconds": 840,
    "compute_failure_maximum_seconds": 840,
    "audit_failure_maximum_seconds": 1020,
    "prepublication_terminal_failure_maximum_seconds": 1080,
    "startup_external_headroom_seconds": 120,
    "network": False,
    "output_root": "/home/ubuntu/rlm/outputs/q35-2b-h-iter-phase0-generator-locality-run1",
}
PHASE0_AUDITOR_EXPOSURE = {
    "train_rows": 96,
    "validation_rows": 48,
    "heldout_rows": 48,
    "train_locality_probes": 4,
    "validation_locality_probes": 2,
    "heldout_locality_probes": 2,
    "receiver_inputs": True,
    "supervision": True,
}
PHASE1_LEARNING_EXPOSURE = {
    "validation_model_or_tokenizer_calls": 0,
    "heldout_model_or_tokenizer_calls": 0,
    "validation_loss_or_learning_metrics": False,
    "heldout_loss_or_learning_metrics": False,
    "validation_learning_threshold_use": False,
    "heldout_learning_threshold_use": False,
    "validation_candidate_selection_use": False,
    "heldout_candidate_selection_use": False,
}
NETWORK_GUARD_CONTRACT = {
    "os_network_namespace": False,
    "python_guard_operations": [
        "socket.socket.connect",
        "socket.socket.connect_ex",
        "socket.create_connection",
        "socket.getaddrinfo",
    ],
    "audit_events": ["socket.connect", "socket.getaddrinfo"],
    "external_subprocess_allowlist": ["git rev-parse", "git status", "git show"],
}
DECISION_BOUNDARY = {
    "claim": "validation/heldout structurally and synthetic-locality auditor-opened; unopened to learned policy, model features/losses, threshold setting, or candidate selection",
    "phase1_learning_contract_set": False,
    "training_authorized": False,
    "model_or_gpu_authorized": False,
    "candidate_created": False,
    "nomination": False,
    "admission": False,
    "promotion": False,
    "live_trajectory_count": 0,
    "four_live_floor_unchanged": True,
    "phase0_auditor_opened": PHASE0_AUDITOR_EXPOSURE,
    "phase1_learning_exposure": PHASE1_LEARNING_EXPOSURE,
    "phase0_probe_selection_precommitted": True,
    "phase1_thresholds_present": False,
}
PHASE_CAP_SECONDS = {
    "compute": 600,
    "audit": 180,
    "failure_audit": 180,
    "terminal_publication": 60,
}
PHASE_RECORD_KEYS = {
    "phase",
    "entered_ns_since_start",
    "exited_ns_since_start",
    "duration_ns",
    "outcome",
    "cap_ns",
    "alarm_after_ns",
    "alarm_safety_margin_ns",
    "timeout_observed",
    "alarm_requested_after_ns",
    "timeout_observed_duration_ns",
    "delivery_overrun_ns",
    "timing_cap_exceeded",
}


def validate_phase_records(records: list[dict[str, Any]], *, failure_status: str | None) -> int:
    prior_exit = 0
    for row in records:
        if set(row) != PHASE_RECORD_KEYS or row["phase"] not in PHASE_CAP_SECONDS:
            raise ContractError("Phase-0 phase record fields differ")
        if row["outcome"] not in {"completed", "error"} or any(
            not isinstance(row[key], int) or isinstance(row[key], bool) or row[key] < 0
            for key in (
                "entered_ns_since_start",
                "exited_ns_since_start",
                "duration_ns",
                "cap_ns",
                "alarm_after_ns",
                "alarm_safety_margin_ns",
                "alarm_requested_after_ns",
                "delivery_overrun_ns",
            )
        ):
            raise ContractError("Phase-0 phase record values differ")
        cap_ns = PHASE_CAP_SECONDS[row["phase"]] * 1_000_000_000
        if (
            row["entered_ns_since_start"] < prior_exit
            or row["duration_ns"] != row["exited_ns_since_start"] - row["entered_ns_since_start"]
            or row["cap_ns"] != cap_ns
            or row["alarm_after_ns"] != cap_ns - 1_000_000_000
            or row["alarm_requested_after_ns"] != row["alarm_after_ns"]
            or row["alarm_safety_margin_ns"] != 1_000_000_000
            or row["alarm_after_ns"] + row["alarm_safety_margin_ns"] != row["cap_ns"]
        ):
            raise ContractError("Phase-0 phase chronology/cap differs")
        if row["timeout_observed"] is True:
            if row["outcome"] != "error" or row["timeout_observed_duration_ns"] != row["duration_ns"]:
                raise ContractError("Phase-0 timeout observation differs")
            if row["delivery_overrun_ns"] != max(0, row["duration_ns"] - cap_ns):
                raise ContractError("Phase-0 timeout overrun arithmetic differs")
        elif (
            row["timeout_observed"] is not False
            or row["timeout_observed_duration_ns"] is not None
            or row["delivery_overrun_ns"] != 0
        ):
            raise ContractError("Phase-0 non-timeout phase evidence differs")
        if row["timing_cap_exceeded"] is not (row["duration_ns"] > cap_ns):
            raise ContractError("Phase-0 timing-cap flag differs")
        if row["duration_ns"] > cap_ns and (
            failure_status != "infrastructure_invalid" or row["timeout_observed"] is not True
        ):
            raise ContractError("Phase-0 phase duration exceeds cap")
        prior_exit = row["exited_ns_since_start"]
    return prior_exit


def validate_plan(plan: dict[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "status",
        "mechanism",
        "run_identity",
        "mechanism_code_commit",
        "execution_authorization",
        "output_root",
        "asset_sha256",
        "historical_source_commits",
        "runtime",
        "resource_bounds",
        "generator_contract",
        "locality_contract",
        "terminal_contract",
        "memory_label_schedule",
        "safety_boundary",
        "full_freeze",
        "plan_sha256",
    }
    if set(plan) != expected_keys:
        raise ContractError("Phase-0 plan fields differ")
    validate_no_threshold_fields(plan)
    if plan["schema_version"] != PLAN_SCHEMA or plan["status"] != "preregistered":
        raise ContractError("Phase-0 plan schema/status differs")
    if plan["mechanism"] != MECHANISM or plan["run_identity"] != RUN_IDENTITY:
        raise ContractError("Phase-0 plan identity differs")
    if not re.fullmatch(r"[0-9a-f]{40}", plan["mechanism_code_commit"]):
        raise ContractError("Phase-0 mechanism commit is malformed")
    if plan["execution_authorization"] != "root_and_gatekeeper_review_required":
        raise ContractError("Phase-0 execution authority changed")
    if plan["output_root"] != RESOURCE_BOUNDS["output_root"] or plan["runtime"] != EXPECTED_RUNTIME:
        raise ContractError("Phase-0 runtime/output root differs")
    if plan["resource_bounds"] != RESOURCE_BOUNDS:
        raise ContractError("Phase-0 resource bounds differ")
    assets = plan["asset_sha256"]
    if not isinstance(assets, dict) or not assets or any(
        not isinstance(path, str) or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
        for path, digest in assets.items()
    ):
        raise ContractError("Phase-0 asset closure is malformed")
    if list(assets) != sorted(assets):
        raise ContractError("Phase-0 asset closure is not sorted")
    if plan["historical_source_commits"] != {
        "a_lane": "a8f347c9a5fdf1c2d532c6527ce169cff0000a07",
        "b_lane": "4ae0308094a71d13520554da40cfe6375438b610",
    }:
        raise ContractError("Phase-0 historical source commits differ")
    if plan["generator_contract"] != {
        "node_count": 24,
        "feature_dim": 4,
        "train_depths": [1, 2, 3, 4],
        "validation_depths": [5, 6],
        "heldout_depths": [7, 8],
        "rows_per_depth": 24,
        "replicates_per_depth_action": 6,
        "total_rows": 192,
        "production_boundary": "receiver_input_only",
        "answer_free_receiver": True,
    }:
        raise ContractError("Phase-0 generator contract differs")
    if plan["locality_contract"] != {
        "dtype": "torch.float64_cpu",
        "transition": "0.75*self_plus_0.25*immediate_successor_sparse_index_select",
        "readout": "indexed_start_vector_shape_4_only",
        "arms": ARM_NAMES,
        "probe_count": 8,
        "operation_counts": {
            "arm_executions": 128,
            "transition_calls": 348,
            "readout_calls": 144,
            "graph_encode_passes": 16,
            "local_node_encode_calls": 384,
            "synthetic_backward_calls": 40,
            "perturb_runs": 80,
            "endpoint_swap_runs": 8,
        },
        "symbolic_counts": expected_symbolic_counts(),
    }:
        raise ContractError("Phase-0 locality contract differs")
    if plan["terminal_contract"] != {
        "success_status": "h_iter_phase0_generator_locality_validated",
        "mechanism_failure_status": "h_iter_phase0_generator_locality_incomplete",
        "infrastructure_failure_status": "infrastructure_invalid",
        "success_file": "PROOF.json",
        "failure_file": "FAILURE.json",
        "exclusive_atomic": True,
        "canonical_roundtrip_twice": True,
    }:
        raise ContractError("Phase-0 terminal contract differs")
    labels = memory_labels()
    if plan["memory_label_schedule"] != {
        "labels": labels,
        "count": len(labels),
        "sha256": canonical_sha256(labels),
    }:
        raise ContractError("Phase-0 memory label schedule differs")
    if plan["safety_boundary"] != DECISION_BOUNDARY | {
        "coordinator_e33_loaded": False,
        "worker_h176_loaded": False,
        "tokenizer_calls": 0,
        "model_forwards": 0,
        "model_backwards": 0,
        "optimizer_steps": 0,
        "synthetic_cpu_backwards": 40,
        "transformers_modeling_imports": 0,
        "network_guard": NETWORK_GUARD_CONTRACT,
    }:
        raise ContractError("Phase-0 safety boundary differs")
    if plan["full_freeze"] != {
        "execution_head_is_exact_clean_child_of_mechanism": True,
        "git_tree_and_assets_unchanged_pre_post": True,
        "phase0_plan_external_file_hash_required": True,
        "phase0_plan_sidecar_required": True,
        "historical_sources_reopened_from_exact_git_commits": True,
    }:
        raise ContractError("Phase-0 full-freeze contract differs")
    if plan["plan_sha256"] != canonical_sha256(plan, omit="plan_sha256"):
        raise ContractError("Phase-0 internal plan hash differs")


def validate_proof(
    proof: dict[str, Any],
    *,
    plan: dict[str, Any],
    banks: dict[str, dict[str, Any]],
    selection: dict[str, Any],
    schedule: dict[str, Any],
    overlap: dict[str, Any],
    expected_execution_commit: str,
    expected_plan_file_sha256: str,
    require_receipt_tampers: bool = True,
    require_final_timing: bool = True,
) -> None:
    keys = {
        "schema_version",
        "status",
        "mechanism",
        "run_identity",
        "execution_commit",
        "mechanism_code_commit",
        "plan_file_sha256",
        "plan_sha256",
        "runtime",
        "asset_audit",
        "banks",
        "structural_audit",
        "overlap_audit",
        "operation_schedule",
        "locality",
        "tamper_audit",
        "counts",
        "safety",
        "resources",
        "memory",
        "full_freeze",
        "decision_boundary",
        "proof_sha256",
    }
    if set(proof) != keys:
        raise ContractError("Phase-0 proof fields differ")
    validate_no_threshold_fields(proof)
    if proof["schema_version"] != PROOF_SCHEMA or proof["status"] != "h_iter_phase0_generator_locality_validated":
        raise ContractError("Phase-0 proof schema/status differs")
    if proof["mechanism"] != MECHANISM or proof["run_identity"] != RUN_IDENTITY:
        raise ContractError("Phase-0 proof identity differs")
    if proof["mechanism_code_commit"] != plan["mechanism_code_commit"] or proof["plan_sha256"] != plan["plan_sha256"]:
        raise ContractError("Phase-0 proof plan/mechanism binding differs")
    if proof["execution_commit"] != expected_execution_commit or not re.fullmatch(
        r"[0-9a-f]{40}", expected_execution_commit
    ):
        raise ContractError("Phase-0 execution commit differs from launch authority")
    if proof["plan_file_sha256"] != expected_plan_file_sha256 or not re.fullmatch(
        r"[0-9a-f]{64}", expected_plan_file_sha256
    ):
        raise ContractError("Phase-0 external plan hash differs from launch authority")
    audit = proof["asset_audit"]
    if set(audit) != {"before", "after", "before_after_equal", "all_plan_assets_exact"}:
        raise ContractError("Phase-0 asset audit fields differ")
    if audit["before"] != plan["asset_sha256"] or audit["after"] != plan["asset_sha256"]:
        raise ContractError("Phase-0 asset audit differs from plan")
    if audit["before_after_equal"] is not True or audit["all_plan_assets_exact"] is not True:
        raise ContractError("Phase-0 asset audit is incomplete")
    runtime = proof["runtime"]
    if set(runtime) != {*EXPECTED_RUNTIME, "cuda_initialized_before", "cuda_initialized_after"}:
        raise ContractError("Phase-0 runtime evidence fields differ")
    if {key: runtime[key] for key in EXPECTED_RUNTIME} != EXPECTED_RUNTIME:
        raise ContractError("Phase-0 runtime evidence differs")
    if runtime["cuda_initialized_before"] is not False or runtime["cuda_initialized_after"] is not False:
        raise ContractError("CUDA initialized during model-free Phase-0")
    expected_bank_evidence = {
        split: {
            "file_sha256": plan["asset_sha256"][f"{ARTIFACT_DIR_REL}/{split}-bank.json"],
            "bank_sha256": banks[split]["bank_sha256"],
            "row_count": len(banks[split]["rows"]),
        }
        for split in SPLITS
    }
    if proof["banks"] != expected_bank_evidence:
        raise ContractError("Phase-0 bank proof differs")
    if proof["structural_audit"] != validate_banks(banks):
        raise ContractError("Phase-0 structural proof differs")
    if proof["overlap_audit"] != {
        "file_sha256": plan["asset_sha256"][f"{ARTIFACT_DIR_REL}/overlap-evidence.json"],
        "overlap_sha256": overlap["overlap_sha256"],
        "source_record_count": 38,
        "all_intersections_empty": True,
        "regenerated_byte_identical": True,
    }:
        raise ContractError("Phase-0 overlap proof differs")
    if proof["operation_schedule"] != {
        "file_sha256": plan["asset_sha256"][f"{ARTIFACT_DIR_REL}/operation-schedule.json"],
        "schedule_sha256": schedule["schedule_sha256"],
        "operation_count": 128,
    }:
        raise ContractError("Phase-0 operation schedule proof differs")
    validate_locality_evidence(proof["locality"], selection)
    tamper = proof["tamper_audit"]
    if set(tamper) != {"mechanism", "receipt"}:
        raise ContractError("Phase-0 tamper audit fields differ")
    if [row["name"] for row in tamper["mechanism"]["results"]] != MECHANISM_TAMPERS:
        raise ContractError("Phase-0 mechanism tamper order differs")
    if tamper["mechanism"]["rejected_count"] != 26 or not all(row["rejected"] is True for row in tamper["mechanism"]["results"]):
        raise ContractError("Phase-0 mechanism tamper audit differs")
    if require_receipt_tampers:
        if [row["name"] for row in tamper["receipt"]["results"]] != RECEIPT_TAMPERS:
            raise ContractError("Phase-0 receipt tamper order differs")
        if tamper["receipt"]["rejected_count"] != 12 or not all(row["rejected"] is True for row in tamper["receipt"]["results"]):
            raise ContractError("Phase-0 receipt tamper audit differs")
    if proof["counts"] != {
        "structural_rows": 192,
        "locality_probes": 8,
        "symbolic_final_arm_checks": 960,
        "symbolic_recurrent_timestep_checks": 1056,
        "synthetic_cpu_backwards": 40,
        "model_or_transformer_forwards": 0,
        "model_backwards": 0,
        "optimizer_steps": 0,
    }:
        raise ContractError("Phase-0 top-level counts differ")
    safety = proof["safety"]
    if set(safety) != {
        "coordinator_e33_loaded",
        "worker_h176_loaded",
        "tokenizer_loaded",
        "candidate_created",
        "checkpoint_created",
        "model_updated",
        "phase0_auditor_opened",
        "phase1_learning_exposure",
        "phase0_probe_selection_precommitted",
        "phase1_thresholds_present",
        "network_guard",
        "transformers_modeling_modules",
        "pretrained_model_objects",
        "tokenizer_objects",
        "optimizer_objects",
        "object_census_method",
        "cuda_observation_method",
        "relevant_modules_absent_for_preimport_inference",
        "output_inventory_before_terminal",
        "static_forbidden_model_call_sites",
        "tokenizer_calls",
        "model_forwards",
        "model_backwards",
        "synthetic_cpu_backwards",
        "optimizer_steps",
    }:
        raise ContractError("Phase-0 safety evidence fields differ")
    expected_false = (
        "coordinator_e33_loaded",
        "worker_h176_loaded",
        "tokenizer_loaded",
        "candidate_created",
        "checkpoint_created",
        "model_updated",
    )
    if any(safety.get(key) is not False for key in expected_false):
        raise ContractError("Phase-0 safety boundary changed")
    if (
        safety.get("transformers_modeling_modules") != []
        or safety.get("pretrained_model_objects") != 0
        or safety.get("tokenizer_objects") != 0
        or safety.get("optimizer_objects") != 0
        or safety.get("output_inventory_before_terminal") != []
        or safety.get("static_forbidden_model_call_sites") != []
    ):
        raise ContractError("model/optimizer object appeared in Phase-0")
    if safety["phase0_auditor_opened"] != PHASE0_AUDITOR_EXPOSURE:
        raise ContractError("Phase-0 auditor exposure differs")
    if safety["phase1_learning_exposure"] != PHASE1_LEARNING_EXPOSURE:
        raise ContractError("Phase-1 learning exposure differs")
    if safety["phase0_probe_selection_precommitted"] is not True or safety["phase1_thresholds_present"] is not False:
        raise ContractError("Phase-0 exposure boundary differs")
    if (
        safety["object_census_method"]
        != "gc_mro_scan_without_importing_model_tokenizer_or_optimizer_classes"
        or safety["cuda_observation_method"] != "torch.cuda.is_initialized"
        or safety["relevant_modules_absent_for_preimport_inference"] is not False
    ):
        raise ContractError("Phase-0 success safety observation method differs")
    network = safety["network_guard"]
    if set(network) != {
        *NETWORK_GUARD_CONTRACT,
        "installed",
        "wrappers_restored",
        "audit_hook_persistent",
        "attempt_count",
    }:
        raise ContractError("Phase-0 network guard fields differ")
    if {key: network[key] for key in NETWORK_GUARD_CONTRACT} != NETWORK_GUARD_CONTRACT:
        raise ContractError("Phase-0 network guard contract differs")
    if network["installed"] is not True or network["wrappers_restored"] is not True:
        raise ContractError("Phase-0 network guard lifecycle differs")
    if network["audit_hook_persistent"] is not True or network["attempt_count"] != 0:
        raise ContractError("Phase-0 network isolation failed")
    if {
        key: safety.get(key)
        for key in ("tokenizer_calls", "model_forwards", "model_backwards", "synthetic_cpu_backwards", "optimizer_steps")
    } != {
        "tokenizer_calls": 0,
        "model_forwards": 0,
        "model_backwards": 0,
        "synthetic_cpu_backwards": 40,
        "optimizer_steps": 0,
    }:
        raise ContractError("Phase-0 safety call counts differ")
    if proof["decision_boundary"] != DECISION_BOUNDARY:
        raise ContractError("Phase-0 decision boundary differs")
    memory = proof["memory"]
    labels = memory_labels()
    if memory.get("labels") != labels or memory.get("label_sha256") != canonical_sha256(labels):
        raise ContractError("Phase-0 memory label evidence differs")
    if memory.get("count") != len(labels) or len(memory.get("rows", [])) != len(labels):
        raise ContractError("Phase-0 memory row count differs")
    prior_peak = 0
    for label, row in zip(labels, memory["rows"], strict=True):
        if set(row) != {"label", "rss_bytes", "peak_rss_bytes"} or row["label"] != label:
            raise ContractError("Phase-0 memory row schema/order differs")
        if any(not isinstance(row[key], int) or isinstance(row[key], bool) or row[key] < 0 for key in ("rss_bytes", "peak_rss_bytes")):
            raise ContractError("Phase-0 memory value differs")
        if row["peak_rss_bytes"] < row["rss_bytes"] or row["peak_rss_bytes"] < prior_peak:
            raise ContractError("Phase-0 memory peak is invalid/nonmonotonic")
        prior_peak = row["peak_rss_bytes"]
    resources = proof["resources"]
    if set(resources) != {
        "bounds",
        "host_ram_bytes",
        "free_disk_bytes_preflight",
        "free_disk_bytes_postflight",
        "artifact_bytes_before_terminal",
        "completed_phase_records",
        "final_terminal_publication",
        "prepublication_elapsed_ns",
    }:
        raise ContractError("Phase-0 resource evidence fields differ")
    if resources.get("bounds") != RESOURCE_BOUNDS:
        raise ContractError("Phase-0 resource evidence bounds differ")
    if not isinstance(resources.get("host_ram_bytes"), int) or resources["host_ram_bytes"] < 8 * 2**30:
        raise ContractError("Phase-0 host RAM evidence differs")
    for key in ("free_disk_bytes_preflight", "free_disk_bytes_postflight"):
        if not isinstance(resources.get(key), int) or resources[key] < 8 * 2**30:
            raise ContractError("Phase-0 free-disk evidence differs")
    if require_final_timing:
        records = resources["completed_phase_records"]
        if not isinstance(records, list) or [row.get("phase") for row in records] != ["compute", "audit"]:
            raise ContractError("Phase-0 success phase order differs")
        if any(row.get("outcome") != "completed" for row in records):
            raise ContractError("Phase-0 success phase outcome differs")
        prior_exit = validate_phase_records(records, failure_status=None)
        terminal = resources["final_terminal_publication"]
        if terminal != {
            "phase": "terminal_publication",
            "entered_ns_since_start": resources["prepublication_elapsed_ns"],
            "limit_ns": 60_000_000_000,
            "completion_observable_inside_terminal": False,
            "self_reference_boundary": "post_write_fsync_reopen_validation_and_process_exit_are_external_to_immutable_terminal_bytes",
        }:
            raise ContractError("Phase-0 terminal self-reference boundary differs")
        if not isinstance(resources["prepublication_elapsed_ns"], int) or isinstance(resources["prepublication_elapsed_ns"], bool):
            raise ContractError("Phase-0 prepublication time differs")
        if resources["prepublication_elapsed_ns"] < prior_exit or resources["prepublication_elapsed_ns"] > 840_000_000_000:
            raise ContractError("Phase-0 success prepublication budget differs")
    if not isinstance(resources["artifact_bytes_before_terminal"], int) or resources["artifact_bytes_before_terminal"] != 0:
        raise ContractError("Phase-0 preterminal artifact inventory differs")
    freeze = proof["full_freeze"]
    if set(freeze) != {
        "head_before",
        "head_after",
        "tree_before",
        "tree_after",
        "status_before",
        "status_after",
        "mechanism_exact_parent",
        "head_tree_unchanged",
        "plan_sidecar_exact",
        "historical_sources_reopened",
    }:
        raise ContractError("Phase-0 full-freeze evidence fields differ")
    if freeze["head_before"] != proof["execution_commit"] or freeze["head_after"] != proof["execution_commit"]:
        raise ContractError("Phase-0 execution HEAD changed")
    if freeze["tree_before"] != freeze["tree_after"] or freeze["status_before"] != "" or freeze["status_after"] != "":
        raise ContractError("Phase-0 execution tree/worktree changed")
    if any(freeze[key] is not True for key in ("mechanism_exact_parent", "head_tree_unchanged", "plan_sidecar_exact", "historical_sources_reopened")):
        raise ContractError("Phase-0 full-freeze evidence is incomplete")
    if proof["proof_sha256"] != canonical_sha256(proof, omit="proof_sha256"):
        raise ContractError("Phase-0 proof self hash differs")


def validate_failure(
    failure: dict[str, Any],
    *,
    plan: dict[str, Any],
    expected_execution_commit: str,
    expected_plan_file_sha256: str,
) -> None:
    validate_plan(plan)
    if set(failure) != {
        "schema_version",
        "status",
        "mechanism",
        "run_identity",
        "error_type",
        "error",
        "traceback",
        "execution_commit",
        "mechanism_code_commit",
        "authorized_plan_file_sha256",
        "plan_file_sha256",
        "plan_sha256",
        "actual_safety",
        "completed_phase_records",
        "final_terminal_publication",
        "prepublication_elapsed_ns",
        "partial_memory",
        "full_freeze_failure_audit",
        "output_inventory_before_failure",
        "candidate_created",
        "checkpoint_created",
        "model_updated",
        "failure_sha256",
    }:
        raise ContractError("Phase-0 failure fields differ")
    validate_no_threshold_fields(failure)
    if failure["schema_version"] != FAILURE_SCHEMA or failure["status"] not in {
        "h_iter_phase0_generator_locality_incomplete",
        "infrastructure_invalid",
    }:
        raise ContractError("Phase-0 failure schema/status differs")
    if failure["mechanism"] != MECHANISM or failure["run_identity"] != RUN_IDENTITY:
        raise ContractError("Phase-0 failure identity differs")
    if failure["execution_commit"] != expected_execution_commit or not re.fullmatch(
        r"[0-9a-f]{40}", expected_execution_commit
    ):
        raise ContractError("Phase-0 failure execution commit differs from launch authority")
    if failure["authorized_plan_file_sha256"] != expected_plan_file_sha256 or not re.fullmatch(
        r"[0-9a-f]{64}", expected_plan_file_sha256
    ):
        raise ContractError("Phase-0 failure plan authority differs")
    if (
        failure["mechanism_code_commit"] != plan["mechanism_code_commit"]
        or failure["plan_file_sha256"] != expected_plan_file_sha256
        or failure["plan_sha256"] != plan["plan_sha256"]
    ):
        raise ContractError("Phase-0 failure plan/mechanism binding differs")
    if not isinstance(failure["error_type"], str) or not failure["error_type"]:
        raise ContractError("Phase-0 failure error type is absent")
    if not isinstance(failure["error"], str) or not isinstance(failure["traceback"], str):
        raise ContractError("Phase-0 failure error/traceback differs")
    records = failure["completed_phase_records"]
    allowed_sequences = [
        [("compute", "error"), ("failure_audit", "completed")],
        [("compute", "completed"), ("audit", "error"), ("failure_audit", "completed")],
        [
            ("compute", "completed"),
            ("audit", "completed"),
            ("terminal_publication", "error"),
            ("failure_audit", "completed"),
        ],
    ]
    if not isinstance(records, list) or [(row.get("phase"), row.get("outcome")) for row in records] not in allowed_sequences:
        raise ContractError("Phase-0 failure phase sequence differs")
    prior_exit = validate_phase_records(records, failure_status=failure["status"])
    terminal = failure["final_terminal_publication"]
    if terminal != {
        "phase": "terminal_publication",
        "entered_ns_since_start": failure["prepublication_elapsed_ns"],
        "limit_ns": 60_000_000_000,
        "completion_observable_inside_terminal": False,
        "self_reference_boundary": "post_write_fsync_reopen_validation_and_process_exit_are_external_to_immutable_terminal_bytes",
    }:
        raise ContractError("Phase-0 failure terminal boundary differs")
    if not isinstance(failure["prepublication_elapsed_ns"], int) or isinstance(failure["prepublication_elapsed_ns"], bool):
        raise ContractError("Phase-0 failure prepublication time differs")
    sequence = [(row["phase"], row["outcome"]) for row in records]
    maximum = {
        tuple(allowed_sequences[0]): 840_000_000_000,
        tuple(allowed_sequences[1]): 1_020_000_000_000,
        tuple(allowed_sequences[2]): 1_080_000_000_000,
    }[tuple(sequence)]
    if failure["prepublication_elapsed_ns"] < prior_exit or failure["prepublication_elapsed_ns"] > maximum:
        raise ContractError("Phase-0 failure prepublication budget differs")
    safety = failure["actual_safety"]
    required_safety = {
        "cuda_visible_devices",
        "torch_imported",
        "cuda_initialized",
        "transformers_modeling_modules",
        "pretrained_model_objects",
        "tokenizer_objects",
        "optimizer_objects",
        "output_inventory",
        "candidate_files",
        "checkpoint_files",
        "static_forbidden_model_call_sites",
        "observation_complete",
        "object_census_method",
        "cuda_observation_method",
        "relevant_modules_absent_for_preimport_inference",
        "network_guard",
    }
    if set(safety) != required_safety or safety["observation_complete"] is not True:
        raise ContractError("Phase-0 failure safety observation is incomplete")
    unsafe = (
        safety["cuda_visible_devices"] != ""
        or safety["cuda_initialized"] is not False
        or safety["transformers_modeling_modules"] != []
        or safety["pretrained_model_objects"] != 0
        or safety["tokenizer_objects"] != 0
        or safety["optimizer_objects"] != 0
        or safety["output_inventory"] != []
        or safety["candidate_files"] != []
        or safety["checkpoint_files"] != []
        or safety["static_forbidden_model_call_sites"] != []
    )
    if unsafe and failure["status"] != "infrastructure_invalid":
        raise ContractError("unsafe Phase-0 failure is not infrastructure-invalid")
    for key in (
        "transformers_modeling_modules",
        "output_inventory",
        "candidate_files",
        "checkpoint_files",
        "static_forbidden_model_call_sites",
    ):
        if not isinstance(safety[key], list) or any(not isinstance(item, str) for item in safety[key]):
            raise ContractError("Phase-0 failure safety list differs")
    for key in ("pretrained_model_objects", "tokenizer_objects", "optimizer_objects"):
        if not isinstance(safety[key], int) or isinstance(safety[key], bool) or safety[key] < 0:
            raise ContractError("Phase-0 failure object count differs")
    if safety["object_census_method"] != "gc_mro_scan_without_importing_model_tokenizer_or_optimizer_classes":
        raise ContractError("Phase-0 failure object census method differs")
    if safety["torch_imported"] is False:
        if safety["cuda_observation_method"] != "torch_module_absence_proves_no_torch_cuda_runtime_contact":
            raise ContractError("Phase-0 pre-Torch CUDA observation differs")
        if safety["relevant_modules_absent_for_preimport_inference"] is not True:
            raise ContractError("Phase-0 pre-Torch module absence was not proven")
    elif (
        safety["torch_imported"] is not True
        or safety["cuda_observation_method"] != "torch.cuda.is_initialized"
        or safety["relevant_modules_absent_for_preimport_inference"] is not False
    ):
        raise ContractError("Phase-0 post-Torch CUDA observation differs")
    network = safety["network_guard"]
    if set(network) != {
        *NETWORK_GUARD_CONTRACT,
        "installed",
        "wrappers_restored",
        "audit_hook_persistent",
        "attempt_count",
    } or {key: network[key] for key in NETWORK_GUARD_CONTRACT} != NETWORK_GUARD_CONTRACT:
        raise ContractError("Phase-0 failure network guard differs")
    if any(
        not isinstance(network[key], bool)
        for key in ("installed", "wrappers_restored", "audit_hook_persistent")
    ) or not isinstance(network["attempt_count"], int) or isinstance(network["attempt_count"], bool) or network["attempt_count"] < 0:
        raise ContractError("Phase-0 failure network guard values differ")
    if (
        network["installed"] is not True
        or network["wrappers_restored"] is not True
        or network["audit_hook_persistent"] is not True
        or network["attempt_count"] != 0
    ):
        if failure["status"] != "infrastructure_invalid":
            raise ContractError("Phase-0 network guard failure is not infrastructure-invalid")
    if failure["candidate_created"] != bool(safety["candidate_files"]):
        raise ContractError("Phase-0 failure candidate observation differs")
    if failure["checkpoint_created"] != bool(safety["checkpoint_files"]):
        raise ContractError("Phase-0 failure checkpoint observation differs")
    if failure["model_updated"] != (safety["pretrained_model_objects"] != 0):
        raise ContractError("Phase-0 failure model-update observation differs")
    if failure["output_inventory_before_failure"] != safety["output_inventory"]:
        raise ContractError("Phase-0 failure output inventory differs")
    partial = failure["partial_memory"]
    if set(partial) != {"expected_labels", "rows"} or partial["expected_labels"] != memory_labels():
        raise ContractError("Phase-0 failure memory schema differs")
    if [row.get("label") for row in partial["rows"]] != memory_labels()[: len(partial["rows"])]:
        raise ContractError("Phase-0 failure memory prefix differs")
    prior_peak = 0
    for row in partial["rows"]:
        if set(row) != {"label", "rss_bytes", "peak_rss_bytes"}:
            raise ContractError("Phase-0 failure memory row fields differ")
        if any(
            not isinstance(row[key], int) or isinstance(row[key], bool) or row[key] < 0
            for key in ("rss_bytes", "peak_rss_bytes")
        ):
            raise ContractError("Phase-0 failure memory value differs")
        if row["peak_rss_bytes"] < row["rss_bytes"] or row["peak_rss_bytes"] < prior_peak:
            raise ContractError("Phase-0 failure memory peak differs")
        prior_peak = row["peak_rss_bytes"]
    audit = failure["full_freeze_failure_audit"]
    if not isinstance(audit, dict) or set(audit) != {
        "head",
        "tree",
        "status",
        "plan_file_sha256",
        "plan_sha256",
        "plan_sidecar_sha256",
        "plan_asset_hashes",
        "errors",
    } or not isinstance(audit["errors"], list):
        raise ContractError("Phase-0 failure freeze audit differs")
    if audit["errors"]:
        if failure["status"] != "infrastructure_invalid" or any(
            not isinstance(error, dict)
            or set(error) != {"check", "error"}
            or not all(isinstance(value, str) and value for value in error.values())
            for error in audit["errors"]
        ):
            raise ContractError("Phase-0 incomplete failure audit is not fail-closed")
        error_checks = {error["check"] for error in audit["errors"]}
        expected_mismatch_errors = {
            "head_exact": "execution HEAD differs",
            "status_clean": "worktree is not clean",
            "plan_file_exact": "external plan file hash differs",
            "plan_sidecar_exact": "external plan sidecar differs",
        }
        if any(
            error["check"] in expected_mismatch_errors
            and error["error"] != expected_mismatch_errors[error["check"]]
            for error in audit["errors"]
        ):
            raise ContractError("Phase-0 failure mismatch error evidence differs")
        mismatch_truth = {
            "head_exact": audit["head"] != expected_execution_commit,
            "status_clean": audit["status"] != "",
            "plan_file_exact": audit["plan_file_sha256"] != expected_plan_file_sha256,
            "plan_sidecar_exact": audit["plan_sidecar_sha256"]
            != sha256_bytes(f"{expected_plan_file_sha256}\n".encode()),
        }
        if any((name in error_checks) is not mismatch for name, mismatch in mismatch_truth.items()):
            raise ContractError("Phase-0 failure mismatch/error closure differs")
        if (
            audit["head"] is not None
            and audit["head"] != expected_execution_commit
            and "head_exact" not in error_checks
        ):
            raise ContractError("Phase-0 observed failure HEAD differs")
        if audit["tree"] is not None and (
            not isinstance(audit["tree"], str) or not re.fullmatch(r"[0-9a-f]{40}", audit["tree"])
        ):
            raise ContractError("Phase-0 observed failure tree differs")
        if audit["status"] is not None and audit["status"] != "" and "status_clean" not in error_checks:
            raise ContractError("Phase-0 observed failure worktree differs")
        if (
            audit["plan_file_sha256"] is not None
            and audit["plan_file_sha256"] != expected_plan_file_sha256
            and "plan_file_exact" not in error_checks
        ):
            raise ContractError("Phase-0 observed failure plan file differs")
        if audit["plan_sha256"] is not None and audit["plan_sha256"] != plan["plan_sha256"]:
            raise ContractError("Phase-0 observed failure internal plan differs")
        if (
            audit["plan_sidecar_sha256"] is not None
            and audit["plan_sidecar_sha256"] != sha256_bytes(f"{expected_plan_file_sha256}\n".encode())
            and "plan_sidecar_exact" not in error_checks
        ):
            raise ContractError("Phase-0 observed failure plan sidecar differs")
        if audit["plan_asset_hashes"] is not None and audit["plan_asset_hashes"] != plan["asset_sha256"]:
            raise ContractError("Phase-0 observed failure asset map differs")
    else:
        if (
            audit["head"] != expected_execution_commit
            or not isinstance(audit["tree"], str)
            or not re.fullmatch(r"[0-9a-f]{40}", audit["tree"])
            or audit["status"] != ""
            or audit["plan_file_sha256"] != expected_plan_file_sha256
            or audit["plan_sha256"] != failure["plan_sha256"]
            or audit["plan_sidecar_sha256"] != sha256_bytes(f"{expected_plan_file_sha256}\n".encode())
            or audit["plan_asset_hashes"] != plan["asset_sha256"]
            or failure["plan_file_sha256"] != expected_plan_file_sha256
            or failure["mechanism_code_commit"] != plan["mechanism_code_commit"]
        ):
            raise ContractError("Phase-0 exact failure freeze binding differs")
    if failure["failure_sha256"] != canonical_sha256(failure, omit="failure_sha256"):
        raise ContractError("Phase-0 failure self hash differs")
