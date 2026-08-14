from scripts.summarize_programmatic_memory_eval import FEEDBACK_SCHEMA, report


def _trace(
    *,
    split: str,
    family: str,
    strict: float,
    contract: dict | None = None,
) -> dict:
    info = {}
    if contract is not None:
        info = {"feedback": contract["message"], "feedback_contract": contract}
    return {
        "ok": True,
        "task": {"data": {"split": split, "family": family}},
        "metrics": {
            "strict_success": strict,
            "answer_correct": strict,
            "retrieval_decision": 1.0,
        },
        "info": info,
    }


def _contract(code: str, category: str) -> dict:
    return {
        "schema_version": FEEDBACK_SCHEMA,
        "code": code,
        "category": category,
        "answer_free": True,
        "retryable": True,
        "message": f"diagnostic: {code}",
    }


def test_report_tracks_typed_failure_mass_without_counting_successes() -> None:
    traces = [
        _trace(split="familiar_heldout", family="latest_state", strict=1.0),
        _trace(
            split="familiar_heldout",
            family="latest_state",
            strict=0.0,
            contract=_contract("event_semantics_mismatch", "event_semantics"),
        ),
        _trace(
            split="semantic_ood",
            family="policy_reinstatement",
            strict=0.0,
            contract=_contract("output_contract_violation", "output_contract"),
        ),
    ]

    result = report(traces)
    overall = result["overall"]

    assert overall["count"] == 3
    assert overall["failure_count"] == 2
    assert overall["means"]["strict_success"] == 1 / 3
    assert overall["typed_failures"]["coverage"] == 1.0
    assert overall["typed_failures"]["untyped_count"] == 0
    assert overall["typed_failures"]["success_contract_count"] == 0
    assert overall["typed_failures"]["unexpected_schema_count"] == 0
    assert overall["typed_failures"]["message_mismatch_count"] == 0
    assert overall["typed_failures"]["code_mass"] == {
        "event_semantics_mismatch": 0.5,
        "output_contract_violation": 0.5,
    }
    assert result["by_split"]["familiar_heldout"]["count"] == 2
    assert result["by_family"]["policy_reinstatement"]["count"] == 1


def test_report_surfaces_untyped_and_malformed_contracts() -> None:
    malformed = _contract("tool_execution_error", "execution")
    malformed["schema_version"] = "unexpected/v1"
    malformed["answer_free"] = False
    trace = _trace(
        split="semantic_ood",
        family="policy_reinstatement",
        strict=0.0,
        contract=malformed,
    )
    trace["info"]["feedback"] = "different rendering"

    result = report(
        [
            trace,
            _trace(
                split="semantic_ood",
                family="policy_reinstatement",
                strict=0.0,
            ),
        ]
    )["overall"]["typed_failures"]

    assert result["coverage"] == 0.5
    assert result["untyped_count"] == 1
    assert result["unexpected_schema_count"] == 1
    assert result["answer_free_violation_count"] == 1
    assert result["message_mismatch_count"] == 1


def test_report_counts_contracts_on_successes_with_duplicate_failure_payloads() -> None:
    contract = _contract("event_semantics_mismatch", "event_semantics")
    failed = _trace(
        split="familiar_heldout",
        family="latest_state",
        strict=0.0,
        contract=contract,
    )
    succeeded = _trace(
        split="familiar_heldout",
        family="latest_state",
        strict=1.0,
        contract=contract,
    )

    result = report([failed, failed.copy(), succeeded])["overall"]["typed_failures"]

    assert result["count"] == 2
    assert result["success_contract_count"] == 1
