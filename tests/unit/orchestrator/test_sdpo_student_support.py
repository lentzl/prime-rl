import json
from types import SimpleNamespace

import pytest

from prime_rl.orchestrator.sdpo_student_support import (
    hydrate_student_support_from_records,
    load_student_support_records,
)


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _write_raw_jsonl(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")


def _record(**overrides):
    record = {
        "schema_version": 2,
        "sample_id": "sample-a",
        "env_name": "sdpo_env",
        "token_ids": [10, 11, 12],
        "position_ids": [0, 1, 2],
        "loss_mask": [False, True, True],
        "temperatures": [1.0, 1.0, 1.0],
        "sdpo_weights": [0.0, 1.0, 1.0],
        "sdpo_student_topk_token_ids": [[0, 0], [111, 112], [211, 212]],
        "sdpo_student_topk_logprobs": [[0.0, 0.0], [-0.75, -1.25], [-0.625, -1.5]],
    }
    record.update(overrides)
    return record


def _sample(**overrides):
    sample = SimpleNamespace(
        token_ids=[10, 11, 12],
        position_ids=[0, 1, 2],
        mask=[False, True, True],
        logprobs=[0.0, -0.1, -0.2],
        temperatures=[1.0, 1.0, 1.0],
        env_name="sdpo_env",
        sample_id="sample-a",
        sdpo_weights=[0.0, 1.0, 1.0],
    )
    for key, value in overrides.items():
        setattr(sample, key, value)
    return sample


def test_load_and_hydrate_student_support_records(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(sdpo_weights=[0.0, 0.0, 0.0]),
            _record(),
        ],
    )
    records = load_student_support_records(tmp_path)
    sample = _sample()
    sample.sdpo_topk_logprobs = [[9.0, 9.0], [9.0, 9.0], [9.0, 9.0]]

    hydrated = hydrate_student_support_from_records([sample], records)

    assert hydrated == 2
    assert sample.sdpo_topk_token_ids == [[0, 0], [111, 112], [211, 212]]
    assert sample.sdpo_topk_logprobs is None


def test_hydrate_student_support_rejects_non_list_sample_sdpo_weights(tmp_path):
    export_file = tmp_path / "run_default" / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record()])
    records = load_student_support_records(tmp_path)

    with pytest.raises(ValueError, match="sample sdpo_weights must be a list"):
        hydrate_student_support_from_records([_sample(sdpo_weights=1.0)], records)


@pytest.mark.parametrize("record", [[], None, "not-object"])
def test_load_student_support_rejects_non_object_jsonl_records(tmp_path, record):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [record])

    with pytest.raises(ValueError, match="expected JSON object record"):
        load_student_support_records(tmp_path)


def test_load_student_support_rejects_duplicate_json_object_keys(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    raw_record = json.dumps(_record())
    duplicate_key_record = raw_record.replace('"sample_id": "sample-a"', '"sample_id": "sample-a", "sample_id": "b"', 1)
    _write_raw_jsonl(export_file, [duplicate_key_record])

    with pytest.raises(ValueError, match="duplicate JSON object key: sample_id"):
        load_student_support_records(tmp_path)


def test_load_student_support_can_require_preflight_only_records(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(preflight_only=True)])

    records = load_student_support_records(tmp_path, require_preflight_only=True)

    assert len(records) == 1
    assert records[0].preflight_only is True


def test_load_student_support_rejects_dense_unweighted_rows_when_preflight_required(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                preflight_only=True,
                sdpo_student_topk_token_ids=[[101, 102], [111, 112], [211, 212]],
                sdpo_student_topk_logprobs=[[-0.7, -1.1], [-0.75, -1.25], [-0.625, -1.5]],
            )
        ],
    )

    with pytest.raises(ValueError, match="unweighted student top-k row 0 must be an all-zero placeholder"):
        load_student_support_records(tmp_path, require_preflight_only=True)


@pytest.mark.parametrize(
    ("field_prefix", "expected_error"),
    [
        ("sdpo_student", "inactive SDPO record must not carry sdpo_student top-k support rows"),
        ("sdpo", "inactive SDPO record must not carry sdpo top-k support rows"),
    ],
)
def test_load_student_support_rejects_inactive_support_rows_when_preflight_required(
    tmp_path, field_prefix, expected_error
):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    inactive = _record(sdpo_weights=[0.0, 0.0, 0.0])
    for prefix in ("sdpo_student", "sdpo"):
        inactive.pop(f"{prefix}_topk_token_ids", None)
        inactive.pop(f"{prefix}_topk_logprobs", None)
    inactive[f"{field_prefix}_topk_token_ids"] = [[0, 0], [0, 0], [0, 0]]
    inactive[f"{field_prefix}_topk_logprobs"] = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    _write_jsonl(export_file, [inactive, _record(preflight_only=True)])

    with pytest.raises(ValueError, match=expected_error):
        load_student_support_records(tmp_path, require_preflight_only=True)


def test_load_student_support_rejects_empty_student_topk_rows(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                preflight_only=True,
                sdpo_student_topk_token_ids=[[], [111, 112], [211, 212]],
                sdpo_student_topk_logprobs=[[], [-0.75, -1.25], [-0.625, -1.5]],
            )
        ],
    )

    with pytest.raises(ValueError, match="sdpo_student_topk_token_ids\\[0\\] must be non-empty"):
        load_student_support_records(tmp_path, require_preflight_only=True)


def test_load_student_support_rejects_ragged_unweighted_student_placeholders(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                preflight_only=True,
                sdpo_student_topk_token_ids=[[0, 0], [111, 112], [211, 212]],
                sdpo_student_topk_logprobs=[[0.0], [-0.75, -1.25], [-0.625, -1.5]],
            )
        ],
    )

    with pytest.raises(ValueError, match="sdpo_student_topk row 0 width mismatch"):
        load_student_support_records(tmp_path, require_preflight_only=True)


def test_load_student_support_rejects_ragged_weighted_student_rows(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_student_topk_token_ids=[[0, 0], [111, 112], [211, 212]],
                sdpo_student_topk_logprobs=[[0.0, 0.0], [-0.75], [-0.625, -1.5]],
            )
        ],
    )

    with pytest.raises(ValueError, match="sdpo_student_topk row 1 width mismatch"):
        load_student_support_records(tmp_path)


def test_load_student_support_rejects_one_sided_null_student_rows(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_student_topk_token_ids=[[0, 0], None, [211, 212]],
                sdpo_student_topk_logprobs=[[0.0, 0.0], [-0.75, -1.25], [-0.625, -1.5]],
            )
        ],
    )

    with pytest.raises(ValueError, match="sdpo_student_topk row 1 ids/logprobs must both be lists or both be null"):
        load_student_support_records(tmp_path)


def test_load_student_support_allows_dense_unweighted_rows_without_strict_preflight(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                preflight_only=True,
                sdpo_student_topk_token_ids=[[101, 102], [111, 112], [211, 212]],
                sdpo_student_topk_logprobs=[[-0.7, -1.1], [-0.75, -1.25], [-0.625, -1.5]],
            )
        ],
    )

    records = load_student_support_records(tmp_path)

    assert len(records) == 1
    assert records[0].student_topk_token_ids[0] == [101, 102]


def test_load_student_support_rejects_transported_teacher_rows_when_preflight_required(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                preflight_only=True,
                sdpo_topk_token_ids=[[0, 0], [101, 102], [201, 202]],
                sdpo_topk_logprobs=[[0.0, 0.0], [-0.5, -1.5], [-0.6, -1.6]],
            )
        ],
    )

    with pytest.raises(ValueError, match="preflight transported teacher top-k row 1"):
        load_student_support_records(tmp_path, require_preflight_only=True)


def test_load_student_support_rejects_unweighted_transported_teacher_rows_when_preflight_required(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                preflight_only=True,
                sdpo_topk_token_ids=[[101, 102], [0, 0], [0, 0]],
                sdpo_topk_logprobs=[[-0.5, -1.5], [0.0, 0.0], [0.0, 0.0]],
            )
        ],
    )

    with pytest.raises(ValueError, match="preflight transported teacher top-k row 0"):
        load_student_support_records(tmp_path, require_preflight_only=True)


def test_load_student_support_allows_transported_teacher_placeholders_when_preflight_required(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                preflight_only=True,
                sdpo_topk_token_ids=[[0, 0], [0, 0], [0, 0]],
                sdpo_topk_logprobs=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            )
        ],
    )

    records = load_student_support_records(tmp_path, require_preflight_only=True)

    assert len(records) == 1


def test_load_student_support_rejects_ragged_transported_teacher_placeholders(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                preflight_only=True,
                sdpo_topk_token_ids=[[0, 0], [0, 0], [0, 0]],
                sdpo_topk_logprobs=[[0.0], [0.0, 0.0], [0.0, 0.0]],
            )
        ],
    )

    with pytest.raises(ValueError, match="preflight transported teacher top-k row 0 width mismatch"):
        load_student_support_records(tmp_path, require_preflight_only=True)


def test_load_student_support_rejects_boolean_transported_teacher_placeholder_logprobs(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                preflight_only=True,
                sdpo_topk_token_ids=[[0, 0], [0, 0], [0, 0]],
                sdpo_topk_logprobs=[[False, 0.0], [0.0, 0.0], [0.0, 0.0]],
            )
        ],
    )

    with pytest.raises(ValueError, match="preflight transported teacher top-k row 0"):
        load_student_support_records(tmp_path, require_preflight_only=True)


@pytest.mark.parametrize("preflight_only", [False, None])
def test_load_student_support_rejects_non_preflight_records_when_required(tmp_path, preflight_only):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    record = _record()
    if preflight_only is None:
        record.pop("preflight_only", None)
    else:
        record["preflight_only"] = preflight_only
    _write_jsonl(export_file, [record])

    with pytest.raises(ValueError, match="student-support records must have preflight_only=true"):
        load_student_support_records(tmp_path, require_preflight_only=True)


def test_load_student_support_rejects_missing_sample_id_when_preflight_required(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    record = _record(preflight_only=True)
    record.pop("sample_id", None)
    _write_jsonl(export_file, [record])

    with pytest.raises(ValueError, match="preflight records must carry a non-empty sample_id"):
        load_student_support_records(tmp_path, require_preflight_only=True)


@pytest.mark.parametrize("env_name", [None, "", "   "])
def test_load_student_support_rejects_missing_env_name_when_preflight_required(tmp_path, env_name):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    record = _record(preflight_only=True, env_name=env_name)
    if env_name is None:
        record.pop("env_name", None)
    _write_jsonl(export_file, [record])

    with pytest.raises(ValueError, match="preflight records must carry a non-empty env_name"):
        load_student_support_records(tmp_path, require_preflight_only=True)


@pytest.mark.parametrize("sample_id", ["", "   "])
def test_load_student_support_rejects_blank_sample_id_when_preflight_required(tmp_path, sample_id):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(preflight_only=True, sample_id=sample_id)])

    with pytest.raises(ValueError, match="sample_id must be a non-empty string"):
        load_student_support_records(tmp_path, require_preflight_only=True)


def test_load_student_support_rejects_non_boolean_preflight_only(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(preflight_only="yes")])

    with pytest.raises(ValueError, match="preflight_only must be a boolean"):
        load_student_support_records(tmp_path)


def test_hydrate_student_support_is_order_preserving(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    second = _record(
        sample_id="sample-b",
        token_ids=[20, 21],
        position_ids=[0, 1],
        loss_mask=[True, True],
        temperatures=[1.0, 1.0],
        sdpo_weights=[1.0, 1.0],
        sdpo_student_topk_token_ids=[[301, 302], [401, 402]],
        sdpo_student_topk_logprobs=[[-0.5, -2.0], [-0.6, -1.5]],
    )
    _write_jsonl(export_file, [_record(), second])
    records = load_student_support_records(tmp_path / "token_exports")
    first_sample = _sample()
    second_sample = _sample(
        token_ids=[20, 21],
        position_ids=[0, 1],
        mask=[True, True],
        logprobs=[-0.3, -0.4],
        temperatures=[1.0, 1.0],
        sample_id="sample-b",
        sdpo_weights=[1.0, 1.0],
    )

    hydrated = hydrate_student_support_from_records([first_sample, second_sample], records)

    assert hydrated == 4
    assert first_sample.sdpo_topk_token_ids == [[0, 0], [111, 112], [211, 212]]
    assert second_sample.sdpo_topk_token_ids == [[301, 302], [401, 402]]


def test_hydrate_student_support_matches_by_sample_id_when_records_are_out_of_order(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    first = _record(sample_id="sample-a")
    second = _record(
        sample_id="sample-b",
        token_ids=[20, 21],
        position_ids=[0, 1],
        loss_mask=[True, True],
        temperatures=[1.0, 1.0],
        sdpo_weights=[1.0, 1.0],
        sdpo_student_topk_token_ids=[[301, 302], [401, 402]],
        sdpo_student_topk_logprobs=[[-0.5, -2.0], [-0.6, -1.5]],
    )
    _write_jsonl(export_file, [second, first])
    records = load_student_support_records(tmp_path)
    first_sample = _sample(sample_id="sample-a")
    second_sample = _sample(
        sample_id="sample-b",
        token_ids=[20, 21],
        position_ids=[0, 1],
        mask=[True, True],
        logprobs=[-0.3, -0.4],
        temperatures=[1.0, 1.0],
        sdpo_weights=[1.0, 1.0],
    )

    hydrated = hydrate_student_support_from_records([first_sample, second_sample], records)

    assert hydrated == 4
    assert first_sample.sdpo_topk_token_ids == [[0, 0], [111, 112], [211, 212]]
    assert second_sample.sdpo_topk_token_ids == [[301, 302], [401, 402]]


def test_hydrate_student_support_rejects_token_mismatch(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(token_ids=[10, 99, 12])])
    records = load_student_support_records(tmp_path)

    with pytest.raises(ValueError, match="record token_ids do not match sample"):
        hydrate_student_support_from_records([_sample()], records)


def test_hydrate_student_support_rejects_env_name_mismatch(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(env_name="other_env")])
    records = load_student_support_records(tmp_path)

    with pytest.raises(ValueError, match="record env_name='other_env' mismatch"):
        hydrate_student_support_from_records([_sample()], records)


def test_hydrate_student_support_rejects_position_id_mismatch(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(position_ids=[0, 4, 5])])
    records = load_student_support_records(tmp_path)

    with pytest.raises(ValueError, match="record position_ids do not match sample"):
        hydrate_student_support_from_records([_sample()], records)


def test_hydrate_student_support_uses_default_position_ids_when_sample_has_none(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(position_ids=[0, 1, 2])])
    records = load_student_support_records(tmp_path)
    sample = _sample()
    delattr(sample, "position_ids")

    hydrated = hydrate_student_support_from_records([sample], records)

    assert hydrated == 2
    assert sample.sdpo_topk_token_ids == [[0, 0], [111, 112], [211, 212]]


def test_load_student_support_rejects_non_integer_token_ids(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(token_ids=[10, True, 12])])

    with pytest.raises(ValueError, match="token_ids must contain integer token ids"):
        load_student_support_records(tmp_path)


def test_load_student_support_rejects_negative_token_ids(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(token_ids=[10, -11, 12])])

    with pytest.raises(ValueError, match="token_ids must contain non-negative token ids"):
        load_student_support_records(tmp_path)


def test_load_student_support_rejects_missing_position_ids(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    record = _record()
    del record["position_ids"]
    _write_jsonl(export_file, [record])

    with pytest.raises(ValueError, match="position_ids must be a list"):
        load_student_support_records(tmp_path)


def test_load_student_support_rejects_misaligned_position_ids(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(position_ids=[0, 1])])

    with pytest.raises(ValueError, match="position_ids length 2 != token_ids length 3"):
        load_student_support_records(tmp_path)


def test_load_student_support_rejects_non_integer_position_ids(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(position_ids=[0, True, 2])])

    with pytest.raises(ValueError, match="position_ids must contain integer token ids"):
        load_student_support_records(tmp_path)


def test_load_student_support_rejects_non_boolean_loss_mask(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(loss_mask=[False, 1, True])])

    with pytest.raises(ValueError, match="loss_mask must contain booleans"):
        load_student_support_records(tmp_path)


def test_load_student_support_rejects_non_numeric_sdpo_weights(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(sdpo_weights=[0.0, True, 1.0])])

    with pytest.raises(ValueError, match="sdpo_weights must contain finite numeric values"):
        load_student_support_records(tmp_path)


def test_load_student_support_rejects_boolean_inactive_sdpo_weights(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(sdpo_weights=[False, False, False])])

    with pytest.raises(ValueError, match="sdpo_weights must contain finite numeric values at token 0"):
        load_student_support_records(tmp_path)


def test_load_student_support_rejects_non_list_sdpo_weights(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(sdpo_weights=1.0)])

    with pytest.raises(ValueError, match="sdpo_weights must be a list"):
        load_student_support_records(tmp_path)


def test_load_student_support_rejects_negative_sdpo_weights(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(sdpo_weights=[0.0, -0.5, 1.0])])

    with pytest.raises(ValueError, match=r"sdpo_weights\[1\] must be non-negative"):
        load_student_support_records(tmp_path)


def test_load_student_support_rejects_non_string_sample_id(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(sample_id=123)])

    with pytest.raises(ValueError, match="sample_id must be a string when present"):
        load_student_support_records(tmp_path)


def test_load_student_support_rejects_non_string_env_name(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(env_name=True)])

    with pytest.raises(ValueError, match="env_name must be a string when present"):
        load_student_support_records(tmp_path)


def test_hydrate_student_support_rejects_temperature_mismatch(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(temperatures=[1.0, 0.7, 1.0])])
    records = load_student_support_records(tmp_path)

    with pytest.raises(ValueError, match="temperature mismatch at token 1"):
        hydrate_student_support_from_records([_sample()], records)


def test_hydrate_student_support_accepts_legacy_record_without_temperatures(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    record = _record()
    del record["temperatures"]
    _write_jsonl(export_file, [record])
    records = load_student_support_records(tmp_path)
    sample = _sample()

    hydrated = hydrate_student_support_from_records([sample], records)

    assert hydrated == 2
    assert sample.sdpo_topk_token_ids == [[0, 0], [111, 112], [211, 212]]


def test_hydrate_student_support_accepts_legacy_order_without_sample_ids(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    record = _record()
    del record["sample_id"]
    _write_jsonl(export_file, [record])
    records = load_student_support_records(tmp_path)
    sample = _sample(sample_id=None)

    hydrated = hydrate_student_support_from_records([sample], records)

    assert hydrated == 2
    assert sample.sdpo_topk_token_ids == [[0, 0], [111, 112], [211, 212]]


def test_hydrate_student_support_strict_mode_rejects_sample_without_id(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record()])
    records = load_student_support_records(tmp_path)

    with pytest.raises(ValueError, match="requires sample_id on every SDPO sample"):
        hydrate_student_support_from_records([_sample(sample_id=None)], records, require_sample_ids=True)


def test_hydrate_student_support_strict_mode_rejects_sample_with_blank_id(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record()])
    records = load_student_support_records(tmp_path)

    with pytest.raises(ValueError, match="requires sample_id on every SDPO sample"):
        hydrate_student_support_from_records([_sample(sample_id="   ")], records, require_sample_ids=True)


@pytest.mark.parametrize("env_name", [None, "", "   "])
def test_hydrate_student_support_strict_mode_rejects_sample_without_env_name(tmp_path, env_name):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record()])
    records = load_student_support_records(tmp_path)

    with pytest.raises(ValueError, match="requires env_name on every SDPO sample"):
        hydrate_student_support_from_records([_sample(env_name=env_name)], records, require_sample_ids=True)


def test_hydrate_student_support_strict_mode_rejects_record_without_id(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    record = _record()
    del record["sample_id"]
    _write_jsonl(export_file, [record])
    records = load_student_support_records(tmp_path)

    with pytest.raises(ValueError, match="requires sample_id on every export record"):
        hydrate_student_support_from_records([_sample()], records, require_sample_ids=True)


@pytest.mark.parametrize("env_name", [None, "", "   "])
def test_hydrate_student_support_strict_mode_rejects_record_without_env_name(tmp_path, env_name):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    record = _record(env_name=env_name)
    if env_name is None:
        record.pop("env_name", None)
    _write_jsonl(export_file, [record])
    records = load_student_support_records(tmp_path)

    with pytest.raises(ValueError, match="requires env_name on every export record"):
        hydrate_student_support_from_records([_sample()], records, require_sample_ids=True)


def test_hydrate_student_support_rejects_weight_membership_mismatch(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(sdpo_weights=[0.0, 1.0, 0.0])])
    records = load_student_support_records(tmp_path)

    with pytest.raises(ValueError, match="sdpo weight membership mismatch at token 2"):
        hydrate_student_support_from_records([_sample()], records)


def test_hydrate_student_support_rejects_weight_value_mismatch(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(sdpo_weights=[0.0, 0.5, 1.0])])
    records = load_student_support_records(tmp_path)

    with pytest.raises(ValueError, match="sdpo weight value mismatch at token 1"):
        hydrate_student_support_from_records([_sample(sdpo_weights=[0.0, 1.0, 1.0])], records)


@pytest.mark.parametrize(
    ("sample_weights", "message"),
    [
        ([0.0, True, 1.0], "finite numeric values at token 1"),
        ([0.0, float("nan"), 1.0], "finite numeric values at token 1"),
        ([0.0, -0.5, 1.0], "non-negative at token 1"),
    ],
)
def test_hydrate_student_support_rejects_malformed_sample_weights(tmp_path, sample_weights, message):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record()])
    records = load_student_support_records(tmp_path)

    with pytest.raises(ValueError, match=message):
        hydrate_student_support_from_records([_sample(sdpo_weights=sample_weights)], records)


def test_hydrate_student_support_rejects_boolean_inactive_sample_weights(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(sdpo_weights=[0.0, 0.0, 0.0])])
    records = load_student_support_records(tmp_path)

    with pytest.raises(ValueError, match="sample sdpo_weights must contain finite numeric values at token 0"):
        hydrate_student_support_from_records([_sample(sdpo_weights=[False, False, False])], records)


def test_hydrate_student_support_rejects_topk_width_mismatch(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record()])
    records = load_student_support_records(tmp_path)

    with pytest.raises(ValueError, match="student top-k width 2 != expected 3"):
        hydrate_student_support_from_records([_sample()], records, expected_topk=3)


def test_load_student_support_rejects_missing_weighted_student_topk_ids(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_student_topk_token_ids=[[0, 0], None, [211, 212]],
            )
        ],
    )

    with pytest.raises(ValueError, match="sdpo_student_topk row 1 ids/logprobs must both be lists or both be null"):
        load_student_support_records(tmp_path)


def test_load_student_support_rejects_missing_weighted_student_topk_logprobs(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_student_topk_logprobs=[[0.0, 0.0], None, [-0.625, -1.5]],
            )
        ],
    )

    with pytest.raises(ValueError, match="sdpo_student_topk row 1 ids/logprobs must both be lists or both be null"):
        load_student_support_records(tmp_path)


def test_load_student_support_rejects_non_integer_student_topk_ids(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_student_topk_token_ids=[[0, 0], [111, "112"], [211, 212]],
            )
        ],
    )

    with pytest.raises(ValueError, match="sdpo_student_topk_token_ids\\[1\\] must contain integer token ids"):
        load_student_support_records(tmp_path)


def test_load_student_support_rejects_negative_student_topk_ids(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_student_topk_token_ids=[[0, 0], [111, -112], [211, 212]],
            )
        ],
    )

    with pytest.raises(ValueError, match="sdpo_student_topk_token_ids\\[1\\] must contain non-negative token ids"):
        load_student_support_records(tmp_path)


def test_hydrate_student_support_rejects_duplicate_weighted_student_topk_ids(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_student_topk_token_ids=[[0, 0], [111, 111], [211, 212]],
            )
        ],
    )
    records = load_student_support_records(tmp_path)

    with pytest.raises(ValueError, match="student top-k ids contain duplicate token ids at token 1"):
        hydrate_student_support_from_records([_sample()], records)


def test_hydrate_student_support_rejects_weighted_placeholder_logprobs(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_student_topk_token_ids=[[0, 0], [0, 0], [211, 212]],
                sdpo_student_topk_logprobs=[[0.0, 0.0], [0.0, 0.0], [-0.625, -1.5]],
            )
        ],
    )
    records = load_student_support_records(tmp_path)

    with pytest.raises(ValueError, match="unfilled placeholder row at token 1"):
        hydrate_student_support_from_records([_sample()], records)


def test_hydrate_student_support_accepts_token_id_zero_with_real_logprob(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_student_topk_token_ids=[[0], [0], [211]],
                sdpo_student_topk_logprobs=[[0.0], [-0.75], [-0.625]],
            )
        ],
    )
    records = load_student_support_records(tmp_path)

    sample = _sample()
    hydrated = hydrate_student_support_from_records([sample], records, expected_topk=1)

    assert hydrated == 2
    assert sample.sdpo_topk_token_ids == [[0], [0], [211]]


def test_load_student_support_rejects_nonfinite_logprobs(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_student_topk_logprobs=[[0.0, 0.0], [float("nan"), -1.25], [-0.625, -1.5]],
            )
        ],
    )
    with pytest.raises(ValueError, match="sdpo_student_topk_logprobs\\[1\\] must contain finite numeric values"):
        load_student_support_records(tmp_path)


def test_load_student_support_rejects_boolean_student_topk_logprobs(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_student_topk_logprobs=[[0.0, 0.0], [False, 0.0], [-0.625, -1.5]],
            )
        ],
    )
    with pytest.raises(ValueError, match="sdpo_student_topk_logprobs\\[1\\] must contain finite numeric values"):
        load_student_support_records(tmp_path)


def test_load_student_support_rejects_integer_weighted_logprobs(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_student_topk_logprobs=[[0.0, 0.0], [-2, -3], [-0.625, -1.5]],
            )
        ],
    )

    with pytest.raises(
        ValueError, match="sdpo_student_topk_logprobs\\[1\\] must contain floating-point logprob values"
    ):
        load_student_support_records(tmp_path)


def test_hydrate_student_support_rejects_logprob_mass_above_one(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(
        export_file,
        [
            _record(
                sdpo_student_topk_logprobs=[[0.0, 0.0], [-0.1, -2.0], [-0.625, -1.5]],
            )
        ],
    )
    records = load_student_support_records(tmp_path)

    with pytest.raises(ValueError, match="probability mass exceeds 1 at token 1"):
        hydrate_student_support_from_records([_sample()], records)


def test_load_student_support_rejects_sdpo_weight_outside_loss_mask(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(sdpo_weights=[1.0, 1.0, 1.0])])

    with pytest.raises(ValueError, match="sdpo weight at token 0 is nonzero outside loss_mask"):
        load_student_support_records(tmp_path)


def test_hydrate_student_support_rejects_extra_records(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(), _record(sample_id="extra-sample")])
    records = load_student_support_records(tmp_path)

    with pytest.raises(ValueError, match="extra sample_id"):
        hydrate_student_support_from_records([_sample()], records)


def test_hydrate_student_support_rejects_missing_sample_id_record(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(sample_id="sample-a")])
    records = load_student_support_records(tmp_path)

    with pytest.raises(ValueError, match="missing student-support export record for sample_id 'sample-b'"):
        hydrate_student_support_from_records([_sample(sample_id="sample-a"), _sample(sample_id="sample-b")], records)


def test_hydrate_student_support_rejects_duplicate_export_sample_ids(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record(sample_id="sample-a"), _record(sample_id="sample-a")])
    records = load_student_support_records(tmp_path)

    with pytest.raises(ValueError, match="duplicate student-support export sample_id 'sample-a'"):
        hydrate_student_support_from_records([_sample(sample_id="sample-a")], records)


def test_hydrate_student_support_rejects_duplicate_sample_ids(tmp_path):
    export_file = tmp_path / "token_exports" / "step_3" / "rank_0.jsonl"
    _write_jsonl(export_file, [_record()])
    records = load_student_support_records(tmp_path)

    with pytest.raises(ValueError, match="duplicate SDPO sample_id 'sample-a'"):
        hydrate_student_support_from_records([_sample(), _sample()], records)
