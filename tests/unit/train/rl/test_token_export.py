import json

import pytest
import torch

from prime_rl.configs.trainer import DefaultLossConfig
from prime_rl.trainer.rl.token_export import TokenExporter, _sparse_sdpo_support
from prime_rl.transport import SDPOTeacherSpan


def _micro_batch() -> dict[str, object]:
    return {
        "input_ids": torch.tensor([[10, 11, 12, 20, 21]]),
        "position_ids": torch.tensor([[0, 1, 2, 0, 1]]),
        "loss_mask": torch.tensor([[False, True, False, True, False]]),
        "advantages": torch.zeros(1, 5),
        "inference_logprobs": torch.full((1, 5), -1.0),
        "rl_weights": torch.zeros(1, 5),
        "sdpo_weights": torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0]]),
        "sdpo_teacher_spans": [
            SDPOTeacherSpan(prefix_ids=[100], completion_ids=[11], student_positions=[1], target_offsets=[0]),
            SDPOTeacherSpan(prefix_ids=[200], completion_ids=[20], student_positions=[3], target_offsets=[0]),
        ],
        "env_names": ["audit"] * 5,
    }


def _support_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_ids = torch.tensor(
        [[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]],
    )
    student = torch.log_softmax(torch.arange(10, dtype=torch.float32).reshape(1, 5, 2), dim=-1)
    teacher = torch.log_softmax(torch.arange(10, 0, -1, dtype=torch.float32).reshape(1, 5, 2), dim=-1)
    return token_ids, student, teacher


def test_sparse_sdpo_support_keeps_only_active_tokens():
    token_ids, student, teacher = _support_tensors()

    support = _sparse_sdpo_support(
        _micro_batch(),
        sdpo_topk_token_ids=token_ids,
        student_topk_logprobs=student,
        teacher_topk_logprobs=teacher,
        teacher_support_token_ids=token_ids.flip(-1),
        student_teacher_support_logprobs=student.flip(-1),
        teacher_support_logprobs=teacher.flip(-1),
    )

    assert support is not None
    assert [entry["position"] for entry in support] == [1, 3]
    assert [entry["student_support"]["token_ids"] for entry in support] == [[3, 4], [7, 8]]
    assert [entry["teacher_support"]["token_ids"] for entry in support] == [[4, 3], [8, 7]]
    assert all(len(entry["student_support"]["student_logprobs"]) == 2 for entry in support)
    assert all(len(entry["teacher_support"]["teacher_logprobs"]) == 2 for entry in support)


def test_sparse_sdpo_support_allows_collective_only_rank_without_local_sdpo():
    micro_batch = _micro_batch()
    micro_batch.pop("sdpo_weights")
    micro_batch["sdpo_teacher_spans"] = []
    token_ids, student, teacher = _support_tensors()

    support = _sparse_sdpo_support(
        micro_batch,
        sdpo_topk_token_ids=token_ids,
        student_topk_logprobs=student,
        teacher_topk_logprobs=teacher,
        teacher_support_token_ids=token_ids,
        student_teacher_support_logprobs=student,
        teacher_support_logprobs=teacher,
    )

    assert support == []


def test_sparse_sdpo_support_rejects_partial_or_nonfinite_distributions():
    token_ids, student, teacher = _support_tensors()

    with pytest.raises(ValueError, match="requires token ids and both"):
        _sparse_sdpo_support(
            _micro_batch(),
            sdpo_topk_token_ids=token_ids,
            student_topk_logprobs=student,
            teacher_topk_logprobs=None,
            teacher_support_token_ids=token_ids,
            student_teacher_support_logprobs=student,
            teacher_support_logprobs=teacher,
        )

    teacher = teacher.clone()
    teacher[0, 1, 0] = torch.inf
    with pytest.raises(ValueError, match="must be finite"):
        _sparse_sdpo_support(
            _micro_batch(),
            sdpo_topk_token_ids=token_ids,
            student_topk_logprobs=student,
            teacher_topk_logprobs=teacher,
            teacher_support_token_ids=token_ids,
            student_teacher_support_logprobs=student,
            teacher_support_logprobs=teacher,
        )


def test_token_export_rebases_sparse_support_inside_packed_sequences(tmp_path):
    micro_batch = _micro_batch()
    token_ids, student, teacher = _support_tensors()
    exporter = TokenExporter(tmp_path, rank=0)

    exporter.export(
        step=1,
        micro_step=0,
        micro_batch=micro_batch,
        model_output={
            "logprobs": torch.full((1, 5), -0.75),
            "entropy": torch.full((1, 5), 0.5),
        },
        sequence_lengths=[3, 2],
        loss_config=DefaultLossConfig(),
        sdpo_topk_token_ids=token_ids,
        student_topk_logprobs=student,
        teacher_topk_logprobs=teacher,
        teacher_support_token_ids=token_ids.flip(-1),
        student_teacher_support_logprobs=student.flip(-1),
        teacher_support_logprobs=teacher.flip(-1),
    )
    exporter.mark_stable()

    path = tmp_path / "token_exports" / "step_1" / "rank_0.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["sdpo_support"][0]["position"] == 1
    assert records[0]["sdpo_support"][0]["student_support"]["token_ids"] == [3, 4]
    assert records[0]["sdpo_teacher_replays"][0]["prefix_ids"] == [100]
    assert records[0]["sdpo_teacher_replays"][0]["student_positions"] == [1]
    assert records[1]["sdpo_support"][0]["position"] == 0
    assert records[1]["sdpo_support"][0]["teacher_support"]["token_ids"] == [8, 7]
    assert records[1]["sdpo_teacher_replays"][0]["prefix_ids"] == [200]
    assert records[1]["sdpo_teacher_replays"][0]["student_positions"] == [0]
    assert (tmp_path / "token_exports" / "step_1" / "STABLE").is_file()
