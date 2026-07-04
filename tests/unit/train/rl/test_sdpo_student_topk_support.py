import pytest
import torch

from prime_rl.trainer.rl.sdpo_train_support import gather_sdpo_student_topk_logprobs, select_sdpo_student_topk_support


def test_gather_sdpo_student_topk_logprobs_matches_shifted_token_support():
    logits = torch.tensor(
        [
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0, 0.0],
                [1.0, 0.0, -1.0, -2.0, -3.0],
                [-9.0, -8.0, -7.0, -6.0, -5.0],
            ]
        ]
    )
    temperatures = torch.tensor([[1.0, 2.0, 0.5, 1.0]])
    topk_token_ids = torch.tensor([[[0, 1], [4, 2], [0, 3], [1, 4]]])

    gathered = gather_sdpo_student_topk_logprobs(logits, temperatures, topk_token_ids)

    uniform_first_position = torch.zeros(5).log_softmax(dim=-1)
    expected = torch.stack(
        [
            uniform_first_position[[0, 1]],
            (logits[0, 0] / temperatures[0, 0]).log_softmax(dim=-1)[[4, 2]],
            (logits[0, 1] / temperatures[0, 1]).log_softmax(dim=-1)[[0, 3]],
            (logits[0, 2] / temperatures[0, 2]).log_softmax(dim=-1)[[1, 4]],
        ]
    ).unsqueeze(0)
    torch.testing.assert_close(gathered, expected)


@pytest.mark.parametrize(
    ("token_ids", "message"),
    [
        (torch.zeros(1, 3, 2, dtype=torch.long), "leading shape"),
        (torch.zeros(1, 4, 0, dtype=torch.long), "non-empty top-k dimension"),
        (torch.zeros(1, 4, 2, dtype=torch.float32), "integer tensor dtype"),
        (torch.zeros(1, 4, 2, dtype=torch.complex64), "integer tensor dtype"),
        (torch.tensor([[[0, 1], [4, 2], [0, 3], [1, 5]]]), "vocabulary range"),
        (torch.tensor([[[0, 1], [4, 2], [0, 3], [1, -1]]]), "vocabulary range"),
    ],
)
def test_gather_sdpo_student_topk_logprobs_rejects_malformed_support_ids(token_ids, message):
    with pytest.raises(ValueError, match=message):
        gather_sdpo_student_topk_logprobs(
            torch.zeros(1, 4, 5),
            torch.ones(1, 4),
            token_ids,
        )


def test_gather_sdpo_student_topk_logprobs_rejects_duplicate_supported_rows_only():
    logits = torch.zeros(1, 4, 5)
    temperatures = torch.ones(1, 4)
    token_ids = torch.tensor([[[0, 0], [1, 2], [3, 3], [0, 0]]])

    gathered = gather_sdpo_student_topk_logprobs(
        logits,
        temperatures,
        token_ids,
        support_mask=torch.tensor([[False, True, False, False]]),
    )

    assert gathered.shape == (1, 4, 2)
    with pytest.raises(ValueError, match="distinct on supported token rows"):
        gather_sdpo_student_topk_logprobs(
            logits,
            temperatures,
            token_ids,
            support_mask=torch.tensor([[False, True, True, False]]),
        )


def test_gather_sdpo_student_topk_logprobs_rejects_misaligned_support_mask():
    with pytest.raises(ValueError, match="support mask shape"):
        gather_sdpo_student_topk_logprobs(
            torch.zeros(1, 4, 5),
            torch.ones(1, 4),
            torch.zeros(1, 4, 2, dtype=torch.long),
            support_mask=torch.ones(1, 3, dtype=torch.bool),
        )


@pytest.mark.parametrize(
    ("logits", "message"),
    [
        (torch.zeros(1, 4, 5, dtype=torch.long), "floating-point tensor dtype"),
        (
            torch.tensor(
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, float("nan"), 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ),
            "finite values",
        ),
    ],
)
def test_gather_sdpo_student_topk_logprobs_rejects_malformed_logits(logits, message):
    with pytest.raises(ValueError, match=message):
        gather_sdpo_student_topk_logprobs(
            logits,
            torch.ones(1, 4),
            torch.zeros(1, 4, 2, dtype=torch.long),
        )


@pytest.mark.parametrize(
    ("temperatures", "message"),
    [
        (torch.ones(1, 3), "temperatures shape"),
        (torch.tensor([[1.0, float("nan"), 1.0, 1.0]]), "finite values"),
        (torch.tensor([[1.0, 0.0, 1.0, 1.0]]), "must be positive"),
    ],
)
def test_gather_sdpo_student_topk_logprobs_rejects_malformed_temperatures(temperatures, message):
    with pytest.raises(ValueError, match=message):
        gather_sdpo_student_topk_logprobs(
            torch.zeros(1, 4, 5),
            temperatures,
            torch.zeros(1, 4, 2, dtype=torch.long),
        )


def test_select_sdpo_student_topk_support_matches_shifted_logits():
    logits = torch.tensor(
        [
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0, 0.0],
                [1.0, 0.0, -1.0, -2.0, -3.0],
                [-9.0, -8.0, -7.0, -6.0, -5.0],
            ]
        ]
    )
    temperatures = torch.tensor([[1.0, 2.0, 0.5, 1.0]])

    token_ids, log_probs = select_sdpo_student_topk_support(logits, temperatures, topk=2)

    assert token_ids.shape == (1, 4, 2)
    assert log_probs.shape == (1, 4, 2)
    assert token_ids[:, 1:].tolist() == [[[4, 3], [0, 1], [0, 1]]]
    expected_log_probs = torch.stack(
        [
            (logits[0, 0] / temperatures[0, 0]).log_softmax(dim=-1)[[4, 3]],
            (logits[0, 1] / temperatures[0, 1]).log_softmax(dim=-1)[[0, 1]],
            (logits[0, 2] / temperatures[0, 2]).log_softmax(dim=-1)[[0, 1]],
        ]
    ).unsqueeze(0)
    torch.testing.assert_close(log_probs[:, 1:], expected_log_probs)


def test_select_sdpo_student_topk_support_matches_gather_on_selected_ids():
    logits = torch.tensor(
        [
            [
                [0.0, 1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0, 0.0],
                [0.0, 2.0, 1.0, 3.0],
            ]
        ]
    )
    temperatures = torch.tensor([[1.0, 0.5, 2.0]])

    token_ids, selected_log_probs = select_sdpo_student_topk_support(logits, temperatures, topk=3)

    gathered_log_probs = gather_sdpo_student_topk_logprobs(logits, temperatures, token_ids)
    torch.testing.assert_close(selected_log_probs, gathered_log_probs)


@pytest.mark.parametrize(
    ("logits", "message"),
    [
        (torch.zeros(1, 4, 5, dtype=torch.long), "floating-point tensor dtype"),
        (
            torch.tensor(
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, float("inf"), 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ),
            "finite values",
        ),
    ],
)
def test_select_sdpo_student_topk_support_rejects_malformed_logits(logits, message):
    with pytest.raises(ValueError, match=message):
        select_sdpo_student_topk_support(
            logits,
            torch.ones(1, 4),
            topk=2,
        )


def test_select_sdpo_student_topk_support_rejects_non_positive_topk():
    with pytest.raises(ValueError, match="topk > 0"):
        select_sdpo_student_topk_support(
            torch.zeros(1, 1, 4),
            torch.ones(1, 1),
            topk=0,
        )


def test_select_sdpo_student_topk_support_rejects_non_integer_topk():
    with pytest.raises(ValueError, match="integer topk"):
        select_sdpo_student_topk_support(
            torch.zeros(1, 1, 4),
            torch.ones(1, 1),
            topk=1.5,
        )


def test_select_sdpo_student_topk_support_rejects_topk_above_vocab_size():
    with pytest.raises(ValueError, match="topk=5.*vocab size is 4"):
        select_sdpo_student_topk_support(
            torch.zeros(1, 1, 4),
            torch.ones(1, 1),
            topk=5,
        )
