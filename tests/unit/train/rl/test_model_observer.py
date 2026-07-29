import math

import pytest
import torch

from prime_rl.configs.rl import RLConfig
from prime_rl.configs.trainer import DefaultLossConfig, ModelObserverConfig, TrainerConfig
from prime_rl.trainer.rl.loss import compute_loss, setup_rl_loss_fn
from prime_rl.trainer.rl.model_observer import ModelObserverBank, RidgeEpiplexityObserver, project_model_states


def test_project_model_states_is_fixed_normalized_and_detached():
    hidden = torch.arange(48, dtype=torch.float32).reshape(2, 3, 8).requires_grad_()
    first = project_model_states(hidden, 4)
    second = project_model_states(hidden, 4)
    assert torch.equal(first, second)
    assert torch.allclose(first.norm(dim=-1), torch.ones(6))
    assert not first.requires_grad


def test_ridge_observer_matches_closed_form_reference():
    config = ModelObserverConfig(feature_dim=2, ridge_lambda=0.5, code_resolution=1.2)
    observer = RidgeEpiplexityObserver(config, correction_dim=2)
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    corrections = torch.tensor([[1.0, -1.0], [2.0, 0.5], [3.0, -0.5]])
    observer.update(features, corrections)
    gram = 0.5 * torch.eye(2, dtype=torch.float64) + features.double().T @ features.double()
    expected_readout = torch.linalg.solve(gram, features.double().T @ corrections.double())
    spectrum = torch.eye(2, dtype=torch.float64) + 1.2 * expected_readout.T @ expected_readout
    expected_score = 0.5 * torch.linalg.slogdet(spectrum).logabsdet.item() / math.log(2.0)
    assert torch.allclose(observer.readout, expected_readout)
    assert observer.score_bits.item() == pytest.approx(expected_score)


def test_ridge_observer_assigns_more_novelty_to_learnable_structure_than_noise():
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(400, 3, generator=generator)
    features = torch.nn.functional.normalize(features, dim=-1)
    structured = torch.stack(
        [2.0 * features[:, 0] - features[:, 1], features[:, 1] + 0.5 * features[:, 2]],
        dim=-1,
    )
    noise = torch.randn(400, 2, generator=generator)
    config = ModelObserverConfig(feature_dim=3, ridge_lambda=0.3)
    structured_observer = RidgeEpiplexityObserver(config, correction_dim=2)
    noise_observer = RidgeEpiplexityObserver(config, correction_dim=2)
    structured_observer.update(features, structured)
    noise_observer.update(features, noise)
    assert structured_observer.score_bits > 5.0 * noise_observer.score_bits


def test_model_observer_bank_scores_correction_and_round_trips(tmp_path):
    config = ModelObserverConfig(feature_dim=2, ridge_lambda=0.3, advantage_clip=3.0)
    bank = ModelObserverBank(config)
    hidden = torch.tensor([[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]]])
    corrections = torch.tensor([[[0.8, -0.4, 0.1], [0.5, -0.2, -0.1]]])
    weights = torch.tensor([[0.1, 0.1]])
    advantages, metrics = bank.score_and_accumulate(hidden, corrections, weights, ["env", "env"])
    assert advantages.shape == weights.shape
    assert torch.isfinite(advantages).all()
    assert metrics["model_observer/raw_novelty"].numel() == 2
    assert metrics["model_observer/correction_norm"].numel() == 2
    assert metrics["model_observer/positive"].numel() == 2
    assert metrics["model_observer/clipped"].numel() == 2
    assert bank.observers["env"].observation_count == 0
    with pytest.raises(RuntimeError, match="before committing"):
        bank.save(tmp_path / "premature.pt")
    bank.commit()
    state_metrics = bank.state_metrics()
    assert state_metrics["model_observer/observation_count"].item() == 2
    assert state_metrics["model_observer/score_bits"].item() == pytest.approx(
        bank.observers["env"].score_bits.item()
    )

    path = tmp_path / "observer.pt"
    bank.save(path)
    restored = ModelObserverBank(config)
    restored.load(path)
    assert restored.observers["env"].observation_count == 2
    assert restored.observers["env"].correction_dim == 3
    assert torch.equal(restored.observers["env"].gram, bank.observers["env"].gram)
    assert torch.equal(restored.observers["env"].cross, bank.observers["env"].cross)

    with pytest.raises(FileNotFoundError, match="observer checkpoint"):
        restored.load(tmp_path / "missing.pt")


def test_model_observer_rejects_a_changed_correction_dimension():
    config = ModelObserverConfig(feature_dim=2)
    bank = ModelObserverBank(config)
    hidden = torch.ones(1, 1, 4)
    weights = torch.ones(1, 1)
    bank.score_and_accumulate(hidden, torch.ones(1, 1, 3), weights, ["env"])

    with pytest.raises(ValueError, match="dimension changed"):
        bank.score_and_accumulate(hidden, torch.ones(1, 1, 4), weights, ["env"])


def test_model_observer_uses_one_pre_batch_state_across_microbatches():
    config = ModelObserverConfig(feature_dim=2)
    bank = ModelObserverBank(config)
    hidden = torch.tensor([[[1.0, 0.0, 0.0, 1.0]]])
    corrections = torch.tensor([[[1.0, -1.0]]])
    weights = torch.ones(1, 1)

    first, _ = bank.score_and_accumulate(hidden, corrections, weights, ["env"])
    second, _ = bank.score_and_accumulate(hidden, corrections, weights, ["env"])
    assert torch.equal(first, second)
    bank.commit()
    assert bank.observers["env"].observation_count == 2


def test_model_observer_shuffled_control_is_deterministic_and_breaks_pairing():
    hidden = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]
    )
    corrections = torch.tensor([[[1.0, 0.0], [2.0, 1.0], [3.0, -1.0], [4.0, 2.0]]])
    weights = torch.ones(1, 4)
    config = ModelObserverConfig(feature_dim=4, shuffle_corrections=True, shuffle_seed=7)

    first = ModelObserverBank(config)
    second = ModelObserverBank(config)
    first.score_and_accumulate(hidden, corrections, weights, ["env"] * 4)
    second.score_and_accumulate(hidden, corrections, weights, ["env"] * 4)

    assert torch.equal(first.pending_updates["env"][1], second.pending_updates["env"][1])
    assert not torch.equal(first.pending_updates["env"][1], corrections.squeeze(0).double())


def test_compute_loss_adds_weighted_model_observer_policy_credit():
    trainer = torch.tensor([-1.0, -1.0], requires_grad=True)
    inference = torch.tensor([-1.0, -1.0])
    mask = torch.tensor([True, True])
    zeros = torch.zeros(2)
    novelty = torch.ones(2)
    weights = torch.full((2,), 0.1)
    loss, metrics = compute_loss(
        trainer_logprobs=[trainer],
        inference_logprobs=[inference],
        ref_logprobs=None,
        advantages=[zeros],
        loss_mask=[mask],
        rl_weights=[zeros],
        ce_weights=None,
        ref_kl_weights=None,
        novelty_advantages=[novelty],
        novelty_weights=[weights],
        rl_loss_fn=setup_rl_loss_fn(DefaultLossConfig()),
        rl_scale=1,
        ce_scale=1,
        ref_kl_scale=1,
        novelty_scale=2,
    )
    assert loss.item() == pytest.approx(-0.1)
    assert "model_observer/unmasked_mismatch_kl" in metrics
    assert metrics["loss_component/model_observer"].item() == pytest.approx(-0.1)
    loss.backward()
    assert torch.allclose(trainer.grad, torch.tensor([-0.05, -0.05]))


def test_trainer_config_rejects_unsupported_observer_parallelism():
    TrainerConfig(model_observer={})
    with pytest.raises(ValueError, match="model.cp=1"):
        TrainerConfig(model={"cp": 2}, model_observer={})


@pytest.mark.parametrize(
    ("trainer", "orchestrator"),
    [
        ({}, {"algo": {"type": "sdpo", "novelty": {}}}),
        ({"model_observer": {}}, {}),
    ],
)
def test_rl_config_requires_novelty_and_observer_together(trainer, orchestrator):
    with pytest.raises(ValueError, match="must be enabled together"):
        RLConfig.model_validate({"trainer": trainer, "orchestrator": orchestrator})


def test_rl_config_accepts_joint_sdpo_model_observer():
    config = RLConfig.model_validate(
        {
            "trainer": {"model_observer": {}},
            "orchestrator": {"algo": {"type": "sdpo", "novelty": {}}},
            "ckpt": {},
        }
    )
    assert config.orchestrator.algo.type == "sdpo"


def test_rl_config_accepts_novelty_only_ablation():
    config = RLConfig.model_validate(
        {
            "trainer": {"model_observer": {}},
            "orchestrator": {"algo": {"type": "sdpo", "novelty": {"weight": 1.0}}},
            "ckpt": {},
        }
    )
    assert config.orchestrator.algo.novelty.weight == 1.0


def test_rl_config_requires_resume_capable_observer_checkpoint():
    base = {
        "trainer": {"model_observer": {}},
        "orchestrator": {"algo": {"type": "sdpo", "novelty": {}}},
    }
    with pytest.raises(ValueError, match="resume-capable"):
        RLConfig.model_validate(base)
    base["trainer"]["ckpt"] = {"weights_only": True}
    with pytest.raises(ValueError, match="resume-capable"):
        RLConfig.model_validate(base)
