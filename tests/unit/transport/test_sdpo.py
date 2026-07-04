from prime_rl.transport.sdpo import has_active_sdpo_weights, is_active_sdpo_weight


def test_sdpo_weight_membership_contract():
    assert not is_active_sdpo_weight(None)
    assert not is_active_sdpo_weight(0)
    assert not is_active_sdpo_weight(0.0)
    assert not is_active_sdpo_weight(False)

    assert is_active_sdpo_weight(0.5)
    assert is_active_sdpo_weight(1.0)
    assert is_active_sdpo_weight(True)
    assert is_active_sdpo_weight("bad-nonzero-weight")

    assert not has_active_sdpo_weights([None, 0.0, False])
    assert has_active_sdpo_weights([0.0, 0.25])
    assert has_active_sdpo_weights([0.0, "bad-nonzero-weight"])
