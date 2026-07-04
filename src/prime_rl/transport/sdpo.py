from collections.abc import Iterable


def is_active_sdpo_weight(weight: object) -> bool:
    """True when a per-token SDPO weight routes the token to the SDPO component.

    This is a membership predicate, not numeric validation. Keep it permissive:
    malformed nonzero sentinels should be treated as active so the schema
    validators reject them instead of silently skipping the SDPO record.
    """
    return weight not in (None, 0, 0.0)


def has_active_sdpo_weights(weights: Iterable[object]) -> bool:
    return any(is_active_sdpo_weight(weight) for weight in weights)
