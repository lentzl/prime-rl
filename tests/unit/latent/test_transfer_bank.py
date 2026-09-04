from prime_rl.latent.transfer_bank import (
    FAMILIES,
    assign_moth_donors,
    build_transfer_bank,
)


def test_bank_is_deterministic_and_supports_multiple_queries_per_evidence() -> None:
    first = build_transfer_bank(seed=17, split="held_out", examples_per_family=3)
    second = build_transfer_bank(seed=17, split="held_out", examples_per_family=3)

    assert first.manifest_sha256() == second.manifest_sha256()
    assert {record.family for record in first.records} == set(FAMILIES)
    assert all(len(record.queries) == 3 for record in first.records)


def test_role_views_do_not_expose_counterpart_payload_or_answers() -> None:
    bank = build_transfer_bank(seed=29, split="train", examples_per_family=2)

    assert all(set(row) == {"evidence_id", "family", "parent_evidence"} for row in bank.parent_view())
    assert all(set(row) == {"evidence_id", "query_id", "family", "child_query"} for row in bank.child_view())
    assert all(set(row) == {"evidence_id", "query_id", "answer"} for row in bank.answer_key())


def test_held_out_surface_templates_are_not_training_templates() -> None:
    train = build_transfer_bank(seed=1, split="train", examples_per_family=2)
    held_out = build_transfer_bank(seed=2, split="held_out", examples_per_family=2)

    train_templates = {
        template
        for record in train.records
        for template in (record.parent_template_id, *(query.template_id for query in record.queries))
    }
    held_out_templates = {
        template
        for record in held_out.records
        for template in (record.parent_template_id, *(query.template_id for query in record.queries))
    }
    assert train_templates.isdisjoint(held_out_templates)


def test_moth_donors_are_deterministic_family_matched_derangements() -> None:
    bank = build_transfer_bank(seed=41, split="held_out", examples_per_family=4)
    records = {record.evidence_id: record for record in bank.records}

    donors = assign_moth_donors(bank)

    assert donors == assign_moth_donors(bank)
    assert all(source != donor for source, donor in donors.items())
    assert all(records[source].family == records[donor].family for source, donor in donors.items())
