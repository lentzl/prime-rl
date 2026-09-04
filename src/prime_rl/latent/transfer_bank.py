from __future__ import annotations

import hashlib
import json
import random
import string
from dataclasses import asdict, dataclass
from typing import Callable, Literal

BankSplit = Literal["train", "validation", "held_out"]
FAMILIES = ("keyed_numeric", "relational_join", "config_structure", "ownership_graph")


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _opaque(rng: random.Random, prefix: str, length: int = 5) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return prefix + "".join(rng.choice(alphabet) for _ in range(length))


@dataclass(frozen=True, slots=True)
class TransferQuery:
    query_id: str
    template_id: str
    child_query: str
    answer: str


@dataclass(frozen=True, slots=True)
class TransferEvidence:
    evidence_id: str
    family: str
    split: BankSplit
    parent_template_id: str
    parent_evidence: str
    structured_evidence_sha256: str
    queries: tuple[TransferQuery, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["queries"] = [asdict(query) for query in self.queries]
        return value


@dataclass(frozen=True, slots=True)
class TransferBank:
    schema_version: str
    split: BankSplit
    seed: int
    records: tuple[TransferEvidence, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "split": self.split,
            "seed": self.seed,
            "records": [record.to_dict() for record in self.records],
        }

    def manifest_sha256(self) -> str:
        return _canonical_hash(
            {
                "bank": self.to_dict(),
                "moth_donors": assign_moth_donors(self),
            }
        )

    def artifact_dict(self) -> dict[str, object]:
        artifact: dict[str, object] = {
            "bank": self.to_dict(),
            "moth_donors": assign_moth_donors(self),
        }
        artifact["manifest_sha256"] = _canonical_hash(artifact)
        return artifact

    def parent_view(self) -> list[dict[str, str]]:
        return [
            {
                "evidence_id": record.evidence_id,
                "family": record.family,
                "parent_evidence": record.parent_evidence,
            }
            for record in self.records
        ]

    def child_view(self) -> list[dict[str, str]]:
        return [
            {
                "evidence_id": record.evidence_id,
                "query_id": query.query_id,
                "family": record.family,
                "child_query": query.child_query,
            }
            for record in self.records
            for query in record.queries
        ]

    def answer_key(self) -> list[dict[str, str]]:
        return [
            {
                "evidence_id": record.evidence_id,
                "query_id": query.query_id,
                "answer": query.answer,
            }
            for record in self.records
            for query in record.queries
        ]


def _templates(split: BankSplit) -> tuple[str, tuple[str, str, str]]:
    if split == "train":
        return "parent_train_v1", ("query_train_a", "query_train_b", "query_train_c")
    if split == "validation":
        return "parent_validation_v1", (
            "query_validation_a",
            "query_validation_b",
            "query_validation_c",
        )
    return "parent_held_out_v1", (
        "query_held_out_a",
        "query_held_out_b",
        "query_held_out_c",
    )


def _keyed_numeric(rng: random.Random, split: BankSplit) -> tuple[str, list[tuple[str, str]], object]:
    size = {"train": 6, "validation": 7, "held_out": 8}[split]
    records = [(_opaque(rng, "K"), rng.randrange(11, 90)) for _ in range(size)]
    a, b, c = rng.sample(records, 3)
    if split == "held_out":
        evidence = "Ledger entries, one per row:\n" + "\n".join(f"{key} => {value}" for key, value in records)
        questions = [
            (f"Return the sum of ledger keys {a[0]} and {b[0]}.", str(a[1] + b[1])),
            (f"Return the absolute difference between {b[0]} and {c[0]}.", str(abs(b[1] - c[1]))),
            (f"Return ({a[0]} + {c[0]}) modulo 17.", str((a[1] + c[1]) % 17)),
        ]
    else:
        evidence = "Numeric records:\n" + "\n".join(f"{key}: {value}" for key, value in records)
        questions = [
            (f"What is {a[0]} plus {b[0]}? Return only the number.", str(a[1] + b[1])),
            (f"What is the nonnegative difference of {b[0]} and {c[0]}?", str(abs(b[1] - c[1]))),
            (f"Add {a[0]} to {c[0]} and reduce modulo 17.", str((a[1] + c[1]) % 17)),
        ]
    return evidence, questions, records


def _relational_join(rng: random.Random, split: BankSplit) -> tuple[str, list[tuple[str, str]], object]:
    groups = [(_opaque(rng, "G"), _opaque(rng, "C")) for _ in range(4)]
    entities = [(_opaque(rng, "E"), rng.choice(groups)[0]) for _ in range(7)]
    group_codes = dict(groups)
    selected = rng.sample(entities, 3)
    if split == "held_out":
        evidence = "Membership edges:\n" + "\n".join(f"{entity} -> group {group}" for entity, group in entities)
        evidence += "\nGroup codes:\n" + "\n".join(f"{group} -> code {code}" for group, code in groups)
        questions = [
            (f"Follow both relations for {entity}; return its code.", group_codes[group]) for entity, group in selected
        ]
    else:
        evidence = "Entities by unit:\n" + "\n".join(f"{entity} belongs to {group}" for entity, group in entities)
        evidence += "\nUnit lookup:\n" + "\n".join(f"{group} has code {code}" for group, code in groups)
        questions = [
            (f"Which code is reached from entity {entity}? Return only the code.", group_codes[group])
            for entity, group in selected
        ]
    return evidence, questions, {"groups": groups, "entities": entities}


def _config_structure(rng: random.Random, split: BankSplit) -> tuple[str, list[tuple[str, str]], object]:
    sections: dict[str, dict[str, str]] = {}
    for _ in range(4):
        section = _opaque(rng, "S")
        sections[section] = {_opaque(rng, "P"): _opaque(rng, "V") for _ in range(3)}
    selected = rng.sample([(s, k, v) for s, values in sections.items() for k, v in values.items()], 3)
    if split == "held_out":
        lines = [
            f"<{section}> " + " | ".join(f"{key}={value}" for key, value in values.items())
            for section, values in sections.items()
        ]
        evidence = "Configuration snapshot:\n" + "\n".join(lines)
        questions = [
            (f"Look up property {key} inside component {section}; return its stored value.", value)
            for section, key, value in selected
        ]
    else:
        lines = []
        for section, values in sections.items():
            lines.append(f"[{section}]")
            lines.extend(f"{key}={value}" for key, value in values.items())
        evidence = "Configuration:\n" + "\n".join(lines)
        questions = [(f"Return only the value of {section}.{key}.", value) for section, key, value in selected]
    return evidence, questions, sections


def _ownership_graph(rng: random.Random, split: BankSplit) -> tuple[str, list[tuple[str, str]], object]:
    owners = [(_opaque(rng, "O"), _opaque(rng, "Z")) for _ in range(4)]
    assets = [(_opaque(rng, "A"), rng.choice(owners)[0]) for _ in range(7)]
    owner_zones = dict(owners)
    selected = rng.sample(assets, 3)
    if split == "held_out":
        evidence = "Custody graph:\n" + "\n".join(f"asset({asset}) -> custodian({owner})" for asset, owner in assets)
        evidence += "\n" + "\n".join(f"custodian({owner}) -> zone({zone})" for owner, zone in owners)
        questions = [
            (f"Traverse the custody graph from {asset}; return the terminal zone.", owner_zones[owner])
            for asset, owner in selected
        ]
    else:
        evidence = "Asset owners:\n" + "\n".join(f"{asset} owned by {owner}" for asset, owner in assets)
        evidence += "\nOwner zones:\n" + "\n".join(f"{owner} located in {zone}" for owner, zone in owners)
        questions = [
            (f"Which zone ultimately owns asset {asset}? Return only the zone.", owner_zones[owner])
            for asset, owner in selected
        ]
    return evidence, questions, {"owners": owners, "assets": assets}


_BUILDERS: dict[str, Callable[[random.Random, BankSplit], tuple[str, list[tuple[str, str]], object]]] = {
    "keyed_numeric": _keyed_numeric,
    "relational_join": _relational_join,
    "config_structure": _config_structure,
    "ownership_graph": _ownership_graph,
}


def build_transfer_bank(*, seed: int, split: BankSplit, examples_per_family: int) -> TransferBank:
    if split not in {"train", "validation", "held_out"}:
        raise ValueError("unknown transfer-bank split")
    if examples_per_family < 2:
        raise ValueError("at least two examples per family are required")
    rng = random.Random(seed)
    parent_template_id, query_template_ids = _templates(split)
    records: list[TransferEvidence] = []
    for family in FAMILIES:
        for example_index in range(examples_per_family):
            parent_evidence, questions, structured = _BUILDERS[family](rng, split)
            identity = _canonical_hash(
                {
                    "family": family,
                    "split": split,
                    "example_index": example_index,
                    "structured": structured,
                }
            )[:16]
            evidence_id = f"{split}-{family}-{identity}"
            queries = tuple(
                TransferQuery(
                    query_id=f"{evidence_id}-q{query_index}",
                    template_id=query_template_ids[query_index],
                    child_query=question,
                    answer=answer,
                )
                for query_index, (question, answer) in enumerate(questions)
            )
            records.append(
                TransferEvidence(
                    evidence_id=evidence_id,
                    family=family,
                    split=split,
                    parent_template_id=parent_template_id,
                    parent_evidence=parent_evidence,
                    structured_evidence_sha256=_canonical_hash(structured),
                    queries=queries,
                )
            )
    bank = TransferBank(
        schema_version="prime-rl/split-information-bank/v1",
        split=split,
        seed=seed,
        records=tuple(records),
    )
    validate_transfer_bank(bank)
    return bank


def validate_transfer_bank(bank: TransferBank) -> None:
    if bank.schema_version != "prime-rl/split-information-bank/v1":
        raise ValueError("unknown transfer-bank schema")
    evidence_ids = [record.evidence_id for record in bank.records]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("evidence IDs must be unique")
    if {record.family for record in bank.records} != set(FAMILIES):
        raise ValueError("transfer bank must contain every preregistered family")
    parent_payloads = [record.parent_evidence for record in bank.records]
    if len(parent_payloads) != len(set(parent_payloads)):
        raise ValueError("parent evidence must be unique")
    query_ids: set[str] = set()
    for record in bank.records:
        if record.split != bank.split:
            raise ValueError("record split does not match bank split")
        if len(record.queries) < 2:
            raise ValueError("each evidence packet must support multiple downstream queries")
        for query in record.queries:
            if query.query_id in query_ids:
                raise ValueError("query IDs must be unique")
            query_ids.add(query.query_id)
            if query.child_query in record.parent_evidence:
                raise ValueError("parent evidence contains the downstream query")
            if not query.answer:
                raise ValueError("query answer must not be empty")


def assign_moth_donors(bank: TransferBank) -> dict[str, str]:
    """Create a deterministic, family-matched derangement for MOTH."""

    donors: dict[str, str] = {}
    for family in FAMILIES:
        records = sorted(
            (record for record in bank.records if record.family == family),
            key=lambda record: (len(record.parent_evidence), record.evidence_id),
        )
        if len(records) < 2:
            raise ValueError("MOTH requires at least two evidence packets per family")
        if len(records) % 2 == 0:
            paired = [records[index ^ 1] for index in range(len(records))]
        else:
            paired = [records[index ^ 1] for index in range(len(records) - 3)]
            paired.extend((records[-2], records[-1], records[-3]))
        donors.update({record.evidence_id: donor.evidence_id for record, donor in zip(records, paired)})
    if any(source == donor for source, donor in donors.items()):
        raise ValueError("MOTH donor assignment must be a derangement")
    return donors
