#!/usr/bin/env python3
"""Build a held-out-safe Prime Agent SFT set for grounded chapter summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from export_q35_2b_document_decision_sft_v1 import _wire_message, sha256_file

SCHEMA_VERSION = "qwen35-2b-document-summary-worker-sft/v1"
OBJECTIVE = "grounded_english_chapter_summary_report"
OUTPUT_PATH = "/logs/artifacts/document-summary-v1/worker-report.json"
EVALUATION_DOCUMENT_ID = "project-northstar-playbook-v1"


def _chapter(
    slug: str,
    family: str,
    title: str,
    paragraphs: tuple[str, str, str, str],
    bullets: tuple[tuple[str, tuple[int, ...]], ...],
) -> dict[str, Any]:
    rows = [
        {
            "id": f"{slug}-p{index:02d}",
            "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "text": text,
        }
        for index, text in enumerate(paragraphs, 1)
    ]
    report = {
        "worker": f"{slug}-summarizer",
        "chapter_id": slug,
        "bullets": [
            {
                "id": f"{slug}-b{index:02d}",
                "text": text,
                "source_ids": [f"{slug}-p{source:02d}" for source in sources],
            }
            for index, (text, sources) in enumerate(bullets, 1)
        ],
        "issues": [],
    }
    return {
        "slug": slug,
        "family": family,
        "title": title,
        "paragraphs": rows,
        "report": report,
    }


TRAINING_CHAPTERS = (
    _chapter(
        "clinic-pilot",
        "planning",
        "Clinic scheduling pilot",
        (
            "The pilot replaces telephone appointment requests with a web form at the East and River clinics for six weeks.",
            "Urgent-care visits and appointments requiring an interpreter remain on the telephone workflow during the pilot.",
            "Success requires 90 percent of routine requests to receive a proposed time within one business day.",
            "The review must compare completion time, abandoned requests, and patient complaints against the prior six-week period.",
        ),
        (
            (
                "Run a six-week web-form pilot at East and River clinics while keeping urgent and interpreter-assisted appointments on the telephone workflow.",
                (1, 2),
            ),
            ("Meet the target by proposing times for 90 percent of routine requests within one business day.", (3,)),
            ("Compare completion time, abandoned requests, and complaints with the preceding six weeks.", (4,)),
        ),
    ),
    _chapter(
        "library-scan",
        "planning",
        "Library archive digitization",
        (
            "The first digitization batch contains local newspapers from 1970 through 1985 and excludes photographs with unclear ownership.",
            "Each scan receives a stable archive identifier before image cleanup begins.",
            "Operators record missing pages and unreadable sections instead of silently omitting them.",
            "A batch is accepted when every physical item has an identifier, a scan, and a recorded quality outcome.",
        ),
        (
            ("Digitize 1970–1985 local newspapers first, excluding photographs whose ownership is unclear.", (1,)),
            (
                "Assign a stable archive identifier before cleanup and explicitly record missing or unreadable material.",
                (2, 3),
            ),
            ("Accept a batch only when every physical item has an identifier, scan, and quality outcome.", (4,)),
        ),
    ),
    _chapter(
        "vendor-onboarding",
        "planning",
        "Vendor onboarding sequence",
        (
            "Procurement opens a vendor record only after receiving the legal name, tax identifier, payment country, and primary contact.",
            "Security review is mandatory for vendors that will access customer data or production systems.",
            "Finance validates bank details through a channel separate from the original submission.",
            "No purchase order may be issued until procurement, security when applicable, and finance have recorded their decisions.",
        ),
        (
            ("Open the vendor record after collecting legal, tax, payment-country, and primary-contact details.", (1,)),
            (
                "Require security review for customer-data or production access, and verify bank details through a separate channel.",
                (2, 3),
            ),
            (
                "Issue no purchase order until procurement, applicable security, and finance decisions are recorded.",
                (4,),
            ),
        ),
    ),
    _chapter(
        "regional-launch",
        "planning",
        "Regional service launch",
        (
            "The service launches in Cork and Malmö on 12 March, with support available from 08:00 to 18:00 local time.",
            "Enterprise accounts migrate first; trial accounts remain on the existing service for the first month.",
            "The launch owner reviews error rate and queue delay every two hours during the first day.",
            "Rollback begins if error rate exceeds 3 percent for two consecutive reviews or queue delay exceeds ten minutes once.",
        ),
        (
            (
                "Launch in Cork and Malmö on 12 March with local support from 08:00 to 18:00, migrating enterprise accounts first.",
                (1, 2),
            ),
            ("Review error rate and queue delay every two hours throughout launch day.", (3,)),
            (
                "Rollback after two consecutive error reviews above 3 percent or any queue delay above ten minutes.",
                (4,),
            ),
        ),
    ),
    _chapter(
        "incident-response",
        "operations",
        "Incident response",
        (
            "A severity-one incident pages the on-call lead immediately and opens a shared incident channel.",
            "The lead assigns one incident commander and one communications owner; neither role may be left implicit.",
            "Status updates are posted every thirty minutes even when there is no material change.",
            "Resolution requires a verified service check, a customer update, and an owner for the follow-up review.",
        ),
        (
            ("For severity one, page the on-call lead immediately and open a shared incident channel.", (1,)),
            (
                "Explicitly assign incident-command and communications owners, then publish status every thirty minutes even without change.",
                (2, 3),
            ),
            ("Close only after verifying service, updating customers, and assigning the follow-up review.", (4,)),
        ),
    ),
    _chapter(
        "warehouse-dispatch",
        "operations",
        "Warehouse dispatch",
        (
            "Pickers scan each order and storage location before removing an item from the shelf.",
            "A second person verifies controlled goods and orders worth more than 2,000 euros.",
            "Packages receive a carrier label only after weight and destination have been checked against the order.",
            "Damaged packaging is photographed and replaced; the original damage record remains attached to the order.",
        ),
        (
            ("Scan both order and location before picking any shelf item.", (1,)),
            ("Use second-person verification for controlled goods and orders above €2,000.", (2,)),
            (
                "Check weight and destination before labeling; photograph and replace damage while retaining its order record.",
                (3, 4),
            ),
        ),
    ),
    _chapter(
        "refund-review",
        "operations",
        "Subscription refund review",
        (
            "Agents may approve refunds up to 100 euros when the request is within thirty days and account ownership is verified.",
            "Larger or older requests require a billing specialist's decision.",
            "Every refund record states the reason, original payment identifier, amount, and approving person.",
            "A rejected request receives a customer-facing explanation and an escalation route.",
        ),
        (
            (
                "Agents may refund up to €100 within thirty days after verifying account ownership; larger or older requests go to billing specialists.",
                (1, 2),
            ),
            ("Record the reason, payment identifier, amount, and approver for every refund.", (3,)),
            ("Explain rejections to the customer and provide an escalation route.", (4,)),
        ),
    ),
    _chapter(
        "research-review",
        "operations",
        "Research review workflow",
        (
            "Two reviewers independently assess each proposal for feasibility, risk, and expected learning value.",
            "Reviewers disclose conflicts before reading the detailed proposal and recuse themselves when necessary.",
            "Disagreements are discussed only after both initial scores are locked.",
            "The final record preserves both initial scores, the discussion outcome, and the named decision owner.",
        ),
        (
            (
                "Have two independent reviewers score feasibility, risk, and learning value after resolving conflicts and recusals.",
                (1, 2),
            ),
            ("Lock both initial scores before reviewers discuss disagreements.", (3,)),
            (
                "Preserve initial scores, the discussion outcome, and the named decision owner in the final record.",
                (4,),
            ),
        ),
    ),
    _chapter(
        "retention-policy",
        "safety",
        "Data retention exceptions",
        (
            "Routine support recordings are deleted after ninety days unless a documented investigation requires a hold.",
            "A hold identifies its owner, scope, approval date, and next review date.",
            "Ending a hold resumes the ordinary deletion schedule rather than resetting the retention clock.",
            "Monthly audits list expired recordings, active holds, overdue hold reviews, and deletion failures.",
        ),
        (
            (
                "Delete routine support recordings after ninety days unless a documented investigation hold applies.",
                (1,),
            ),
            (
                "Each hold needs an owner, scope, approval date, and next review; ending it resumes rather than resets retention.",
                (2, 3),
            ),
            ("Audit expired recordings, active holds, overdue reviews, and deletion failures monthly.", (4,)),
        ),
    ),
    _chapter(
        "lab-access",
        "safety",
        "Laboratory access",
        (
            "Visitors enter the wet laboratory only with a trained host and visible temporary badge.",
            "Protective eyewear and closed shoes are mandatory beyond the marked entry line.",
            "A chemical spill stops all nearby work and is reported to the safety officer before cleanup begins.",
            "The host records visitor name, arrival, departure, and any incident before returning the badge.",
        ),
        (
            ("Wet-lab visitors need a trained host and visible temporary badge.", (1,)),
            (
                "Require eye protection and closed shoes beyond the entry line; stop work and notify safety before cleaning a spill.",
                (2, 3),
            ),
            ("Record visitor identity, arrival, departure, and incidents before the badge is returned.", (4,)),
        ),
    ),
    _chapter(
        "account-reconcile",
        "safety",
        "Account reconciliation",
        (
            "The daily reconciliation compares payment-provider totals with ledger totals for each currency.",
            "Differences below five euros are recorded but may be carried to the next business day.",
            "Larger differences freeze the affected payout batch until a finance reviewer documents the cause.",
            "Manual ledger adjustments require the preparer's name, a separate approver, and links to supporting evidence.",
        ),
        (
            ("Compare provider and ledger totals by currency every day, recording even differences below €5.", (1, 2)),
            ("Freeze payouts for larger differences until finance documents the cause.", (3,)),
            ("Manual adjustments require a named preparer, separate approver, and linked evidence.", (4,)),
        ),
    ),
    _chapter(
        "publication-fix",
        "safety",
        "Publication corrections",
        (
            "Minor spelling fixes may be published without notice when they do not change meaning.",
            "A factual correction adds a dated notice describing the original statement and corrected information.",
            "Changing a reported measurement requires editor approval and a link to the revised underlying data.",
            "Retractions preserve the original URL and display the reason, decision date, and responsible editor.",
        ),
        (
            (
                "Publish meaning-neutral spelling fixes silently, but attach a dated notice to factual corrections.",
                (1, 2),
            ),
            ("Measurement changes need editor approval and a link to revised source data.", (3,)),
            ("Retractions retain the original URL and show the reason, date, and responsible editor.", (4,)),
        ),
    ),
)


def _source_context(traces: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]], Path, str]:
    candidates = []
    for path in traces:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                envelope = json.loads(line)
                for trace in envelope.get("traces") or []:
                    nodes = trace.get("nodes") or []
                    tools = trace.get("tools") or []
                    if (
                        trace.get("task", {}).get("type") == "DocumentSummaryWorkerTask"
                        and len(nodes) >= 2
                        and nodes[0].get("parent") is None
                        and nodes[0].get("message", {}).get("role") == "user"
                        and [tool.get("name") for tool in tools] == ["ipython"]
                    ):
                        candidates.append((nodes[0]["message"], tools, path.resolve(), trace["id"]))
    if len(candidates) != 1:
        raise ValueError(f"expected one Prime Agent summary-worker context, found {len(candidates)}")
    return candidates[0]


def _job(chapter: dict[str, Any]) -> dict[str, Any]:
    slug = chapter["slug"]
    return {
        "schema_version": "prime-rl/document-chapter-summary-job/v1",
        "document_id": f"training-handbook-{slug}-v1",
        "worker": f"{slug}-summarizer",
        "chapter_id": slug,
        "chapter_title": chapter["title"],
        "path": f"/workspace/document-summary-training-v1/jobs/{slug}.json",
        "paragraphs": chapter["paragraphs"],
        "task_contract": {
            "operation": "summarize_chapter_in_english",
            "bullet_count": 3,
            "bullet_ids": [f"{slug}-b{index:02d}" for index in range(1, 4)],
            "report_keys": ["worker", "chapter_id", "bullets", "issues"],
            "bullet_keys": ["id", "text", "source_ids"],
            "requirements": [
                "Capture the most decision-relevant facts without copying whole paragraphs.",
                "Use concise English bullets of 5 to 45 words each.",
                "Represent each bullet as an object with exactly id, text, and source_ids.",
            "Use the supplied bullet IDs once each and in the supplied order.",
            "Ground every bullet in one or more exact paragraph IDs.",
            "Use every paragraph ID exactly once across the three bullets.",
            "Cite only paragraphs whose facts appear in that bullet; combine the most closely related pair when four paragraphs must become three bullets.",
                "Use an empty JSON list for issues when there are no issues.",
            ],
            "delivery": f"write_json:{OUTPUT_PATH}",
        },
    }


def _messages(runtime_message: dict[str, Any], chapter: dict[str, Any]) -> list[dict[str, Any]]:
    job = _job(chapter)
    report = chapter["report"]
    read_id = f"summary-read-{hashlib.sha256(chapter['slug'].encode()).hexdigest()[:16]}"
    write_id = f"summary-write-{hashlib.sha256((chapter['slug'] + ':write').encode()).hexdigest()[:16]}"
    read_code = (
        "import json\nfrom pathlib import Path\n"
        f"job_path = Path({job['path']!r})\n"
        "job = json.loads(job_path.read_text(encoding='utf-8'))\njob"
    )
    write_code = (
        "from pathlib import Path\nimport json\n"
        f"report = {report!r}\n"
        f"output_path = Path({OUTPUT_PATH!r})\n"
        "assert set(report) == {'worker', 'chapter_id', 'bullets', 'issues'}\n"
        "assert all(set(row) == {'id', 'text', 'source_ids'} for row in report['bullets'])\n"
        "expected_source_ids = {row['id'] for row in job['paragraphs']}\n"
        "ordered_source_ids = [source_id for bullet in report['bullets'] for source_id in bullet['source_ids']]\n"
        "assert set(ordered_source_ids) == expected_source_ids\n"
        "assert len(ordered_source_ids) == len(expected_source_ids)\n"
        "assert isinstance(report['issues'], list)\n"
        "output_path.parent.mkdir(parents=True, exist_ok=True)\n"
        "written = output_path.write_text(json.dumps(report, indent=2) + '\\n', encoding='utf-8')\n"
        "written"
    )
    task_prompt = (
        "Act as the terminal chapter summarizer, not a coordinator. Do not spawn a child. "
        f"Read only `{job['path']}`, follow its task_contract exactly, and write the complete "
        f"JSON report to `{OUTPUT_PATH}`. Use Path(output_path).write_text with json.dumps; "
        "issues must be a list, bullets must be objects with the exact required keys, and each "
        "paragraph ID must appear exactly once beside the bullet that actually summarizes it. "
        "Stop after the file exists."
    )
    runtime = _wire_message(runtime_message)
    if not isinstance(runtime.get("content"), str):
        raise ValueError("summary worker runtime context must contain text")
    runtime["content"] += (
        "\n\nFor chapter summaries, keep each bullet as a Python dict until the complete report is "
        "serialized. Use literal paragraph ID strings, cite each paragraph exactly once, and "
        "keep every cited source's facts in that same bullet."
    )
    return [
        runtime,
        {"role": "user", "content": task_prompt},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "I will read the assigned job before authoring its grounded summary.",
            "tool_calls": [
                {
                    "id": read_id,
                    "type": "function",
                    "function": {
                        "name": "ipython",
                        "arguments": json.dumps({"code": read_code}, separators=(",", ":")),
                    },
                }
            ],
        },
        {"role": "tool", "content": repr(job), "tool_call_id": read_id},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": (
                "I have the source and exact contract. I will write three concise bullet objects, "
                "keep each fact with its literal source ID, cover every paragraph exactly once, "
                "and use the explicit output path rather than the input path."
            ),
            "tool_calls": [
                {
                    "id": write_id,
                    "type": "function",
                    "function": {
                        "name": "ipython",
                        "arguments": json.dumps({"code": write_code}, separators=(",", ":")),
                    },
                }
            ],
        },
        {"role": "tool", "content": str(len(json.dumps(report, indent=2)) + 1), "tool_call_id": write_id},
        {
            "role": "assistant",
            "content": f"Wrote the grounded three-bullet summary to {OUTPUT_PATH}.",
            "reasoning_content": "The artifact write succeeded, so the task is complete and I must stop.",
            "tool_calls": [],
        },
    ]


def export(*, traces: list[Path], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite summary worker SFT: {output_dir}")
    runtime_message, tools, source_path, source_trace_id = _source_context(traces)
    if len(TRAINING_CHAPTERS) != 12 or EVALUATION_DOCUMENT_ID in repr(TRAINING_CHAPTERS):
        raise ValueError("summary worker training set overlaps or has the wrong size")
    rows = [
        {
            "messages": _messages(runtime_message, chapter),
            "tools": json.dumps(tools, sort_keys=True, separators=(",", ":")),
            "task_key": f"summary-worker-{chapter['slug']}",
            "trace_id": f"summary-worker-authored:{chapter['slug']}",
            "family": f"summary_{chapter['family']}",
            "role": "child",
            "objective": OBJECTIVE,
            "source_trace": str(source_path),
        }
        for chapter in TRAINING_CHAPTERS
    ]
    family_counts = {
        family: sum(row["family"] == family for row in rows) for family in sorted({row["family"] for row in rows})
    }
    if set(family_counts.values()) != {4} or len({row["task_key"] for row in rows}) != 12:
        raise ValueError("summary worker training set is not balanced and unique")

    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "role": "child",
        "objective": OBJECTIVE,
        "rows": len(rows),
        "family_counts": family_counts,
        "task_keys": [row["task_key"] for row in rows],
        "source_traces": [{"path": str(source_path), "sha256": sha256_file(source_path)}],
        "source_trace_id": source_trace_id,
        "answer_free": False,
        "authored_reference": True,
        "native_prime_agent_context": True,
        "evaluation_document_excluded": True,
        "evaluation_document_id": EVALUATION_DOCUMENT_ID,
        "tool_call_format": "openai_function_v1",
        "dataset": {"path": parquet.name, "sha256": sha256_file(parquet)},
    }
    (output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                traces=[path.resolve() for path in args.traces],
                output_dir=args.output_dir.resolve(),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
