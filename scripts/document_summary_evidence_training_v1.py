"""Authored TRAIN-only notes and realizations; never supplied to live evaluation."""

from export_q35_2b_document_summary_worker_sft_v1 import TRAINING_CHAPTERS

# Each record pairs a source-faithful note with its concise final bullet.
TEACHER_RECORDS = {
    "clinic-pilot": (
        (
            "East and River clinics replace telephone requests with a web form for a six-week pilot.",
            "Replace telephone requests with a six-week web-form pilot at East and River clinics.",
        ),
        (
            "Urgent-care and interpreter-required appointments keep the telephone workflow throughout the pilot.",
            "Keep urgent-care and interpreter-required appointments on the telephone workflow.",
        ),
        (
            "Success: 90 percent of routine requests receive a proposed time within one business day.",
            "Propose times within one business day for 90 percent of routine requests.",
        ),
        (
            "Review compares completion time, abandoned requests and patient complaints with the preceding six weeks.",
            "Compare completion time, abandoned requests and complaints against the preceding six weeks.",
        ),
    ),
    "library-scan": (
        (
            "First batch: local newspapers from 1970 through 1985; exclude photographs whose ownership is unclear.",
            "Start with 1970–1985 local newspapers; exclude photographs with unclear ownership.",
        ),
        (
            "Give each scan a stable archive identifier before image cleanup.",
            "Assign each scan a stable identifier before cleanup.",
        ),
        (
            "Operators record missing pages and unreadable sections; never silently omit them.",
            "Operators must record missing pages and unreadable sections, never silently omit them.",
        ),
        (
            "Accept a batch only when every physical item has an identifier, scan and recorded quality outcome.",
            "Accept batches only when every physical item has an identifier, scan and quality outcome.",
        ),
    ),
    "vendor-onboarding": (
        (
            "Procurement opens a record only after legal name, tax identifier, payment country and primary contact arrive.",
            "Procurement needs legal name, tax ID, payment country and primary contact before opening records.",
        ),
        (
            "Security review is required if a vendor will access customer data OR production systems.",
            "Require security review for customer-data or production-system access.",
        ),
        (
            "Finance verifies bank details through a channel separate from the original submission.",
            "Finance verifies bank details through a separate channel.",
        ),
        (
            "Do not issue a purchase order until procurement, finance and security when applicable record decisions.",
            "Issue purchase orders only after recorded procurement, finance and applicable security decisions.",
        ),
    ),
    "regional-launch": (
        (
            "Launch Cork and Malmö on 12 March; local support runs 08:00 to 18:00.",
            "Launch Cork and Malmö on 12 March; support runs 08:00–18:00 locally.",
        ),
        (
            "Enterprise accounts migrate first; trial accounts stay on the existing service for the first month.",
            "Migrate enterprise accounts first; keep trials on the existing service for one month.",
        ),
        (
            "Launch owner reviews both error rate and queue delay every two hours on day one.",
            "On day one, launch owner reviews errors and queue delay every two hours.",
        ),
        (
            "Begin rollback if errors exceed 3 percent in two consecutive reviews OR queue delay exceeds ten minutes once.",
            "Rollback for errors above 3 percent twice consecutively or queue delay above ten minutes once.",
        ),
    ),
    "incident-response": (
        (
            "Severity one immediately pages the on-call lead and opens a shared incident channel.",
            "Severity one immediately pages the on-call lead and opens a shared incident channel.",
        ),
        (
            "The lead explicitly assigns one incident commander and one communications owner; neither role is implicit.",
            "The lead explicitly assigns an incident commander and communications owner.",
        ),
        (
            "Publish status every thirty minutes even if nothing material changed.",
            "Publish status every thirty minutes, even without material change.",
        ),
        (
            "Resolution needs a verified service check, customer update and named follow-up-review owner.",
            "Resolve only after verifying service, updating customers and assigning the follow-up-review owner.",
        ),
    ),
    "warehouse-dispatch": (
        (
            "Pickers scan both order and storage location before removing an item from its shelf.",
            "Pickers scan order and storage location before removing shelf items.",
        ),
        (
            "A second person verifies controlled goods and orders worth more than 2,000 euros.",
            "Require second-person verification for controlled goods and orders above €2,000.",
        ),
        (
            "Carrier labeling follows checking package weight and destination against the order.",
            "Check weight and destination against the order before carrier labeling.",
        ),
        (
            "Photograph and replace damaged packaging; retain the original damage record on the order.",
            "Photograph and replace damaged packaging; retain its original order-linked damage record.",
        ),
    ),
    "refund-review": (
        (
            "Agents can approve at most 100 euros only within thirty days and after verifying account ownership.",
            "Agents may refund up to €100 within thirty days after verifying account ownership.",
        ),
        (
            "A billing specialist decides requests exceeding the amount OR age limit.",
            "Billing specialists decide larger or older requests.",
        ),
        (
            "Every refund records reason, original payment identifier, amount and approving person.",
            "Record each refund's reason, original payment ID, amount and approver.",
        ),
        (
            "Rejections need a customer-facing explanation and an escalation route.",
            "Explain rejections to customers and provide an escalation route.",
        ),
    ),
    "research-review": (
        (
            "Two reviewers independently assess each proposal's feasibility, risk and expected learning value.",
            "Two reviewers independently score feasibility, risk and expected learning value.",
        ),
        (
            "Reviewers disclose conflicts before reading the detailed proposal and recuse themselves when necessary.",
            "Disclose conflicts before reading details; recuse when necessary.",
        ),
        (
            "Discussion of disagreements must wait until both initial scores are locked.",
            "Lock both initial scores before discussing disagreements.",
        ),
        (
            "Final record retains both initial scores, discussion outcome and named decision owner.",
            "Retain both initial scores, discussion outcome and named decision owner.",
        ),
    ),
    "retention-policy": (
        (
            "Delete routine support recordings after ninety days unless a documented investigation hold applies.",
            "Delete routine support recordings after ninety days, except documented investigation holds.",
        ),
        (
            "Each hold identifies owner, scope, approval date and next review date.",
            "Holds identify owner, scope, approval date and next review date.",
        ),
        (
            "Ending a hold resumes ordinary deletion timing; it does not reset the retention clock.",
            "Ending holds resumes deletion schedules without resetting retention clocks.",
        ),
        (
            "Monthly audit lists expired recordings, active holds, overdue hold reviews and deletion failures.",
            "Monthly audits list expired recordings, active holds, overdue hold reviews and deletion failures.",
        ),
    ),
    "lab-access": (
        (
            "Wet-lab visitors require a trained host and visible temporary badge.",
            "Wet-lab visitors require a trained host and visible temporary badge.",
        ),
        (
            "Beyond the marked entry line, protective eyewear and closed shoes are mandatory.",
            "Require protective eyewear and closed shoes beyond the marked entry line.",
        ),
        (
            "Chemical spills stop all nearby work and require reporting to the safety officer before cleanup.",
            "Chemical spills stop nearby work; report to the safety officer before cleanup.",
        ),
        (
            "Before returning the badge, the host records visitor name, arrival, departure and any incident.",
            "Before badge return, hosts record visitor name, arrival, departure and incidents.",
        ),
    ),
    "account-reconcile": (
        (
            "Daily reconciliation compares payment-provider totals and ledger totals separately for each currency.",
            "Compare payment-provider and ledger totals daily by currency.",
        ),
        (
            "Differences below five euros are recorded and may be carried to the next business day.",
            "Record sub-€5 differences; carryover to the next business day is allowed.",
        ),
        (
            "Larger differences freeze only the affected payout batch until a finance reviewer documents the cause.",
            "Freeze affected payout batches for larger differences until a finance reviewer documents the cause.",
        ),
        (
            "Manual ledger adjustments require preparer name, separate approver and supporting-evidence links.",
            "Manual adjustments need preparer name, separate approver and supporting-evidence links.",
        ),
    ),
    "publication-fix": (
        (
            "Minor spelling fixes may appear without notice only when meaning is unchanged.",
            "Minor spelling fixes need no notice only if meaning stays unchanged.",
        ),
        (
            "Factual corrections add a dated notice with the original statement and corrected information.",
            "Factual corrections need dated notices containing original and corrected information.",
        ),
        (
            "Changing a reported measurement requires editor approval and a link to revised underlying data.",
            "Measurement changes require editor approval and links to revised underlying data.",
        ),
        (
            "Retractions preserve the original URL and show reason, decision date and responsible editor.",
            "Retractions retain original URLs and show reason, decision date and responsible editor.",
        ),
    ),
    "permit-additional": (
        (
            "The duty planner approves routine delivery permits; hazardous-load permits additionally need safety-manager approval.",
            "Duty planners approve routine delivery permits; hazardous loads additionally need safety-manager approval.",
        ),
        (
            "Record vehicle registration, loading bay and approved arrival window before releasing a permit.",
            "Before release, record vehicle registration, loading bay and approved arrival window.",
        ),
        (
            "If the permit service is offline, planners use numbered paper forms and log issuance times.",
            "During permit-service outages, planners use numbered paper forms and log issuance times.",
        ),
        (
            "After recovery, import paper forms preserving original numbers and times; do not issue replacement permits.",
            "After recovery, import forms with original numbers and times; issue no replacement permits.",
        ),
    ),
    "permit-exclusive": (
        (
            "The duty planner approves routine delivery permits; only the safety manager approves hazardous-load permits, replacing planner approval.",
            "Duty planners approve routine permits; hazardous-load approval belongs solely to safety managers, replacing planner approval.",
        ),
        (
            "Record vehicle registration, loading bay and approved arrival window before releasing a permit.",
            "Before release, record vehicle registration, loading bay and approved arrival window.",
        ),
        (
            "If the permit service is offline, planners use numbered paper forms and log issuance times.",
            "During permit-service outages, planners use numbered paper forms and log issuance times.",
        ),
        (
            "After recovery, import paper forms preserving original numbers and times; do not issue replacement permits.",
            "After recovery, import forms with original numbers and times; issue no replacement permits.",
        ),
    ),
    "sensor-consecutive": (
        (
            "Pause cold-storage dispatch only after two consecutive sensor reviews above eight degrees; an isolated excursion does not trigger a pause.",
            "Pause dispatch after two consecutive reviews above eight degrees, not an isolated excursion.",
        ),
        (
            "The shift supervisor reviews sensor temperatures every ten minutes and records sensor ID and time.",
            "Shift supervisors review temperatures every ten minutes, recording sensor IDs and review times.",
        ),
        (
            "During a sensor-network outage, staff use a calibrated handheld probe; record location, temperature and measurement time.",
            "During network outages, use calibrated handheld probes; record location, temperature and measurement time.",
        ),
        (
            "Resume dispatch only after a technician confirms network recovery and the supervisor authorizes restart in the log.",
            "Resume only after technician-confirmed network recovery and the supervisor's logged restart authorization.",
        ),
    ),
    "sensor-single": (
        (
            "Pause cold-storage dispatch after any single sensor review above eight degrees; never wait for a second consecutive excursion.",
            "Pause dispatch after one review above eight degrees; never wait for a second excursion.",
        ),
        (
            "The shift supervisor reviews sensor temperatures every ten minutes and records sensor ID and time.",
            "Shift supervisors review temperatures every ten minutes, recording sensor IDs and review times.",
        ),
        (
            "During a sensor-network outage, staff use a calibrated handheld probe; record location, temperature and measurement time.",
            "During network outages, use calibrated handheld probes; record location, temperature and measurement time.",
        ),
        (
            "Resume dispatch only after a technician confirms network recovery and the supervisor authorizes restart in the log.",
            "Resume only after technician-confirmed network recovery and the supervisor's logged restart authorization.",
        ),
    ),
}

PERMIT_SOURCE = (
    "Before releasing any delivery permit, record the vehicle registration, the assigned loading bay and the approved arrival window.",
    "During a permit-service outage, duty planners use numbered paper forms instead of electronic permits and record each form's issuance time.",
    "After the permit service recovers, import all paper forms while preserving their original numbers and issuance times; do not issue replacement permits.",
)
SENSOR_SOURCE = (
    "The shift supervisor reviews all sensor temperatures every ten minutes and records the sensor identifier together with the review time.",
    "During an outage of the sensor network, staff use a calibrated handheld probe and record its measurement location, temperature and measurement time.",
    "Dispatch may resume only after a technician confirms network recovery and the shift supervisor records authorization to restart in the log.",
)


def training_chapters():
    chapters = [dict(row) for row in TRAINING_CHAPTERS]
    for slug, paragraphs in {
        "permit-additional": (
            "The duty planner approves routine delivery permits. A permit for a hazardous load also requires approval from the safety manager.",
            *PERMIT_SOURCE,
        ),
        "permit-exclusive": (
            "The duty planner approves routine delivery permits. Only the safety manager approves hazardous-load permits; that decision replaces rather than supplements planner approval.",
            *PERMIT_SOURCE,
        ),
        "sensor-consecutive": (
            "Pause cold-storage dispatch when sensor temperature exceeds eight degrees in two consecutive reviews. A single isolated excursion does not trigger a pause.",
            *SENSOR_SOURCE,
        ),
        "sensor-single": (
            "Pause cold-storage dispatch whenever a single sensor review exceeds eight degrees. Do not wait for a second consecutive excursion before pausing.",
            *SENSOR_SOURCE,
        ),
    }.items():
        chapters.append(
            {
                "slug": slug,
                "family": "contrasts",
                "paragraphs": [{"id": f"{slug}-p{i:02d}", "text": text} for i, text in enumerate(paragraphs, 1)],
            }
        )
    for chapter in chapters:
        records = TEACHER_RECORDS[chapter["slug"]]
        chapter["notes"] = (
            "\n".join(
                f"[{source['id']}] {note}" for source, (note, _) in zip(chapter["paragraphs"], records, strict=True)
            )
            + "\n"
        )
        chapter["summary"] = "\n".join(f"- {bullet}" for _, bullet in records) + "\n"
    return chapters
