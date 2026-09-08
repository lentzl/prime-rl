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

# Original synthetic explanatory chapters, separate from all evaluation sources.
EXPOSITORY_CHAPTERS = {
    "ferry-survey": (
        (
            "A fictional harbor survey compared passenger reports during four weeks before a timetable change and four weeks afterward. Reported waiting times fell after the change, but the later period also had fewer storms. The comparison therefore did not isolate the effect of the new timetable from the effect of better weather.",
            "Four-week before/after passenger reports show shorter waits; fewer later storms confound the timetable effect.",
            "Reported waits fell after the timetable change, but better weather prevents isolating its effect.",
        ),
        (
            "The survey asked passengers who had already boarded a ferry. It did not reach people who abandoned a trip or chose another route because they expected a long wait. The authors warned that satisfaction among current passengers could improve even while the experience of people excluded from the survey remained unchanged.",
            "Only boarded passengers were surveyed; abandoned trips and alternative-route users are excluded, limiting satisfaction claims.",
            "Surveying only boarded passengers excludes abandoned trips and alternative-route users, limiting satisfaction claims.",
        ),
        (
            "Morning passengers reported more reliable connections with the local bus, while evening passengers still described missed connections. Combining all responses into one average hid that difference. The authors proposed reporting results separately by time of day before deciding whether the new service helped the travelers with the least flexible schedules.",
            "Morning bus connections improved; evening misses persisted. Report by time of day before judging benefits to inflexible travelers.",
            "Morning bus connections improved while evening problems persisted; separate time-of-day results before judging who benefited.",
        ),
        (
            "The harbor team planned another survey during a stormier month and wanted to contact people who had stopped using the ferry. This was a proposal for better evidence, not a decision to reverse the timetable. No conclusion about ticket revenue was possible because the survey did not collect sales records.",
            "Propose stormier-month and former-user surveys, not timetable reversal; no revenue conclusion without sales data.",
            "Survey stormier months and former users; no timetable reversal or revenue conclusion follows from these data.",
        ),
    ),
    "oral-history": (
        (
            "A fictional town archive collected interviews about the closure of an old market. Interviews conducted decades afterward often described the market as the center of neighborhood life. The curator treated these recollections as evidence of remembered experience, not as a complete register of who used the market or how frequently they visited.",
            "Decades-later market interviews document remembered experience, not a complete visitor or frequency register.",
            "Market interviews reveal remembered experience, not complete visitor numbers or attendance frequency.",
        ),
        (
            "Contemporary photographs showed a crowded square on several festival days. Those images supported the claim that the market could attract large gatherings, but not that it was equally crowded on ordinary weekdays. The dates and occasions attached to each image mattered as much as the visible number of people.",
            "Festival photographs establish occasional crowds, not ordinary weekday attendance; preserve dates and occasions.",
            "Festival photographs show occasional crowds, not typical weekdays; dates and occasions are essential context.",
        ),
        (
            "Shop ledgers and personal letters sometimes contradicted the interviews. Rather than deleting conflicting accounts, the curator displayed them together and explained what each source could establish. Agreement between two later interviews was not automatically independent confirmation, since both speakers might have repeated the same published local history.",
            "Retain and explain conflicting ledgers, letters and interviews; later interviews sharing one published source are not independent confirmation.",
            "Explain conflicting sources together; interviews repeating one published history do not independently confirm it.",
        ),
        (
            "The exhibition separated established dates from uncertain explanations for the closure. Visitors were invited to contribute additional documents, but contributions would be cataloged with their origin before being used. New evidence could alter the explanation without requiring the archive to pretend that the earlier uncertainty had never existed.",
            "Separate established dates from uncertain causes; catalog origins of new documents and preserve the record of earlier uncertainty.",
            "Distinguish dates from uncertain causes; record new evidence's origins and retain earlier uncertainty when explanations change.",
        ),
    ),
    "repair-workshop": (
        (
            "In a fictional community workshop, volunteers logged fifty household items brought for repair over a month. Thirty items worked when their owners collected them. The organizers called this a collection-day result, not a long-term success rate, because they had not yet checked whether those items were still working several months later.",
            "Thirty of fifty items worked at collection; without later checks this is not a long-term repair success rate.",
            "Thirty of fifty items worked at collection, but long-term repair success was not measured.",
        ),
        (
            "Some unrepaired items needed parts that were unavailable locally, while others could not be opened without damage. The log distinguished these reasons from cases in which no volunteer had the relevant expertise. Calling every unfinished repair a skills failure would therefore misdescribe both the work and the resources the workshop needed.",
            "Separate missing parts, destructive access and missing expertise; unfinished repairs do not all indicate a skills failure.",
            "Distinguish unavailable parts, destructive access and missing expertise instead of treating every unfinished repair as a skills failure.",
        ),
        (
            "Owners who completed a feedback card often valued learning how their items worked, even when a repair was unsuccessful. However, filling out the card was optional. The organizers could report what respondents valued but could not assume that silent participants shared the same opinion or had received the same explanation.",
            "Optional feedback respondents valued learning even after failed repairs; do not generalize to silent participants.",
            "Respondents valued learning despite failed repairs, but optional feedback cannot represent silent participants.",
        ),
        (
            "For the following month, the workshop proposed a parts-sharing shelf and a scheduled expert session. It also planned to contact consenting owners after three months. These were responses to different gaps: supplies, expertise and durability evidence. The plan did not promise that every item would become repairable or that all owners would answer the follow-up.",
            "Parts shelf, expert session and consenting-owner three-month follow-up address supply, expertise and durability gaps; no universal success or response guarantee.",
            "Plan shared parts, expert sessions and consenting-owner three-month follow-up without promising universal repair or response.",
        ),
    ),
    "water-meter-trial": (
        (
            "A fictional housing cooperative tested weekly water-use statements in two buildings for six weeks. Their recorded use fell compared with the preceding six weeks. Another building without statements also used less water during the same period. The report therefore distinguished the decline within the trial buildings from evidence that statements caused the decline.",
            "Trial buildings used less water over six weeks, but an untreated building also declined; the statements' causal effect is not established.",
            "Water use fell in trial and untreated buildings, so the statements' causal effect remains unestablished.",
        ),
        (
            "The readings came from building-level meters rather than individual apartments. They included water used in shared spaces and could also reflect leaks. Although the statements divided totals by occupied apartments to make buildings easier to compare, this calculation did not reveal how much water any particular household had used.",
            "Building meters include shared use and possible leaks; totals per occupied apartment do not measure individual household use.",
            "Building totals include shared use and possible leaks; apartment averages do not identify household consumption.",
        ),
        (
            "Residents asked for clearer explanations of unusual weekly increases. The maintenance team proposed annotating known repairs and changes in occupancy alongside the readings. Such annotations would help interpret the totals, but an unexplained increase would still be a question to investigate rather than proof that residents had ignored the statements.",
            "Annotate repairs and occupancy changes; unexplained increases warrant investigation, not accusations that residents ignored statements.",
            "Annotate repairs and occupancy; unexplained increases require investigation, not assumptions about residents' behavior.",
        ),
        (
            "The cooperative planned a longer comparison with additional buildings before deciding whether to continue the statements. It also wanted to record the staff time needed to prepare them. Lower water use alone would not establish that the reporting process was worthwhile if its costs and residents' understanding remained unknown.",
            "Plan longer, broader comparison and staff-time measurement; assess costs and residents' understanding before judging reporting worthwhile.",
            "Extend comparisons and measure staff time, costs and residents' understanding before judging the statements worthwhile.",
        ),
    ),
}


def training_chapters():
    chapters = [dict(row) for row in TRAINING_CHAPTERS]
    for slug, records in EXPOSITORY_CHAPTERS.items():
        chapters.append({
            "slug": slug, "family": "expository",
            "paragraphs": [{"id": f"{slug}-p{i:02d}", "text": record[0]}
                           for i, record in enumerate(records, 1)],
        })
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
        records = (
            tuple((note, bullet) for _, note, bullet in EXPOSITORY_CHAPTERS[chapter["slug"]])
            if chapter["slug"] in EXPOSITORY_CHAPTERS else TEACHER_RECORDS[chapter["slug"]]
        )
        chapter["notes"] = (
            "\n".join(
                f"[{source['id']}] {note}" for source, (note, _) in zip(chapter["paragraphs"], records, strict=True)
            )
            + "\n"
        )
        chapter["summary"] = "\n".join(f"- {bullet}" for _, bullet in records) + "\n"
    return chapters
