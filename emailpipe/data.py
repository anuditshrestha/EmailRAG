"""Deterministic synthetic email archive.

~50 random filler messages plus a hand-planted multi-hop chain:

  Project Falcon is delayed by vendor Kestrel Logistics  (Dana, Oct 30)
  -> the Kestrel Logistics contract amendment was approved by Marcus Lee (Oct 31)

with distractors (Project Heron / Orion Freight / a different approver) so that a
single keyword search on the final question can land on the wrong answer.
"""
from __future__ import annotations

import random
from datetime import datetime

from .models import Email

ME = "Alex Morgan <alex.morgan@acme.example>"

PEOPLE = {
    "Dana Whitfield": "dana.whitfield@acme.example",
    "Marcus Lee": "marcus.lee@acme.example",
    "Sofia Alvarez": "sofia.alvarez@acme.example",
    "Priya Raman": "priya.raman@acme.example",
    "Tom Becker": "tom.becker@acme.example",
    "Jun Park": "jun.park@acme.example",
    "Hannah Cole": "hannah.cole@acme.example",
}

PROJECTS = ["Project Sparrow", "Project Osprey", "Project Lark", "Project Wren", "Project Plover"]
DETAILS = [
    "the data migration is 70% complete",
    "QA found three minor defects, all triaged",
    "the design review went well",
    "pilot users report faster load times",
    "the integration tests are green",
]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TIMES = ["10am", "11am", "2pm", "3:30pm"]
ITEMS = ["standing desks", "noise-cancelling headphones", "ergonomic chairs", "monitor arms"]
SHOPS = ["CloudPrint", "Harbour Coffee", "Rapid Courier", "Paper & Pine"]
NEWS_TOPICS = ["five dashboards worth stealing", "SQL tips", "data contracts explained", "a primer on vector search"]

D = lambda m, d, hh, mm: datetime(2026, m, d, hh, mm)  # noqa: E731


def _sender(name: str) -> str:
    return f"{name} <{PEOPLE[name]}>"


def _filler(rng: random.Random):
    day = rng.choice([(10, 28), (10, 29), (10, 30), (10, 31), (11, 1)])
    when = D(day[0], day[1], rng.randint(7, 18), rng.randint(0, 59))
    sender = _sender(rng.choice(sorted(PEOPLE)))
    project = rng.choice(PROJECTS)
    kind = rng.choice(
        ["status", "meeting", "question", "timesheet", "newsletter", "promo", "receipt", "personal", "fyi"]
    )
    if kind == "status":
        subj = f"{project} weekly status"
        body = f"Quick status on {project}: {rng.choice(DETAILS)}. No blockers this week. Next sync is {rng.choice(DAYS)}."
    elif kind == "meeting":
        t = rng.choice(TIMES)
        subj = f"Invite: {project} sync {rng.choice(DAYS)} {t}"
        body = f"Setting up a meeting to go over {project}. Agenda: scope, risks, next steps. Let me know if {t} doesn't work."
    elif kind == "question":
        subj = f"Question on {project} requirements"
        body = f"Can you clarify whether {project} needs SSO in the first release? I'd like to lock the requirements this week."
    elif kind == "timesheet":
        subj = "Timesheet reminder"
        body = f"Reminder: please submit your timesheet by {rng.choice(DAYS)}. The deadline is firm because payroll runs early."
    elif kind == "newsletter":
        sender = "Weekly Data Digest <news@datadigest.example>"
        subj = f"The Weekly Data Digest #{rng.randint(100, 200)}"
        body = f"This week: {rng.choice(NEWS_TOPICS)}. You are receiving this newsletter because you subscribed. Unsubscribe any time."
    elif kind == "promo":
        sender = "DeskDeals <offers@deskdeals.example>"
        pct, item = rng.choice([20, 30, 40]), rng.choice(ITEMS)
        subj = f"{pct}% off {item} this weekend"
        body = f"Sale ends Sunday! Save {pct}% off {item}. Shop now."
    elif kind == "receipt":
        shop = rng.choice(SHOPS)
        sender = f"{shop} <billing@{shop.split()[0].lower()}.example>"
        subj = f"Your receipt from {shop}"
        body = f"Thanks for your order. Receipt #{rng.randint(10000, 99999)}. Total ${rng.randint(8, 140)}.{rng.randint(10, 99)}."
    elif kind == "personal":
        d = rng.choice(DAYS)
        subj = f"Lunch {d}?"
        body = f"Are you free for lunch {d}? Thinking about that new ramen place."
    else:
        subj = "Office closed Monday"
        body = "FYI the office will be closed Monday for the holiday. Enjoy the long weekend."
    return (when, sender, subj, body)


def _planted():
    return [
        # --- multi-hop chain + distractors ---
        (D(10, 30, 9, 14), _sender("Dana Whitfield"), "Project Falcon launch slips two weeks",
         "Heads up: the Project Falcon launch is delayed. Our vendor Kestrel Logistics missed the delivery of the "
         "sensor modules and has not given a new ship date. I'm escalating internally and will update the team Monday."),
        (D(10, 31, 16, 40), _sender("Marcus Lee"), "Re: Kestrel Logistics contract amendment",
         "Following the escalation, I approved the amendment to the Kestrel Logistics contract this afternoon. "
         "It waives the $40k late-delivery penalty in exchange for a firm ship date of Nov 20. Legal has the signed copy."),
        (D(10, 29, 11, 5), _sender("Sofia Alvarez"), "Orion Freight contract amendment approved",
         "I approved the amendment to the Orion Freight contract for Project Heron. Terms: 5% volume discount, "
         "net-45 payment. Procurement will file it."),
        (D(10, 30, 15, 22), _sender("Tom Becker"), "Project Heron slipping too",
         "Project Heron is delayed as well; our vendor Orion Freight is backed up at the port and can't commit to a date."),
        # --- today's inbox (2026-11-01) ---
        (D(11, 1, 8, 2), _sender("Priya Raman"), "URGENT: billing API outage, need rollback approval",
         "The billing API has been down since 07:40. We need your approval to roll back release 4.12 by 3pm today. "
         "Please reply ASAP."),
        (D(11, 1, 8, 30), _sender("Marcus Lee"), "Q4 budget approval needed",
         "Please review and approve the attached Q4 budget by EOD Monday. Finance closes the books Tuesday."),
        (D(11, 1, 9, 0), _sender("Dana Whitfield"), "Falcon launch war room Monday 10am",
         "Calling a meeting for Monday at 10am to re-plan the Falcon launch. Agenda: new ship date, comms, owners."),
        (D(11, 1, 9, 40), _sender("Hannah Cole"), "Lunch Tuesday?",
         "Are you free for lunch Tuesday? Thinking about that new ramen place."),
    ]


def generate_archive(seed: int = 7, n_filler: int = 50) -> list[Email]:
    rng = random.Random(seed)
    raw = [_filler(rng) for _ in range(n_filler)] + _planted()
    raw.sort(key=lambda r: (r[0], r[2]))
    return [
        Email(id=f"e{i:04d}", sender=s, to=ME, subject=subj, body=body, date=when)
        for i, (when, s, subj, body) in enumerate(raw, 1)
    ]
