"""Offline stand-in for the LLM so the whole pipeline runs without an API key.

This is NOT a model. It is a small rule-based script that speaks the same prompt/JSON
contracts as the real calls, so the plumbing (batching, validation, retrieval loop,
citations, fallbacks) can be exercised and tested deterministically. Quality numbers
only mean something with a real model.
"""
from __future__ import annotations

import json
import re

from .digest import render_digest_markdown

_EMAIL_RE = re.compile(r'<email id="([^"]+)">\s*From: (.*?)\nDate: (.*?)\nSubject: (.*?)\n\n(.*?)</email>', re.DOTALL)
_DEADLINE_RE = re.compile(
    r"\bby (\d{1,2}(?::\d{2})?\s?(?:am|pm)|EOD(?: \w+)?|(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day)", re.I
)

_NOISE = ("unsubscribe", "% off", "receipt", "newsletter")
_CRITICAL = ("urgent", "outage", "asap")
_IMPORTANT = ("deadline", "approval needed", "need your approval", "sign-off", "action required")
_NORMAL = ("meeting", "invite", "agenda", "question", "reminder", "review")


def _proper_nouns(text: str) -> list[str]:
    return re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b", text)


def _bridge(anchors: list[str], emails: list[tuple]) -> tuple[str, str] | None:
    """Find (email_id, vendor) from an email that mentions every anchor phrase."""
    for eid, _frm, _dt, subj, body in emails:
        text = f"{subj} {body}"
        if anchors and all(a in text for a in anchors):
            m = re.search(r"vendor,? ([A-Z][a-z]+ [A-Z][a-z]+)", text)
            if m:
                return eid, m.group(1)
    return None


class MockLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []  # task name per call, for tests

    def complete(self, system: str, user: str, *, max_tokens: int = 2000) -> str:
        task = re.match(r"TASK=(\w+)", system).group(1)  # type: ignore[union-attr]
        self.calls.append(task)
        return getattr(self, f"_{task}")(user)

    # ---- prioritize -----------------------------------------------------
    def _prioritize(self, user: str) -> str:
        results = []
        for eid, _frm, _dt, subj, body in _EMAIL_RE.findall(user):
            text = f"{subj} {body}".lower()
            m = _DEADLINE_RE.search(body)
            if any(w in text for w in _NOISE):
                pr = 1
                cat = ("promotion" if "% off" in text else "receipt" if "receipt" in text else "newsletter")
            elif any(w in text for w in _CRITICAL):
                pr, cat = 5, "action_required"
            elif any(w in text for w in _IMPORTANT):
                pr, cat = 4, "action_required"
            elif any(w in text for w in _NORMAL):
                pr = 3
                cat = "meeting" if any(w in text for w in ("meeting", "invite", "agenda")) else "other"
            elif "lunch" in text:
                pr, cat = 2, "personal"
            else:
                pr, cat = 2, "fyi"
            results.append({
                "id": eid, "priority": pr, "category": cat,
                "needs_reply": pr >= 4 or "?" in body,
                "deadline": m.group(1) if m else None,
                "reason": f"keyword heuristic ({cat})",
            })
        return json.dumps({"results": results})

    # ---- digest ---------------------------------------------------------
    def _digest(self, user: str) -> str:
        payload = json.loads(re.search(r"<data>(.*?)</data>", user, re.DOTALL).group(1))  # type: ignore[union-attr]
        return render_digest_markdown(payload)

    # ---- multi-hop RAG --------------------------------------------------
    def _rag_plan(self, user: str) -> str:
        q = re.search(r"<question>(.*?)</question>", user, re.DOTALL).group(1).strip()  # type: ignore[union-attr]
        prev_block = re.search(r"<previous_queries>(.*?)</previous_queries>", user, re.DOTALL).group(1)  # type: ignore[union-attr]
        prev = re.findall(r"^- (.*)$", prev_block, re.M)
        emails = _EMAIL_RE.findall(user)
        anchors = _proper_nouns(q)

        if not emails:  # hop 1: search only for the anchor entity, not the whole question
            extra = [w for w in ("vendor", "delay") if w in q.lower()]
            query = " ".join(anchors + extra) or q
            return json.dumps({"reasoning": "Start from the entities named in the question.",
                               "sufficient": False, "next_query": query})

        bridge = _bridge(anchors, emails)
        if bridge:
            _eid, vendor = bridge
            answered = any("approved" in body and vendor in f"{subj} {body}" for _i, _f, _d, subj, body in emails)
            nxt = f"{vendor} contract amendment approved"
            if not answered and nxt not in prev:
                return json.dumps({"reasoning": f"Evidence names vendor {vendor}; now look for who approved its amendment.",
                                   "sufficient": False, "next_query": nxt})
        return json.dumps({"reasoning": "Evidence appears sufficient.", "sufficient": True, "next_query": ""})

    def _rag_answer(self, user: str) -> str:
        q = re.search(r"<question>(.*?)</question>", user, re.DOTALL).group(1)  # type: ignore[union-attr]
        emails = _EMAIL_RE.findall(user)
        bridge = _bridge(_proper_nouns(q), emails)
        if bridge:
            src_id, vendor = bridge
            for eid, frm, _dt, subj, body in emails:
                if "approved" in body and vendor in f"{subj} {body}":
                    name = frm.split("<")[0].strip()
                    return (f"{name} approved the contract amendment for {vendor} [{eid}], "
                            f"the vendor named as the cause of the delay in [{src_id}].")
        return "The evidence does not contain enough information to answer this question."
