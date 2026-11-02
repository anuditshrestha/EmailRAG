"""Core data model, archive I/O, and the prompt-safe email renderer."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class Email:
    id: str
    sender: str  # "Name <addr>"
    to: str
    subject: str
    body: str
    date: datetime
    thread_id: Optional[str] = None

    @property
    def sender_name(self) -> str:
        name = self.sender.split("<")[0].strip()
        return name or self.sender

    def to_dict(self) -> dict:
        d = asdict(self)
        d["date"] = self.date.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Email":
        d = dict(d)
        d["date"] = datetime.fromisoformat(d["date"])
        return cls(**d)


def save_archive(emails: Iterable[Email], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in emails:
            f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")


def load_archive(path: str | Path) -> list[Email]:
    out: list[Email] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Email.from_dict(json.loads(line)))
    return out


def render_email(e: Email, body_chars: int = 600) -> str:
    """Render an email as a delimited block for an LLM prompt.

    Email text is untrusted: the closing tag is neutralised so a message
    cannot break out of its block, and bodies are truncated to control cost.
    """
    body = e.body.strip()
    if len(body) > body_chars:
        body = body[:body_chars].rstrip() + " …"
    body = body.replace("</email>", "</ email>")
    subject = e.subject.replace("\n", " ")
    return (
        f'<email id="{e.id}">\n'
        f"From: {e.sender}\n"
        f"Date: {e.date:%Y-%m-%d %H:%M}\n"
        f"Subject: {subject}\n\n"
        f"{body}\n"
        f"</email>"
    )
