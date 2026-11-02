"""Command line: python -m emailpipe <command> [options]"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from .bm25 import EmailSearch
from .data import generate_archive
from .digest import generate_digest
from .llm import make_llm
from .models import load_archive, save_archive
from .prioritizer import prioritize
from .rag import MultiHopRAG

DEFAULT_ARCHIVE = Path("data/archive.jsonl")


def _load(args) -> list:
    if not Path(args.archive).exists():
        sys.exit(f"archive not found: {args.archive} (run `python -m emailpipe generate` first)")
    return load_archive(args.archive)


def _llm(args):
    llm = make_llm(args.llm, args.model)
    if type(llm).__name__ == "MockLLM":
        print("[using offline MockLLM: set ANTHROPIC_API_KEY and --llm anthropic for real model output]", file=sys.stderr)
    return llm


def _day(args, emails) -> date:
    return date.fromisoformat(args.day) if args.day else max(e.date for e in emails).date()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="emailpipe", description=__doc__)
    p.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    p.add_argument("--llm", choices=["auto", "anthropic", "mock"], default="auto")
    p.add_argument("--model", default=None, help="Anthropic model id (default: $EMAILPIPE_MODEL or claude-sonnet-5-5)")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write a synthetic archive")
    g.add_argument("--seed", type=int, default=7)

    s = sub.add_parser("search", help="BM25 search")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)

    pr = sub.add_parser("prioritize", help="single-call triage of one day's mail")
    pr.add_argument("--day", help="YYYY-MM-DD (default: latest day in archive)")

    d = sub.add_parser("digest", help="daily digest for one day")
    d.add_argument("--day", help="YYYY-MM-DD (default: latest day in archive)")

    a = sub.add_parser("ask", help="multi-hop question answering")
    a.add_argument("question")
    a.add_argument("--max-hops", type=int, default=4)
    a.add_argument("-k", type=int, default=4, help="emails retrieved per hop")

    args = p.parse_args(argv)

    if args.cmd == "generate":
        emails = generate_archive(args.seed)
        save_archive(emails, args.archive)
        print(f"wrote {len(emails)} emails to {args.archive}")
        return

    emails = _load(args)

    if args.cmd == "search":
        for e, score in EmailSearch(emails).search(args.query, args.k):
            print(f"{score:7.3f}  {e.id}  {e.date:%m-%d}  {e.sender_name}: {e.subject}")
    elif args.cmd == "prioritize":
        day = _day(args, emails)
        todays = [e for e in emails if e.date.date() == day]
        by_id = {e.id: e for e in todays}
        for t in prioritize(todays, _llm(args)):
            dl = f"  [by {t.deadline}]" if t.deadline else ""
            print(f"P{t.priority}  {t.email_id}  {t.category:<15} {by_id[t.email_id].subject}{dl}")
    elif args.cmd == "digest":
        print(generate_digest(emails, _llm(args), _day(args, emails)).markdown)
    elif args.cmd == "ask":
        rag = MultiHopRAG(EmailSearch(emails), _llm(args), max_hops=args.max_hops, k=args.k)
        print(rag.answer(args.question).trace())


if __name__ == "__main__":
    main()
