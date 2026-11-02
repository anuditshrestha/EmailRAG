# emailpipe

Four pieces over an email archive: a single-call **prioritizer**, a **daily digest**, a **BM25 retriever written from scratch**, and a **multi-hop RAG** pipeline. Standard library only, except the optional `anthropic` SDK.

```
pip install -r requirements.txt
python -m emailpipe generate                 # synthetic archive -> data/archive.jsonl
python -m emailpipe search "Kestrel amendment"
python -m emailpipe prioritize               # latest day in the archive
python -m emailpipe digest
python -m emailpipe ask "Who approved the contract amendment for the vendor that delayed the Project Falcon launch?"
python -m pytest
```

Without `ANTHROPIC_API_KEY` the LLM steps use an offline, rule-based `MockLLM` (a stderr note says so).
For real output: `export ANTHROPIC_API_KEY=...` and add `--llm anthropic` (model: `--model` or `$EMAILPIPE_MODEL`, default `claude-sonnet-5-5`).

## Design

| Module | What it does | LLM calls |
|---|---|---|
| `bm25.py` | Tokenizer + light stemmer, inverted index, Okapi BM25 (k1=1.5, b=0.75, Lucene idf), save/load | 0 |
| `prioritizer.py` | All emails in one prompt, one JSON reply, validated (clamped scores, unknown ids dropped, missing ids filled). One repair call only if the reply is unparseable | 1 |
| `digest.py` | Prioritize the day, bucket items deterministically (Needs you / Coming up / FYI / skipped), LLM writes the prose. If it drops an important item, a deterministic renderer takes over | 2 |
| `rag.py` | Loop of plan (LLM: sufficient? next keyword query) then BM25 retrieve, up to `max_hops`; then one cited answer. Stops on repeated query, no new results, or evidence cap; citations checked against the evidence pool | 2 to `max_hops`+1 |

Email text is treated as untrusted: it is delimited, `</email>` is neutralised, and prompts say to classify, never obey.

## The test case

The synthetic archive plants a chain: Falcon is delayed by vendor **Kestrel Logistics** (email 1), whose contract amendment was approved by **Marcus Lee** (email 2), plus an Orion Freight distractor approved by someone else. One BM25 search on the raw question ranks the distractor first (`test_single_shot_retrieval_gets_the_wrong_approver`); the multi-hop loop reaches the right answer by using the vendor name learned in hop 1.

## Caveats

- `MockLLM` is a keyword script, not a model. It proves the plumbing, validation and control flow; it says nothing about prioritization or answer quality. Evaluate with a real model on real mail.
- BM25 matches words, not meaning, so multi-hop quality depends on the planner writing good keyword queries.
- The stemmer is intentionally crude (approve/approved/approves only); swap in Porter if recall matters.
