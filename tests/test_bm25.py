import math

from emailpipe.bm25 import BM25Index, EmailSearch, stem, tokenize
from emailpipe.data import generate_archive


def test_tokenize_lowercases_drops_stopwords_and_stems():
    assert tokenize("The Vendors APPROVED the contracts!") == ["vendor", "approv", "contract"]
    assert stem("approve") == stem("approved") == stem("approves") == "approv"
    assert stem("ss") == "ss" and stem("access") == "access"  # no over-stripping


def test_scores_match_hand_computed_formula():
    docs = {"d1": "apple banana apple", "d2": "banana cherry", "d3": "cherry cherry cherry date"}
    idx = BM25Index(k1=1.5, b=0.75).build(docs.items())
    N, avgdl = 3, (3 + 2 + 4) / 3
    n_apple = 1
    idf = math.log(1 + (N - n_apple + 0.5) / (n_apple + 0.5))
    tf, dl = 2, 3
    expected = idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl))
    hits = idx.search("apple")
    assert [h.doc_id for h in hits] == ["d1"]
    assert math.isclose(hits[0].score, expected, rel_tol=1e-12)


def test_rare_terms_outweigh_common_terms_and_idf_is_nonnegative():
    idx = BM25Index().build([("a", "common rare"), ("b", "common"), ("c", "common"), ("d", "common")])
    assert idx.idf("rare") > idx.idf("common") > 0
    assert idx.search("common rare", 1)[0].doc_id == "a"


def test_length_normalisation_prefers_shorter_doc():
    idx = BM25Index().build([("short", "falcon"), ("long", "falcon " + "filler " * 40)])
    assert [h.doc_id for h in idx.search("falcon")] == ["short", "long"]


def test_edge_cases():
    idx = BM25Index()
    assert idx.search("anything") == []  # empty index
    idx.add("a", "hello world")
    assert idx.search("zzz") == [] and idx.search("") == [] and idx.search("the of") == []
    assert len(idx.search("hello", k=10)) == 1


def test_ties_are_deterministic_by_insertion_order():
    idx = BM25Index().build([("x", "same words"), ("y", "same words")])
    assert [h.doc_id for h in idx.search("same words")] == ["x", "y"]


def test_save_load_roundtrip(tmp_path):
    idx = BM25Index(1.2, 0.5).build([("a", "alpha beta"), ("b", "beta gamma gamma")])
    idx.save(tmp_path / "i.json")
    loaded = BM25Index.load(tmp_path / "i.json")
    assert loaded.search("gamma beta") == idx.search("gamma beta")


def test_email_search_finds_planted_email():
    es = EmailSearch(generate_archive())
    top = es.search("Kestrel Logistics", 1)[0][0]
    assert "Kestrel" in top.body
