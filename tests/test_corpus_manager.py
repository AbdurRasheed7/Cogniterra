"""
tests/test_corpus_manager.py
============================
Unit tests for data/corpus_manager.py (Upgrade 3).

Run with:
  python -m pytest tests/test_corpus_manager.py -v

All tests use a minimal in-memory corpus fixture so no file I/O
or network calls are needed.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.corpus_manager as cm


# ---------------------------------------------------------------------------
# Fixture corpus — minimal but covers all domains
# ---------------------------------------------------------------------------

_FIXTURE_CORPUS = {
    "_meta": {"version": "3.0", "total": 6},
    "papers": [
        {
            "id": "1512.03385",
            "title": "Deep Residual Learning",
            "authors": ["He"],
            "year": 2015,
            "domain": "ml",
            "tags": ["cv", "residual"],
            "model": "ResNet",
            "dataset": "CIFAR-10",
            "expected_accuracy": 99.1,
            "notes": "Existing paper.",
        },
        {
            "id": "1707.06347",
            "title": "Proximal Policy Optimization",
            "authors": ["Schulman"],
            "year": 2017,
            "domain": "rl",
            "tags": ["rl", "ppo", "policy_gradient"],
            "model": "PPO",
            "dataset": "CartPole-v1",
            "expected_accuracy": None,
            "notes": "New in Upgrade 3.",
        },
        {
            "id": "1801.01290",
            "title": "Soft Actor-Critic",
            "authors": ["Haarnoja"],
            "year": 2018,
            "domain": "rl",
            "tags": ["rl", "sac", "off_policy"],
            "model": "SAC",
            "dataset": "CartPole-v1",
            "expected_accuracy": None,
            "notes": "New in Upgrade 3.",
        },
        {
            "id": "1706.03762",
            "title": "Attention Is All You Need",
            "authors": ["Vaswani"],
            "year": 2017,
            "domain": "nlp",
            "tags": ["transformer", "attention"],
            "model": "Transformer",
            "dataset": "20Newsgroups",
            "expected_accuracy": 64.75,
            "notes": "Existing paper.",
        },
        {
            "id": "1803.03635",
            "title": "The Lottery Ticket Hypothesis",
            "authors": ["Frankle"],
            "year": 2018,
            "domain": "ml_systems",
            "tags": ["pruning", "sparsity", "lottery_ticket"],
            "model": "Pruned MLP",
            "dataset": "MNIST",
            "expected_accuracy": None,
            "notes": "New in Upgrade 3.",
        },
        {
            "id": "1609.02907",
            "title": "Semi-Supervised Classification with GCN",
            "authors": ["Kipf"],
            "year": 2016,
            "domain": "graph",
            "tags": ["gnn", "gcn"],
            "model": "GCN",
            "dataset": "Synthetic",
            "expected_accuracy": 60.0,
            "notes": "Existing paper.",
        },
    ],
}


def _patch_corpus(func):
    """Decorator: replace _corpus_cache with fixture for the duration of the test."""
    import functools
    @functools.wraps(func)   # preserves __name__ so pytest can discover test_* methods
    def wrapper(self, *args, **kwargs):
        orig = cm._corpus_cache
        cm._corpus_cache = _FIXTURE_CORPUS
        try:
            return func(self, *args, **kwargs)
        finally:
            cm._corpus_cache = orig
    return wrapper


# ===========================================================================
# Test Group 1: get_paper()
# ===========================================================================


class TestGetPaper(unittest.TestCase):

    @_patch_corpus
    def test_returns_correct_paper(self):
        paper = cm.get_paper("1512.03385")
        self.assertIsNotNone(paper)
        self.assertEqual(paper["title"], "Deep Residual Learning")

    @_patch_corpus
    def test_returns_none_for_unknown_id(self):
        self.assertIsNone(cm.get_paper("9999.99999"))

    @_patch_corpus
    def test_returns_none_for_empty_string(self):
        self.assertIsNone(cm.get_paper(""))

    @_patch_corpus
    def test_returned_dict_has_required_keys(self):
        paper = cm.get_paper("1512.03385")
        for key in ("id", "title", "domain", "tags", "model", "dataset"):
            self.assertIn(key, paper)


# ===========================================================================
# Test Group 2: list_papers()
# ===========================================================================


class TestListPapers(unittest.TestCase):

    @_patch_corpus
    def test_no_filter_returns_all(self):
        papers = cm.list_papers()
        self.assertEqual(len(papers), 6)

    @_patch_corpus
    def test_domain_filter_rl(self):
        papers = cm.list_papers(domain="rl")
        self.assertEqual(len(papers), 2)
        for p in papers:
            self.assertEqual(p["domain"], "rl")

    @_patch_corpus
    def test_domain_filter_ml_systems(self):
        papers = cm.list_papers(domain="ml_systems")
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]["id"], "1803.03635")

    @_patch_corpus
    def test_domain_filter_unknown_returns_empty(self):
        papers = cm.list_papers(domain="nonexistent_domain")
        self.assertEqual(papers, [])

    @_patch_corpus
    def test_tag_filter(self):
        papers = cm.list_papers(tag="pruning")
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]["id"], "1803.03635")

    @_patch_corpus
    def test_domain_and_tag_filter_combined(self):
        # rl domain + ppo tag -> only PPO paper
        papers = cm.list_papers(domain="rl", tag="ppo")
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]["id"], "1707.06347")

    @_patch_corpus
    def test_tag_filter_no_match_returns_empty(self):
        papers = cm.list_papers(tag="nonexistent_tag")
        self.assertEqual(papers, [])


# ===========================================================================
# Test Group 3: get_all_ids()
# ===========================================================================


class TestGetAllIds(unittest.TestCase):

    @_patch_corpus
    def test_returns_all_ids_when_no_domain(self):
        ids = cm.get_all_ids()
        self.assertEqual(len(ids), 6)
        self.assertIn("1512.03385", ids)
        self.assertIn("1707.06347", ids)

    @_patch_corpus
    def test_domain_filter_returns_subset(self):
        ids = cm.get_all_ids(domain="rl")
        self.assertEqual(len(ids), 2)
        self.assertIn("1707.06347", ids)
        self.assertIn("1801.01290", ids)

    @_patch_corpus
    def test_returns_list_of_strings(self):
        ids = cm.get_all_ids()
        for id_ in ids:
            self.assertIsInstance(id_, str)

    @_patch_corpus
    def test_unknown_domain_returns_empty(self):
        self.assertEqual(cm.get_all_ids(domain="unknown"), [])


# ===========================================================================
# Test Group 4: corpus_summary()
# ===========================================================================


class TestCorpusSummary(unittest.TestCase):

    @_patch_corpus
    def test_total_is_correct(self):
        s = cm.corpus_summary()
        self.assertEqual(s["total"], 6)

    @_patch_corpus
    def test_domain_counts_correct(self):
        s = cm.corpus_summary()
        self.assertEqual(s["by_domain"]["rl"], 2)
        self.assertEqual(s["by_domain"]["ml"], 1)
        self.assertEqual(s["by_domain"]["nlp"], 1)
        self.assertEqual(s["by_domain"]["ml_systems"], 1)
        self.assertEqual(s["by_domain"]["graph"], 1)

    @_patch_corpus
    def test_top_tags_present(self):
        s = cm.corpus_summary()
        self.assertIn("top_tags", s)
        self.assertIsInstance(s["top_tags"], dict)

    @_patch_corpus
    def test_rl_tag_appears_in_top_tags(self):
        s = cm.corpus_summary()
        # "rl" tag appears in both rl papers
        self.assertIn("rl", s["top_tags"])
        self.assertEqual(s["top_tags"]["rl"], 2)


# ===========================================================================
# Test Group 5: real corpus.json validation
# ===========================================================================


class TestRealCorpusFile(unittest.TestCase):
    """
    Validates the actual corpus.json on disk -- no mocking.
    These tests catch mistakes in the JSON file itself.
    """

    def setUp(self):
        # Reset cache so we load from disk
        cm._corpus_cache = None
        self.corpus_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", "corpus.json"
        )
        if not os.path.exists(self.corpus_path):
            self.skipTest("data/corpus.json not found -- place file first")

    def tearDown(self):
        cm._corpus_cache = None

    def test_corpus_loads_without_error(self):
        corpus = cm._load_corpus()
        self.assertIn("papers", corpus)

    def test_corpus_has_20_papers(self):
        papers = cm.list_papers()
        self.assertEqual(len(papers), 20)

    def test_all_papers_have_required_fields(self):
        required = {"id", "title", "domain", "tags", "model", "dataset", "notes"}
        for paper in cm.list_papers():
            missing = required - set(paper.keys())
            self.assertEqual(
                missing, set(),
                f"Paper {paper.get('id')} missing fields: {missing}"
            )

    def test_no_duplicate_ids(self):
        ids = cm.get_all_ids()
        self.assertEqual(len(ids), len(set(ids)), "Duplicate paper IDs found")

    def test_all_domains_are_valid(self):
        valid = {"ml", "rl", "nlp", "graph", "recommendation", "ml_systems"}
        for paper in cm.list_papers():
            self.assertIn(
                paper["domain"], valid,
                f"Paper {paper['id']} has invalid domain: {paper['domain']}"
            )

    def test_rl_papers_count(self):
        rl = cm.list_papers(domain="rl")
        self.assertGreaterEqual(len(rl), 5, "Expected at least 5 RL papers")

    def test_ml_systems_papers_count(self):
        sys_papers = cm.list_papers(domain="ml_systems")
        self.assertGreaterEqual(len(sys_papers), 4,
                                "Expected at least 4 ML systems papers")

    def test_existing_9_papers_all_present(self):
        existing = [
            "1512.03385", "1409.1556",  "1506.02078", "1505.04597",
            "1202.2745",  "1609.02907", "1509.02971", "1706.03762",
            "1708.05031",
        ]
        ids = cm.get_all_ids()
        for paper_id in existing:
            self.assertIn(
                paper_id, ids,
                f"Existing paper {paper_id} missing from corpus"
            )

    def test_new_rl_papers_present(self):
        new_rl = ["1707.06347", "1801.01290", "1509.06461", "1802.09477", "2004.14990"]
        ids = cm.get_all_ids()
        for paper_id in new_rl:
            self.assertIn(paper_id, ids, f"New RL paper {paper_id} missing")

    def test_tags_are_lists(self):
        for paper in cm.list_papers():
            self.assertIsInstance(
                paper.get("tags"), list,
                f"Paper {paper['id']} tags must be a list"
            )

    def test_get_domains_returns_all_domains(self):
        domains = cm.get_domains()
        self.assertIn("rl", domains)
        self.assertIn("ml", domains)
        self.assertIn("ml_systems", domains)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)