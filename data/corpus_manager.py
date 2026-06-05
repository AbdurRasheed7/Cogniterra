"""
data/corpus_manager.py
======================
Upgrade 3 -- Paper Corpus Manager for Cogniterra
-------------------------------------------------
Loads and queries the paper corpus defined in data/corpus.json.

Provides:
  - List all papers or filter by domain/tag
  - Look up a single paper's metadata by arXiv ID
  - CLI interface for exploring the corpus without running the full pipeline

Public API:
  get_paper(paper_id: str) -> dict | None
  list_papers(domain: str | None, tag: str | None) -> list[dict]
  get_all_ids(domain: str | None) -> list[str]
  print_corpus_table(domain: str | None)

CLI usage:
  python data/corpus_manager.py                        # show all 20 papers
  python data/corpus_manager.py --domain rl            # show RL papers only
  python data/corpus_manager.py --domain ml_systems    # show ML systems papers
  python data/corpus_manager.py --tag pruning          # filter by tag
  python data/corpus_manager.py --id 1707.06347        # show one paper detail
  python data/corpus_manager.py --ids-only             # print IDs only (for scripting)
  python data/corpus_manager.py --ids-only --domain rl # RL IDs only
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

# ---------------------------------------------------------------------------
# Corpus file location
# ---------------------------------------------------------------------------

_CORPUS_PATH = os.path.join(os.path.dirname(__file__), "corpus.json")

# ---------------------------------------------------------------------------
# Internal cache
# ---------------------------------------------------------------------------

_corpus_cache: Optional[dict] = None


def _load_corpus() -> dict:
    """Load and cache corpus.json. Raises FileNotFoundError if missing."""
    global _corpus_cache
    if _corpus_cache is not None:
        return _corpus_cache
    if not os.path.exists(_CORPUS_PATH):
        raise FileNotFoundError(
            f"Corpus file not found: {_CORPUS_PATH}\n"
            "Make sure data/corpus.json exists in your project root."
        )
    with open(_CORPUS_PATH, "r", encoding="utf-8") as f:
        _corpus_cache = json.load(f)
    return _corpus_cache


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_paper(paper_id: str) -> Optional[dict]:
    """
    Look up a single paper by arXiv ID.

    Args:
        paper_id: arXiv ID, e.g. "1512.03385"

    Returns:
        Paper dict or None if not in corpus.
    """
    corpus = _load_corpus()
    for paper in corpus["papers"]:
        if paper["id"] == paper_id:
            return paper
    return None


def list_papers(
    domain: Optional[str] = None,
    tag:    Optional[str] = None,
) -> list[dict]:
    """
    Return papers filtered by domain and/or tag.

    Args:
        domain: e.g. "rl", "ml", "nlp", "graph", "recommendation", "ml_systems"
                Pass None to return all domains.
        tag:    e.g. "pruning", "ppo", "residual". Pass None to skip tag filter.

    Returns:
        List of paper dicts matching all supplied filters.
    """
    corpus = _load_corpus()
    papers = corpus["papers"]

    if domain is not None:
        papers = [p for p in papers if p["domain"] == domain]

    if tag is not None:
        papers = [p for p in papers if tag in p.get("tags", [])]

    return papers


def get_all_ids(domain: Optional[str] = None) -> list[str]:
    """
    Return just the arXiv IDs, optionally filtered by domain.

    Useful for scripting batch runs:
      ids = get_all_ids("rl")
      for id_ in ids:
          subprocess.run(["python", "pipeline.py", "--paper", id_])
    """
    return [p["id"] for p in list_papers(domain=domain)]


def get_domains() -> list[str]:
    """Return the sorted list of unique domains in the corpus."""
    corpus = _load_corpus()
    return sorted({p["domain"] for p in corpus["papers"]})


def get_tags() -> list[str]:
    """Return the sorted list of unique tags across all papers."""
    corpus = _load_corpus()
    tags: set[str] = set()
    for paper in corpus["papers"]:
        tags.update(paper.get("tags", []))
    return sorted(tags)


def corpus_summary() -> dict:
    """
    Return a summary dict: total papers, count per domain, count per tag.
    """
    corpus = _load_corpus()
    papers = corpus["papers"]

    domain_counts: dict[str, int] = {}
    tag_counts:    dict[str, int] = {}

    for p in papers:
        d = p["domain"]
        domain_counts[d] = domain_counts.get(d, 0) + 1
        for t in p.get("tags", []):
            tag_counts[t] = tag_counts.get(t, 0) + 1

    return {
        "total":         len(papers),
        "by_domain":     domain_counts,
        "top_tags":      dict(sorted(tag_counts.items(), key=lambda x: -x[1])[:10]),
    }


# ---------------------------------------------------------------------------
# CLI display helpers
# ---------------------------------------------------------------------------

_DOMAIN_EMOJI = {
    "ml":             "🖼️ ",
    "rl":             "🤖",
    "nlp":            "📝",
    "graph":          "🕸️ ",
    "recommendation": "🎯",
    "ml_systems":     "⚙️ ",
}

_COL_WIDTHS = {
    "id":     14,
    "domain": 14,
    "title":  48,
    "model":  22,
    "tags":   30,
    "acc":    10,
}


def print_corpus_table(
    domain: Optional[str] = None,
    tag:    Optional[str] = None,
) -> None:
    """Pretty-print the corpus as a table, optionally filtered."""
    papers = list_papers(domain=domain, tag=tag)

    if not papers:
        print(f"No papers found for domain={domain!r} tag={tag!r}")
        return

    # Header
    header = (
        f"{'ID':<14} {'Domain':<14} {'Title':<48} "
        f"{'Model':<22} {'Expected Acc':>12}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    # Rows
    for p in papers:
        emoji  = _DOMAIN_EMOJI.get(p["domain"], "  ")
        domain_ = f"{emoji} {p['domain']}"
        title  = p["title"][:46] + ".." if len(p["title"]) > 48 else p["title"]
        model  = p.get("model", "")[:20]
        acc    = p.get("expected_accuracy")
        acc_str = f"{acc:.2f}%" if acc is not None else "N/A"
        upgrade = " ★" if "Upgrade 3" in p.get("notes", "") else "  "

        print(
            f"{p['id']:<14} {domain_:<20} {title:<48} "
            f"{model:<22} {acc_str:>12}{upgrade}"
        )

    print("=" * len(header))
    print(f"Total: {len(papers)} papers  (★ = new in Upgrade 3)")

    # Summary by domain if showing all
    if domain is None and tag is None:
        summary = corpus_summary()
        print("\nBy domain:")
        for d, count in sorted(summary["by_domain"].items()):
            emoji = _DOMAIN_EMOJI.get(d, "  ")
            print(f"  {emoji} {d:<16} {count} papers")


def print_paper_detail(paper_id: str) -> None:
    """Pretty-print all fields for a single paper."""
    paper = get_paper(paper_id)
    if paper is None:
        print(f"Paper {paper_id!r} not found in corpus.")
        return

    emoji = _DOMAIN_EMOJI.get(paper["domain"], "")
    print(f"\n{'='*60}")
    print(f"  {emoji}  {paper['title']}")
    print(f"{'='*60}")
    print(f"  arXiv ID  : {paper['id']}")
    print(f"  Domain    : {paper['domain']}")
    print(f"  Authors   : {', '.join(paper['authors'])}")
    print(f"  Year      : {paper['year']}")
    print(f"  Model     : {paper.get('model', 'N/A')}")
    print(f"  Dataset   : {paper.get('dataset', 'N/A')}")
    acc = paper.get("expected_accuracy")
    print(f"  Expected  : {f'{acc:.2f}%' if acc is not None else 'N/A'}")
    print(f"  Tags      : {', '.join(paper.get('tags', []))}")
    print(f"  Notes     : {paper.get('notes', '')}")
    print(f"{'='*60}")
    print(f"\n  Run with:")
    print(f"    python pipeline.py --paper {paper['id']}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cogniterra corpus manager — browse and query the paper registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data/corpus_manager.py                        show all 20 papers
  python data/corpus_manager.py --domain rl            show RL papers
  python data/corpus_manager.py --domain ml_systems    show ML systems papers
  python data/corpus_manager.py --tag pruning          filter by tag
  python data/corpus_manager.py --id 1707.06347        show one paper
  python data/corpus_manager.py --ids-only             print IDs for scripting
  python data/corpus_manager.py --ids-only --domain rl RL paper IDs only
  python data/corpus_manager.py --summary              domain/tag summary
        """,
    )
    p.add_argument("--domain",   default=None, help="Filter by domain")
    p.add_argument("--tag",      default=None, help="Filter by tag")
    p.add_argument("--id",       default=None, help="Show detail for one paper ID")
    p.add_argument("--ids-only", action="store_true",
                   help="Print IDs only (useful for scripting batch runs)")
    p.add_argument("--summary",  action="store_true",
                   help="Print corpus summary by domain")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.id:
        print_paper_detail(args.id)
        return

    if args.summary:
        s = corpus_summary()
        print(f"\nCorpus summary: {s['total']} papers total")
        print("\nBy domain:")
        for d, count in sorted(s["by_domain"].items()):
            emoji = _DOMAIN_EMOJI.get(d, "  ")
            print(f"  {emoji} {d:<18} {count} papers")
        print("\nTop tags:")
        for tag, count in s["top_tags"].items():
            print(f"  {tag:<24} {count}")
        return

    if args.ids_only:
        ids = get_all_ids(domain=args.domain)
        for id_ in ids:
            print(id_)
        return

    print_corpus_table(domain=args.domain, tag=args.tag)


if __name__ == "__main__":
    main()