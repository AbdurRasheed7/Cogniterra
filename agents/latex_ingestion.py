"""
agents/latex_ingestion.py
=========================
Upgrade 1 — LaTeX Source Ingestion for Cogniterra
--------------------------------------------------
Replaces PDF/HTML-only parsing with arXiv LaTeX source as primary input.

Pipeline:
  1. Use arxiv-py to fetch paper metadata
  2. Download source tarball from arxiv.org/src/{id}
  3. Extract the main .tex file (heuristic: largest .tex file)
  4. Parse structured sections via regex (methods, experiments,
     implementation, results, hyperparameters)
  5. Extract hyperparameter values directly from LaTeX markup
  6. Fallback to existing pdfplumber / HTML parser if no LaTeX source

Public API (drop-in replacement for parser_agent.parse_paper):
  parse_paper(arxiv_id: str) -> str

Tracked state (module-level, read by pipeline.py after Step 1):
  LAST_SOURCE_TYPE: Literal["latex", "pdf", "html"]  → logged in pipeline
  LAST_PAPER_METADATA: dict                           → paper title, authors, year

Dependencies (add to requirements.txt):
  arxiv>=2.1.0
  pdfplumber>=0.10.0
  requests>=2.31.0
  beautifulsoup4>=4.12.0

Usage in pipeline.py (no interface change needed):
  from agents.latex_ingestion import parse_paper
  # Or keep importing from parser_agent — see monkey-patch note at bottom.
"""

from __future__ import annotations

import io
import logging
import os
import re
import tarfile
import tempfile
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Optional imports — graceful degradation if not installed
# ---------------------------------------------------------------------------

try:
    import arxiv as arxiv_lib          # pip install arxiv
    _ARXIV_AVAILABLE = True
except ImportError:
    _ARXIV_AVAILABLE = False

try:
    import pdfplumber                  # pip install pdfplumber
    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False

try:
    from bs4 import BeautifulSoup      # pip install beautifulsoup4
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Module-level state — read by pipeline.py after Step 1
# ---------------------------------------------------------------------------

LAST_SOURCE_TYPE: str = "unknown"       # "latex" | "pdf" | "html"
LAST_PAPER_METADATA: dict = {}          # title, authors, year, abstract

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sections we want to keep (same spirit as config.KEEP_KEYWORDS but LaTeX-aware)
_KEEP_SECTION_PATTERNS: list[str] = [
    r"method", r"approach", r"architecture", r"model",
    r"experiment", r"implement", r"training", r"setup",
    r"hyperparameter", r"result", r"evaluat", r"benchmark",
    r"dataset", r"preprocess", r"network", r"layer",
    r"loss", r"optim", r"batch", r"epoch",
]

# Sections we want to discard
_STOP_SECTION_PATTERNS: list[str] = [
    r"conclusion", r"discussion", r"related.?work",
    r"reference", r"appendix", r"acknowledgment",
    r"acknowledgement", r"future.?work", r"supplementar",
    r"proof", r"theorem", r"bibliograph", r"broader.?impact",
    r"abstract",     # we keep metadata.abstract separately
]

# LaTeX hyperparameter extraction — value patterns after = or :
# NOTE: [^,\n]*? prevents patterns from crossing comma/newline boundaries,
# which stops one pattern from consuming text that belongs to the next param
# (e.g. "dropout = 0.1, and momentum = 0.9" — without the guard the dropout
# pattern would greedily match past the comma and steal momentum's value).
_HYPERPARAM_PATTERNS: list[tuple[str, str]] = [
    (r"learning[_ ]?rate[^,\n]*?(?:=|:)\s*([\d.e+\-]+)",  "lr"),
    (r"\\alpha\s*(?:=|:)\s*([\d.e+\-]+)",                  "lr"),      # \alpha = 0.001
    (r"batch[_ ]?size[^,\n]*?(?:=|:)\s*(\d+)",             "batch_size"),
    (r"(?:num_?)?epochs?[^,\n]*?(?:=|:)\s*(\d+)",          "epochs"),
    (r"momentum[^,\n]*?(?:=|:)\s*([\d.e+\-]+)",            "momentum"),
    (r"weight[_ ]?decay[^,\n]*?(?:=|:)\s*([\d.e+\-]+)",    "weight_decay"),
    (r"dropout[^,\n]*?(?:=|:)\s*([\d.e+\-]+)",             "dropout"),
    (r"hidden[_ ]?(?:size|dim)[^,\n]*?(?:=|:)\s*(\d+)",    "hidden_size"),
    (r"embed(?:ding)?[_ ]?(?:size|dim)[^,\n]*?(?:=|:)\s*(\d+)", "embedding_dim"),
    (r"num[_ ]?heads?[^,\n]*?(?:=|:)\s*(\d+)",             "num_heads"),
    (r"num[_ ]?layers?[^,\n]*?(?:=|:)\s*(\d+)",            "num_layers"),
    (r"gamma[^,\n]*?(?:=|:)\s*([\d.e+\-]+)",               "gamma"),
    (r"epsilon[^,\n]*?(?:=|:)\s*([\d.e+\-]+)",             "epsilon"),
]

# arXiv source tarball URL template
_ARXIV_SRC_URL = "https://arxiv.org/src/{arxiv_id}"

# Request timeout for source downloads (large tarballs can be slow)
_DOWNLOAD_TIMEOUT_SEC = 60

# Minimum character length to consider a .tex file as the "main" file
_MIN_TEX_CHARS = 500

# ---------------------------------------------------------------------------
# Public entry point — same signature as original parse_paper()
# ---------------------------------------------------------------------------


def parse_paper(arxiv_id: str) -> str:
    """
    Primary entry point.  Drop-in replacement for parser_agent.parse_paper().

    Attempts LaTeX ingestion first; falls back to PDF / HTML if unavailable.
    Sets module-level LAST_SOURCE_TYPE and LAST_PAPER_METADATA as side effects.

    Args:
        arxiv_id: arXiv paper identifier, e.g. "1512.03385" or "2301.07608"

    Returns:
        Filtered paper text as a single string.  Empty string on total failure.
    """
    global LAST_SOURCE_TYPE, LAST_PAPER_METADATA

    logger.info("=== parse_paper(%s) — Upgrade 1: LaTeX Ingestion ===", arxiv_id)

    # ── Step 1: Fetch metadata via arxiv-py ──────────────────────────────────
    metadata = _fetch_metadata(arxiv_id)
    LAST_PAPER_METADATA = metadata
    if metadata:
        logger.info(
            "Metadata: title=%r | year=%s | authors=%s",
            metadata.get("title", ""),
            metadata.get("year", ""),
            ", ".join(metadata.get("authors", [])[:3]),
        )

    # ── Step 2: Try LaTeX source ──────────────────────────────────────────────
    latex_text = _try_latex_ingestion(arxiv_id)
    if latex_text:
        LAST_SOURCE_TYPE = "latex"
        logger.info("Source type: latex  (%d chars after filtering)", len(latex_text))
        return latex_text

    # ── Step 3: Fallback — pdfplumber ────────────────────────────────────────
    if _PDF_AVAILABLE:
        pdf_text = _try_pdf_fallback(arxiv_id)
        if pdf_text:
            LAST_SOURCE_TYPE = "pdf"
            logger.info("Source type: pdf  (%d chars)", len(pdf_text))
            return pdf_text

    # ── Step 4: Fallback — HTML (original parser_agent logic) ────────────────
    html_text = _try_html_fallback(arxiv_id)
    if html_text:
        LAST_SOURCE_TYPE = "html"
        logger.info("Source type: html  (%d chars)", len(html_text))
        return html_text

    logger.error("All ingestion strategies failed for %s", arxiv_id)
    LAST_SOURCE_TYPE = "unknown"
    return ""


# ---------------------------------------------------------------------------
# LaTeX ingestion pipeline
# ---------------------------------------------------------------------------


def _fetch_metadata(arxiv_id: str) -> dict:
    """
    Fetch paper metadata using arxiv-py.

    Returns dict with keys: title, authors (list), year, abstract, categories.
    Returns {} on failure — never raises.
    """
    if not _ARXIV_AVAILABLE:
        logger.warning("arxiv-py not installed — skipping metadata fetch")
        return {}

    try:
        client = arxiv_lib.Client()
        search = arxiv_lib.Search(id_list=[arxiv_id], max_results=1)
        results = list(client.results(search))

        if not results:
            logger.warning("arxiv-py returned no results for %s", arxiv_id)
            return {}

        paper = results[0]
        return {
            "title":      paper.title,
            "authors":    [str(a) for a in paper.authors],
            "year":       paper.published.year if paper.published else None,
            "abstract":   paper.summary,
            "categories": paper.categories,
            "doi":        paper.doi,
        }

    except Exception as exc:
        # Network failure, rate limit, invalid ID, etc.
        logger.warning("Metadata fetch failed for %s: %s", arxiv_id, exc)
        return {}


def _try_latex_ingestion(arxiv_id: str) -> str:
    """
    Download arXiv source tarball, extract the main .tex file,
    parse structured sections, and return filtered text.

    Returns empty string if source is unavailable or not LaTeX-based.
    """
    logger.info("Attempting LaTeX ingestion for %s ...", arxiv_id)

    # Download tarball
    tarball_bytes = _download_source_tarball(arxiv_id)
    if not tarball_bytes:
        return ""

    # Extract main .tex file
    tex_content = _extract_main_tex(tarball_bytes)
    if not tex_content:
        logger.info("No usable .tex file found in tarball")
        return ""

    logger.info("Extracted main .tex: %d chars", len(tex_content))

    # Parse sections and hyperparameters
    filtered = _parse_latex_sections(tex_content)

    # Append extracted hyperparams as a clean block at the end
    hyperparams = _extract_hyperparams_from_latex(tex_content)
    if hyperparams:
        hp_block = "\n\n=== EXTRACTED HYPERPARAMETERS (LaTeX source) ===\n"
        for name, value in hyperparams.items():
            hp_block += f"  {name}: {value}\n"
        filtered += hp_block
        logger.info("Extracted %d hyperparameters from LaTeX", len(hyperparams))

    if len(filtered) < _MIN_TEX_CHARS:
        logger.warning(
            "LaTeX filtered text too short (%d chars) — treating as failed",
            len(filtered),
        )
        return ""

    return filtered


def _download_source_tarball(arxiv_id: str) -> Optional[bytes]:
    """
    Download the arXiv source tarball for arxiv_id.

    arXiv serves source at https://arxiv.org/src/{id}
    Content-Type is application/x-eprint-tar for LaTeX papers,
    or application/pdf for PDF-only submissions.

    Returns raw bytes on success, None on failure or PDF-only submission.
    """
    url = _ARXIV_SRC_URL.format(arxiv_id=arxiv_id)
    logger.info("Downloading source tarball: %s", url)

    try:
        resp = requests.get(
            url,
            timeout=_DOWNLOAD_TIMEOUT_SEC,
            headers={"User-Agent": "Cogniterra/1.0 (reproducibility research)"},
            stream=True,
        )
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")

        # PDF-only submission — no LaTeX available
        if "pdf" in content_type and "tar" not in content_type:
            logger.info("Paper %s is PDF-only — skipping LaTeX path", arxiv_id)
            return None

        raw = resp.content
        logger.info("Downloaded %d bytes (Content-Type: %s)", len(raw), content_type)
        return raw

    except requests.exceptions.HTTPError as exc:
        # 403/404 are common for very old papers or restricted source
        logger.info("HTTP error downloading source for %s: %s", arxiv_id, exc)
        return None

    except requests.exceptions.RequestException as exc:
        logger.warning("Network error downloading source for %s: %s", arxiv_id, exc)
        return None


def _extract_main_tex(tarball_bytes: bytes) -> str:
    """
    Open tarball in-memory and return the content of the main .tex file.

    Heuristic for "main" file:
      1. File named main.tex, paper.tex, manuscript.tex, or the arxiv_id.
      2. Otherwise the largest .tex file by character count.

    Returns empty string if tarball is not a valid archive or contains no .tex.
    """
    try:
        with tarfile.open(fileobj=io.BytesIO(tarball_bytes), mode="r:*") as tar:
            tex_files: dict[str, str] = {}   # filename → decoded content
            all_tex: dict[str, str] = {}     # all .tex regardless of size

            for member in tar.getmembers():
                if not member.name.endswith(".tex"):
                    continue
                if member.size == 0:
                    continue

                try:
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    raw = f.read()

                    # Decode — try utf-8 first, fall back to latin-1
                    try:
                        content = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        content = raw.decode("latin-1", errors="replace")

                    all_tex[member.name] = content          # keep ALL for \begin{document} check
                    if len(content) >= _MIN_TEX_CHARS:
                        tex_files[member.name] = content    # size-filtered set for name/size checks

                except Exception as exc:
                    logger.debug("Skipping member %s: %s", member.name, exc)

            if not all_tex:
                return ""

            # Priority 1: file that contains \begin{document} — checked across ALL
            # .tex files regardless of size, because the root file can be small when
            # it only contains \input{} calls plus the document environment wrapper.
            for fname, content in all_tex.items():
                if r"\begin{document}" in content:
                    logger.info("Selected main .tex by \\begin{document}: %s", fname)
                    return content

            # From here on, only consider size-filtered files
            if not tex_files:
                # All files were tiny — return the largest of all_tex as last resort
                largest = max(all_tex, key=lambda k: len(all_tex[k]))
                return all_tex[largest]

            # Priority 2: known "main" filenames (case-insensitive basename)
            # Fallback for rare cases where \begin{document} was stripped/split.
            _MAIN_NAMES = {"main.tex", "paper.tex", "manuscript.tex",
                           "article.tex", "template.tex"}
            for fname, content in tex_files.items():
                if os.path.basename(fname).lower() in _MAIN_NAMES:
                    logger.info("Selected main .tex by name: %s", fname)
                    return content

            # Priority 3: largest file
            largest = max(tex_files, key=lambda k: len(tex_files[k]))
            logger.info(
                "Selected main .tex by size: %s (%d chars)",
                largest, len(tex_files[largest]),
            )
            return tex_files[largest]

    except tarfile.TarError as exc:
        logger.warning("tarfile error: %s", exc)
        return ""

    except Exception as exc:
        logger.warning("Unexpected error extracting .tex: %s", exc)
        return ""


def _parse_latex_sections(tex_content: str) -> str:
    """
    Extract relevant sections from raw LaTeX source.

    Strategy:
      - Split on \\section{...} and \\subsection{...} boundaries
      - Keep sections whose title matches _KEEP_SECTION_PATTERNS
      - Discard sections matching _STOP_SECTION_PATTERNS
      - Clean LaTeX markup from kept sections

    Returns filtered plain text.
    """
    # Split on \section{...} or \subsection{...} or \subsubsection{...}
    section_re = re.compile(
        r"\\(?:sub)*section\*?\s*\{([^}]*)\}",
        re.IGNORECASE,
    )

    parts = section_re.split(tex_content)
    # parts alternates: [pre-section-text, title, body, title, body, ...]

    kept_sections: list[str] = []

    # Always keep the preamble / abstract
    if parts:
        preamble = _clean_latex(parts[0])
        if len(preamble.strip()) > 50:
            kept_sections.append(preamble[:3000])   # cap preamble size

    # Walk section pairs
    i = 1
    while i < len(parts) - 1:
        title = parts[i].strip()
        body  = parts[i + 1] if i + 1 < len(parts) else ""
        i += 2

        title_lower = title.lower()

        # Check stop list first
        if any(re.search(pat, title_lower) for pat in _STOP_SECTION_PATTERNS):
            logger.debug("Dropping section: %r", title)
            continue

        # Check keep list
        if any(re.search(pat, title_lower) for pat in _KEEP_SECTION_PATTERNS):
            clean_body = _clean_latex(body)
            kept_sections.append(f"\n## {title}\n{clean_body}")
            logger.debug("Keeping section: %r (%d chars)", title, len(clean_body))
        else:
            logger.debug("Ignoring section (no keyword match): %r", title)

    result = "\n".join(kept_sections)
    result = re.sub(r"\n{3,}", "\n\n", result)

    logger.info(
        "LaTeX section parsing: kept %d sections → %d chars",
        len(kept_sections), len(result),
    )
    return result


def _clean_latex(text: str) -> str:
    """
    Strip common LaTeX markup, leaving readable plain text.

    Handles: commands, environments, math mode, comments, citations.
    Does NOT require a full LaTeX parser — regex is sufficient for our
    use case (we only need readable sentences for the LLM downstream).
    """
    # Remove comments
    text = re.sub(r"%.*", "", text)

    # Expand common text commands — keep the argument
    text = re.sub(r"\\(?:textbf|textit|emph|texttt|underline|textsc)\{([^}]*)\}", r"\1", text)

    # Remove citation/label/ref commands entirely
    text = re.sub(r"\\(?:cite|label|ref|eqref|autoref|cref)\{[^}]*\}", "", text)

    # Remove figure/table environments entirely
    text = re.sub(
        r"\\begin\{(?:figure|table|tabular|algorithm|lstlisting)[^}]*\}.*?"
        r"\\end\{(?:figure|table|tabular|algorithm|lstlisting)[^}]*\}",
        "",
        text,
        flags=re.DOTALL,
    )

    # Keep equation content (numbers are valuable for hyperparams)
    text = re.sub(r"\\begin\{(?:equation|align|gather)[^}]*\}", "\n[EQ] ", text)
    text = re.sub(r"\\end\{(?:equation|align|gather)[^}]*\}", " [/EQ]\n", text)

    # Inline math — preserve the raw content for hyperparam extraction
    text = re.sub(r"\$\$([^$]+)\$\$", r" [MATH: \1] ", text)
    text = re.sub(r"\$([^$\n]+)\$",   r" [MATH: \1] ", text)

    # Strip remaining LaTeX commands but keep their text arguments
    text = re.sub(r"\\[a-zA-Z]+\*?\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\*?",             "",    text)

    # Remove leftover braces and special chars
    text = re.sub(r"[{}\\]", " ", text)
    text = re.sub(r"~",       " ", text)   # non-breaking space

    # Collapse whitespace
    text = re.sub(r" {2,}",  " ",  text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _extract_hyperparams_from_latex(tex_content: str) -> dict[str, str]:
    """
    Extract hyperparameter values directly from raw LaTeX source.

    Scans the entire .tex file (before section filtering) to catch
    hyperparams defined inline, in captions, or in appendices.

    Returns dict: {param_name: value_string}
    Only the FIRST occurrence of each param is kept.
    """
    hyperparams: dict[str, str] = {}

    # Work on cleaned text for better pattern matching
    clean = re.sub(r"%.*", "", tex_content)          # strip comments
    clean = re.sub(r"\$([^$\n]+)\$", r" \1 ", clean) # expose inline math

    for pattern, name in _HYPERPARAM_PATTERNS:
        if name in hyperparams:
            continue  # already found — skip
        match = re.search(pattern, clean, re.IGNORECASE)
        if match:
            raw_val = match.group(1).strip().rstrip(".,;:")  # strip trailing punctuation
            # Sanity check — reject absurd values
            try:
                val = float(raw_val)
                if name == "lr" and (val <= 0 or val > 1.0):
                    continue
                if name in ("batch_size", "epochs", "num_heads", "num_layers",
                            "hidden_size", "embedding_dim"):
                    if val <= 0 or val > 10_000:
                        continue
                if name in ("dropout", "momentum", "gamma", "epsilon",
                            "weight_decay"):
                    if val < 0 or val > 1.0:
                        continue
            except ValueError:
                continue  # non-numeric match — skip

            hyperparams[name] = raw_val
            logger.debug("Extracted hyperparam from LaTeX: %s = %s", name, raw_val)

    return hyperparams


# ---------------------------------------------------------------------------
# Fallback: pdfplumber
# ---------------------------------------------------------------------------


def _try_pdf_fallback(arxiv_id: str) -> str:
    """
    Attempt to download and parse the arXiv PDF via pdfplumber.

    Downloads the PDF from https://arxiv.org/pdf/{id},
    extracts text page by page, then applies the same section filter
    used in the original parser_agent for consistency.

    Returns filtered text or empty string on failure.
    """
    if not _PDF_AVAILABLE:
        logger.warning("pdfplumber not installed — skipping PDF fallback")
        return ""

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    logger.info("PDF fallback: downloading %s", pdf_url)

    try:
        resp = requests.get(
            pdf_url,
            timeout=_DOWNLOAD_TIMEOUT_SEC,
            headers={"User-Agent": "Cogniterra/1.0"},
            stream=True,
        )
        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name

        try:
            pages: list[str] = []
            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages.append(page_text)

            raw_text = "\n\n".join(pages)
            logger.info("pdfplumber extracted %d chars from %d pages",
                        len(raw_text), len(pages))

        finally:
            os.unlink(tmp_path)

        if len(raw_text) < 500:
            return ""

        return _filter_sections_generic(raw_text)

    except Exception as exc:
        logger.warning("PDF fallback failed for %s: %s", arxiv_id, exc)
        return ""


# ---------------------------------------------------------------------------
# Fallback: HTML (original parser_agent logic, preserved verbatim)
# ---------------------------------------------------------------------------


def _try_html_fallback(arxiv_id: str) -> str:
    """
    HTML fallback — mirrors the original parser_agent logic exactly.
    Tries ar5iv → arxiv HTML → arxiv abstract in order.
    """
    if not _BS4_AVAILABLE:
        logger.warning("beautifulsoup4 not installed — skipping HTML fallback")
        return ""

    urls = [
        f"https://ar5iv.org/html/{arxiv_id}",
        f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}",
        f"https://arxiv.org/html/{arxiv_id}",
        f"https://arxiv.org/abs/{arxiv_id}",
    ]

    for url in urls:
        logger.info("HTML fallback: trying %s", url)
        try:
            resp = requests.get(
                url,
                timeout=45,
                headers={"User-Agent": "Mozilla/5.0 (compatible; Cogniterra/1.0)"},
            )
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "figure", "table",
                              "nav", "header", "footer", "aside"]):
                tag.decompose()

            raw = soup.get_text(separator="\n", strip=True)
            raw = re.sub(r"\n{3,}", "\n\n", raw)
            raw = re.sub(r"Page \d+", "", raw)
            raw = re.sub(r"\[[\d\s]+\]", "", raw)

            if len(raw) > 1000:
                logger.info("HTML fallback succeeded: %d chars from %s", len(raw), url)
                return _filter_sections_generic(raw)

            logger.info("Too short (%d chars) — trying next URL", len(raw))

        except requests.exceptions.RequestException as exc:
            logger.info("HTML fetch failed for %s: %s", url, exc)

    logger.error("HTML fallback: all URLs exhausted for %s", arxiv_id)
    return ""


# ---------------------------------------------------------------------------
# Shared section filter (used by PDF + HTML fallbacks)
# ---------------------------------------------------------------------------

# Imported from config if available, else duplicated here for portability
try:
    from config import KEEP_KEYWORDS as _KEEP_KEYWORDS
    from config import STOP_SECTIONS as _STOP_SECTIONS
except ImportError:
    _KEEP_KEYWORDS = [
        "method", "architecture", "model", "approach",
        "training", "experiment", "implement", "implementation",
        "hyperparameter", "optimizer", "dataset", "preprocess",
        "result", "evaluat", "benchmark", "accuracy", "top-1",
        "loss", "batch", "epoch", "layer", "network", "convolution",
        "residual", "attention", "transformer", "head", "block",
    ]
    _STOP_SECTIONS = [
        "conclusion", "conclusions", "discussion", "references",
        "appendix", "acknowledgment", "acknowledgements",
        "future work", "related work", "supplementary",
        "supplemental", "ablation", "proof", "theorem", "bibliography",
    ]


def _filter_sections_generic(text: str) -> str:
    """
    Line-by-line section filter — same logic as original parser_agent.filter_sections().
    Used for PDF and HTML fallback paths to ensure consistent output.
    """
    lines     = text.split("\n")
    keep: list[str] = []
    capturing = False

    for line in lines:
        lower = line.lower().strip()
        if any(p in lower for p in _KEEP_KEYWORDS) and len(lower) < 100:
            capturing = True
            keep.append(line)
        elif any(s in lower for s in _STOP_SECTIONS) and len(lower) < 100:
            capturing = False
        elif capturing and line.strip():
            keep.append(line)

    filtered = "\n".join(keep)
    filtered = re.sub(r"\n{3,}", "\n\n", filtered)

    if len(filtered) < 500:
        logger.warning("Section filter returned too little — using full text")
        return text

    return filtered


# ---------------------------------------------------------------------------
# Convenience: expose source_type for pipeline.py logging
# ---------------------------------------------------------------------------


def get_last_source_type() -> str:
    """Return source type used in the most recent parse_paper() call."""
    return LAST_SOURCE_TYPE


def get_last_metadata() -> dict:
    """Return paper metadata from the most recent parse_paper() call."""
    return LAST_PAPER_METADATA


# ---------------------------------------------------------------------------
# Monkey-patch note for pipeline.py (no code change required there)
# ---------------------------------------------------------------------------
# To use this module as a drop-in replacement without touching pipeline.py:
#
#   Option A — rename/replace agents/parser_agent.py with this file
#              (keep parse_paper() as the public name — already done above)
#
#   Option B — in pipeline.py, change the import at Step 1:
#              # from agents.parser_agent import parse_paper
#              from agents.latex_ingestion import parse_paper, get_last_source_type
#              ...
#              filtered_text = parse_paper(paper_id)
#              print(f"   Source type: {get_last_source_type()}")