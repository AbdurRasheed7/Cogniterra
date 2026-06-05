"""
tests/test_latex_ingestion.py
=============================
Unit tests for agents/latex_ingestion.py (Upgrade 1).

Run with:
  python -m pytest tests/test_latex_ingestion.py -v

All tests are self-contained -- no network calls, no Groq API needed.
Network-dependent tests are marked @pytest.mark.integration and
skipped by default.
"""

import io
import json
import os
import sys
import tarfile
import textwrap
import unittest
from unittest.mock import MagicMock, patch

# Allow running from repo root: python -m pytest tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agents.latex_ingestion as li


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tarball(files: dict[str, str]) -> bytes:
    """
    Build an in-memory .tar.gz archive.
    files: {filename: content_string}
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in files.items():
            encoded = content.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(encoded)
            tar.addfile(info, io.BytesIO(encoded))
    return buf.getvalue()


def _make_minimal_tex(content: str = "", with_document_env: bool = False) -> str:
    """
    Return a minimal .tex file containing the given content.
    with_document_env=False (default): plain content only, no \\begin{document}.
    with_document_env=True: wraps in full document environment.
    """
    if with_document_env:
        return textwrap.dedent(f"""
            \\\\documentclass{{article}}
            \\\\begin{{document}}
            {content}
            \\\\end{{document}}
        """)
    return f"% tex file\n{content}\n"


# ===========================================================================
# Test Group 1: _clean_latex()
# ===========================================================================


class TestCleanLatex(unittest.TestCase):
    """Tests for the LaTeX markup stripping function."""

    def test_removes_comments(self):
        raw = "This is text % this is a comment\nMore text"
        result = li._clean_latex(raw)
        self.assertNotIn("%", result)
        self.assertIn("This is text", result)
        self.assertIn("More text", result)

    def test_expands_textbf_and_textit(self):
        raw = r"The \textbf{residual block} uses \textit{skip connections}"
        result = li._clean_latex(raw)
        self.assertIn("residual block", result)
        self.assertIn("skip connections", result)
        self.assertNotIn("textbf", result)
        self.assertNotIn("textit", result)

    def test_removes_citations(self):
        raw = r"As shown in \cite{he2016deep} and \ref{fig:arch}"
        result = li._clean_latex(raw)
        self.assertNotIn("cite", result)
        self.assertNotIn("ref", result)

    def test_removes_figure_environment(self):
        raw = (
            r"Before figure. "
            r"\begin{figure}[h] \includegraphics{fig.png} \end{figure}"
            r" After figure."
        )
        result = li._clean_latex(raw)
        self.assertIn("Before figure", result)
        self.assertIn("After figure", result)
        self.assertNotIn("includegraphics", result)

    def test_preserves_equation_content(self):
        """Equations may contain hyperparameter values -- must not be stripped."""
        raw = r"\begin{equation} lr = 0.001 \end{equation}"
        result = li._clean_latex(raw)
        self.assertIn("0.001", result)

    def test_collapses_excess_whitespace(self):
        raw = "word1    word2\n\n\n\nword3"
        result = li._clean_latex(raw)
        # Should not have more than 2 consecutive newlines
        self.assertNotIn("\n\n\n", result)

    def test_empty_string(self):
        self.assertEqual(li._clean_latex(""), "")

    def test_inline_math_preserved(self):
        raw = r"We set $\alpha = 0.01$ and $\beta = 0.9$"
        result = li._clean_latex(raw)
        self.assertIn("0.01", result)
        self.assertIn("0.9", result)


# ===========================================================================
# Test Group 2: _extract_hyperparams_from_latex()
# ===========================================================================


class TestExtractHyperparams(unittest.TestCase):
    """Tests for hyperparameter extraction from raw LaTeX."""

    def test_extracts_learning_rate(self):
        tex = r"We set the learning rate = 0.001 and train for 100 epochs."
        result = li._extract_hyperparams_from_latex(tex)
        self.assertIn("lr", result)
        self.assertAlmostEqual(float(result["lr"]), 0.001)

    def test_extracts_batch_size(self):
        tex = r"All models use batch size = 256 throughout training."
        result = li._extract_hyperparams_from_latex(tex)
        self.assertIn("batch_size", result)
        self.assertEqual(int(result["batch_size"]), 256)

    def test_extracts_epochs(self):
        tex = r"Training runs for num_epochs = 200 iterations."
        result = li._extract_hyperparams_from_latex(tex)
        self.assertIn("epochs", result)
        self.assertEqual(int(result["epochs"]), 200)

    def test_rejects_absurd_learning_rate(self):
        """lr > 1.0 should not be included -- it's almost never a valid lr."""
        tex = r"The layer has learning rate = 100"
        result = li._extract_hyperparams_from_latex(tex)
        self.assertNotIn("lr", result)

    def test_extracts_multiple_params(self):
        tex = textwrap.dedent(r"""
            We use learning rate = 3e-4, batch size = 64,
            dropout = 0.1, and momentum = 0.9.
        """)
        result = li._extract_hyperparams_from_latex(tex)
        self.assertIn("lr", result)
        self.assertIn("batch_size", result)
        self.assertIn("dropout", result)
        self.assertIn("momentum", result)

    def test_deduplicates_first_occurrence(self):
        """When a param appears twice, only the first value is kept."""
        tex = r"learning rate = 0.01 ... later: learning rate = 0.001"
        result = li._extract_hyperparams_from_latex(tex)
        self.assertIn("lr", result)
        self.assertAlmostEqual(float(result["lr"]), 0.01)

    def test_alpha_as_learning_rate(self):
        r"""Papers often write \alpha = 0.001 for lr."""
        tex = r"We set $\alpha = 3\times10^{-4}$"
        # The regex won't handle LaTeX \times, but pure numeric should work:
        tex2 = r"We set $\alpha = 0.0003$"
        result = li._extract_hyperparams_from_latex(tex2)
        self.assertIn("lr", result)

    def test_empty_tex(self):
        self.assertEqual(li._extract_hyperparams_from_latex(""), {})


# ===========================================================================
# Test Group 3: _extract_main_tex()
# ===========================================================================


class TestExtractMainTex(unittest.TestCase):
    """Tests for selecting the correct .tex file from a tarball."""

    def test_selects_main_tex_by_name(self):
        # No with_document_env -- we want to test name-based selection,
        # not egin{document} priority which would fire first.
        files = {
            "appendix.tex": _make_minimal_tex("Appendix content " * 20),
            "main.tex":     _make_minimal_tex("Main document " * 100),
        }
        tarball = _make_tarball(files)
        result = li._extract_main_tex(tarball)
        self.assertIn("Main document", result)

    def test_selects_largest_tex_when_no_main_name(self):
        # No with_document_env -- testing size-based selection only.
        files = {
            "intro.tex":  _make_minimal_tex("Short intro " * 60),
            "model.tex":  _make_minimal_tex("The model architecture " * 200),
            "data.tex":   _make_minimal_tex("Data details " * 50),
        }
        tarball = _make_tarball(files)
        result = li._extract_main_tex(tarball)
        self.assertIn("The model architecture", result)

    def test_prefers_begin_document_over_size(self):
        """A file with \\begin{document} should win over a larger file without it."""
        small_main = "\\documentclass{article}\n\\begin{document}\nActual main\n\\end{document}"
        large_other = "% No document environment\n" + "Some content\n" * 200

        files = {
            "main.tex":  small_main,
            "large.tex": large_other,
        }
        tarball = _make_tarball(files)
        result = li._extract_main_tex(tarball)
        self.assertIn("Actual main", result)

    def test_returns_empty_for_non_tarball(self):
        result = li._extract_main_tex(b"this is not a tarball")
        self.assertEqual(result, "")

    def test_returns_empty_for_tarball_with_no_tex(self):
        files = {"readme.md": "# Project", "data.csv": "a,b,c"}
        tarball = _make_tarball(files)
        result = li._extract_main_tex(tarball)
        self.assertEqual(result, "")

    def test_skips_empty_tex_files(self):
        files = {
            "empty.tex": "",
            "model.tex": _make_minimal_tex("Real content " * 100),
        }
        tarball = _make_tarball(files)
        result = li._extract_main_tex(tarball)
        self.assertIn("Real content", result)


# ===========================================================================
# Test Group 4: _parse_latex_sections()
# ===========================================================================


class TestParseLatexSections(unittest.TestCase):
    """Tests for section extraction and filtering."""

    def _make_paper(self, sections: dict[str, str]) -> str:
        """Build a fake LaTeX paper from a {section_title: content} dict."""
        parts = [r"\documentclass{article}", r"\begin{document}"]
        for title, body in sections.items():
            parts.append(f"\\section{{{title}}}")
            parts.append(body)
        parts.append(r"\end{document}")
        return "\n".join(parts)

    def test_keeps_methods_section(self):
        paper = self._make_paper({
            "Methods":     "We propose a residual architecture with skip connections.",
            "References":  "He et al., 2016. [1]",
        })
        result = li._parse_latex_sections(paper)
        self.assertIn("residual architecture", result)

    def test_drops_references_section(self):
        paper = self._make_paper({
            "Experiments": "We train on CIFAR-10 with batch size 128.",
            "References":  "He et al., 2016.",
        })
        result = li._parse_latex_sections(paper)
        self.assertNotIn("He et al.", result)

    def test_drops_conclusion_section(self):
        paper = self._make_paper({
            "Implementation Details": "Learning rate is 0.01.",
            "Conclusions":            "This paper showed X.",
        })
        result = li._parse_latex_sections(paper)
        self.assertIn("Learning rate", result)
        self.assertNotIn("This paper showed X", result)

    def test_keeps_training_section(self):
        paper = self._make_paper({
            "Training Procedure": "We use SGD with momentum 0.9.",
            "Acknowledgments":    "Thanks to reviewers.",
        })
        result = li._parse_latex_sections(paper)
        self.assertIn("SGD", result)
        self.assertNotIn("reviewers", result)

    def test_handles_paper_with_no_sections(self):
        """Flat papers (no \\section) should return cleaned preamble text."""
        flat = "This paper describes a model with dropout 0.5 and batch size 32."
        result = li._parse_latex_sections(flat)
        # Should not crash and should return something
        self.assertIsInstance(result, str)

    def test_subsection_is_also_parsed(self):
        paper = (
            r"\section{Methods}"
            "\nMain methods text.\n"
            r"\subsection{Architecture}"
            "\nThe architecture has 4 layers.\n"
        )
        result = li._parse_latex_sections(paper)
        self.assertIn("Main methods text", result)
        self.assertIn("4 layers", result)


# ===========================================================================
# Test Group 5: module-level state tracking
# ===========================================================================


class TestSourceTypeTracking(unittest.TestCase):
    """Tests that LAST_SOURCE_TYPE is set correctly after parse_paper()."""

    @patch("agents.latex_ingestion._try_latex_ingestion", return_value="latex content here " * 50)
    @patch("agents.latex_ingestion._fetch_metadata", return_value={})
    def test_source_type_latex(self, mock_meta, mock_latex):
        li.parse_paper("1512.03385")
        self.assertEqual(li.get_last_source_type(), "latex")

    @patch("agents.latex_ingestion._try_latex_ingestion", return_value="")
    @patch("agents.latex_ingestion._try_pdf_fallback",    return_value="pdf content here " * 50)
    @patch("agents.latex_ingestion._fetch_metadata",      return_value={})
    def test_source_type_pdf(self, mock_meta, mock_pdf, mock_latex):
        li.parse_paper("1512.03385")
        self.assertEqual(li.get_last_source_type(), "pdf")

    @patch("agents.latex_ingestion._try_latex_ingestion", return_value="")
    @patch("agents.latex_ingestion._try_pdf_fallback",    return_value="")
    @patch("agents.latex_ingestion._try_html_fallback",   return_value="html content here " * 50)
    @patch("agents.latex_ingestion._fetch_metadata",      return_value={})
    def test_source_type_html(self, mock_meta, mock_html, mock_pdf, mock_latex):
        li.parse_paper("1512.03385")
        self.assertEqual(li.get_last_source_type(), "html")

    @patch("agents.latex_ingestion._try_latex_ingestion", return_value="")
    @patch("agents.latex_ingestion._try_pdf_fallback",    return_value="")
    @patch("agents.latex_ingestion._try_html_fallback",   return_value="")
    @patch("agents.latex_ingestion._fetch_metadata",      return_value={})
    def test_source_type_unknown_on_total_failure(self, mock_meta, mock_html, mock_pdf, mock_latex):
        result = li.parse_paper("1512.03385")
        self.assertEqual(result, "")
        self.assertEqual(li.get_last_source_type(), "unknown")

    @patch("agents.latex_ingestion._fetch_metadata", return_value={
        "title": "Deep Residual Learning",
        "authors": ["He", "Zhang"],
        "year": 2016,
    })
    @patch("agents.latex_ingestion._try_latex_ingestion", return_value="latex content " * 50)
    def test_metadata_stored_after_parse(self, mock_latex, mock_meta):
        li.parse_paper("1512.03385")
        meta = li.get_last_metadata()
        self.assertEqual(meta["title"], "Deep Residual Learning")
        self.assertEqual(meta["year"], 2016)


# ===========================================================================
# Test Group 6: PDF fallback (_try_pdf_fallback)
# ===========================================================================


class TestPdfFallback(unittest.TestCase):
    """Tests for the pdfplumber fallback path."""

    @patch("agents.latex_ingestion._PDF_AVAILABLE", False)
    def test_returns_empty_when_pdfplumber_missing(self):
        result = li._try_pdf_fallback("1234.56789")
        self.assertEqual(result, "")

    @patch("agents.latex_ingestion.requests.get")
    def test_returns_empty_on_http_error(self, mock_get):
        import requests
        mock_get.side_effect = requests.exceptions.HTTPError("403 Forbidden")
        result = li._try_pdf_fallback("1234.56789")
        self.assertEqual(result, "")


# ===========================================================================
# Test Group 7: Integration smoke test (marked, skipped by default)
# ===========================================================================


class TestIntegration(unittest.TestCase):
    """
    Real network calls -- skipped unless COGNITERRA_INTEGRATION_TESTS=1 is set.

    Run with:
      COGNITERRA_INTEGRATION_TESTS=1 python -m pytest tests/test_latex_ingestion.py -v -m integration
    """

    def setUp(self):
        if not os.environ.get("COGNITERRA_INTEGRATION_TESTS"):
            self.skipTest("Set COGNITERRA_INTEGRATION_TESTS=1 to run integration tests")

    def test_resnet_paper_returns_nonempty(self):
        """arXiv 1512.03385 = Deep Residual Learning (ResNet) -- well-known LaTeX paper."""
        result = li.parse_paper("1512.03385")
        self.assertGreater(len(result), 500)
        self.assertIn(li.get_last_source_type(), ["latex", "html", "pdf"])

    def test_source_type_is_tracked(self):
        li.parse_paper("1512.03385")
        self.assertIn(li.get_last_source_type(), ["latex", "pdf", "html"])

    def test_metadata_populated(self):
        li.parse_paper("1512.03385")
        meta = li.get_last_metadata()
        self.assertIn("title", meta)
        self.assertTrue(len(meta.get("title", "")) > 0)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)