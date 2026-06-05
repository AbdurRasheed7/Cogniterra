"""
tests/test_scoring_engine.py
============================
Unit tests for agents/scoring_engine.py (Upgrade 2).

Run with:
  python -m pytest tests/test_scoring_engine.py -v

All tests are offline -- no Groq API calls, no network.
LLM-dependent functions (_score_methodology, _score_completeness) are mocked.
Integration tests are gated behind COGNITERRA_INTEGRATION_TESTS=1.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agents.scoring_engine as se


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_golden(path: str, expected: float, tolerance: float = 2.0) -> None:
    """Write a golden JSON file to disk."""
    with open(path, "w") as f:
        json.dump({"expected_accuracy": expected, "tolerance": tolerance}, f)


def _mock_methodology_response(scores: dict) -> MagicMock:
    """Return a mock LLM response for methodology scoring."""
    msg = MagicMock()
    msg.content = json.dumps({
        "architecture_match":       scores.get("architecture_match", 1.0),
        "optimizer_match":          scores.get("optimizer_match", 1.0),
        "loss_match":               scores.get("loss_match", 1.0),
        "hyperparams_present":      scores.get("hyperparams_present", 1.0),
        "contribution_implemented": scores.get("contribution_implemented", 1.0),
        "reasoning": "Test mock response",
    })
    return msg


def _mock_completeness_response(score: int, reason: str = "Test") -> MagicMock:
    """Return a mock LLM response for completeness scoring."""
    msg = MagicMock()
    msg.content = json.dumps({"completeness_score": score, "reason": reason})
    return msg


# ===========================================================================
# Test Group 1: _score_execution()
# ===========================================================================


class TestExecutionScore(unittest.TestCase):
    """Dimension 1: execution scoring without any LLM calls."""

    def test_clean_run_scores_100(self):
        stdout = "Epoch 1/5 Loss: 0.4\nFinal Accuracy: 98.50%"
        self.assertEqual(se._score_execution(stdout, ""), 100.0)

    def test_run_with_warnings_scores_75(self):
        stdout = "Epoch 1/5\nFinal Accuracy: 97.00%"
        stderr = "UserWarning: deprecated function"
        self.assertEqual(se._score_execution(stdout, stderr), 75.0)

    def test_run_with_errors_but_output_scores_50(self):
        stdout = "Final Accuracy: 85.00%"
        stderr = "RuntimeError: something went wrong"
        self.assertEqual(se._score_execution(stdout, stderr), 50.0)

    def test_crash_no_output_scores_0(self):
        stdout = ""
        stderr = "RuntimeError: CUDA out of memory"
        self.assertEqual(se._score_execution(stdout, stderr), 0.0)

    def test_empty_stdout_and_stderr_scores_0(self):
        self.assertEqual(se._score_execution("", ""), 0.0)

    def test_output_without_final_accuracy_scores_0(self):
        # Code ran but did not print the expected accuracy line
        stdout = "Epoch 1 Loss: 0.5\nDone."
        self.assertEqual(se._score_execution(stdout, ""), 0.0)


# ===========================================================================
# Test Group 2: _score_results()
# ===========================================================================


class TestResultsScore(unittest.TestCase):
    """Dimension 3: results scoring with relative diff."""

    def test_exact_match_scores_100(self):
        self.assertEqual(se._score_results(95.0, 95.0), 100.0)

    def test_within_2_percent_scores_100(self):
        # rel_diff = |96.8 - 95| / 95 = 0.019 < 0.02
        self.assertEqual(se._score_results(96.8, 95.0), 100.0)

    def test_within_5_percent_scores_85(self):
        # rel_diff = |99.5 - 95| / 95 = 0.047 < 0.05
        self.assertEqual(se._score_results(99.5, 95.0), 85.0)

    def test_within_10_percent_scores_70(self):
        # rel_diff = |86.0 - 95| / 95 = 0.095 < 0.10
        self.assertEqual(se._score_results(86.0, 95.0), 70.0)

    def test_within_20_percent_scores_50(self):
        # rel_diff = |80.0 - 95| / 95 = 0.158 < 0.20
        self.assertEqual(se._score_results(80.0, 95.0), 50.0)

    def test_beyond_20_percent_scores_low(self):
        # rel_diff = |60.0 - 95| / 95 = 0.368
        score = se._score_results(60.0, 95.0)
        self.assertLessEqual(score, 30.0)
        self.assertGreaterEqual(score, 0.0)

    def test_no_actual_scores_0(self):
        self.assertEqual(se._score_results(None, 95.0), 0.0)

    def test_no_expected_high_actual_scores_reasonably(self):
        # No golden -- high accuracy should still get a decent score
        score = se._score_results(95.0, None)
        self.assertGreaterEqual(score, 70.0)

    def test_no_expected_low_actual_scores_low(self):
        score = se._score_results(40.0, None)
        self.assertLess(score, 50.0)

    def test_no_actual_no_expected_scores_0(self):
        self.assertEqual(se._score_results(None, None), 0.0)

    def test_zero_expected_does_not_raise(self):
        # Should handle division by zero gracefully
        try:
            score = se._score_results(5.0, 0.0)
            self.assertIsInstance(score, float)
        except ZeroDivisionError:
            self.fail("_score_results raised ZeroDivisionError on expected=0")


# ===========================================================================
# Test Group 3: _extract_accuracy()
# ===========================================================================


class TestExtractAccuracy(unittest.TestCase):
    """Accuracy extraction from stdout."""

    def test_extracts_final_accuracy_line(self):
        stdout = "Epoch 5/5 Loss: 0.1\nFinal Accuracy: 98.76%"
        self.assertAlmostEqual(se._extract_accuracy(stdout), 98.76)

    def test_extracts_test_accuracy(self):
        stdout = "Test Accuracy: 97.50%"
        self.assertAlmostEqual(se._extract_accuracy(stdout), 97.50)

    def test_converts_decimal_to_percent(self):
        # Some models print 0.985 instead of 98.5
        stdout = "Final Accuracy: 0.985"
        self.assertAlmostEqual(se._extract_accuracy(stdout), 98.5)

    def test_returns_none_for_no_accuracy(self):
        self.assertIsNone(se._extract_accuracy("Training complete."))

    def test_returns_none_for_empty_stdout(self):
        self.assertIsNone(se._extract_accuracy(""))

    def test_returns_none_for_none_input(self):
        self.assertIsNone(se._extract_accuracy(None))


# ===========================================================================
# Test Group 4: _load_golden()
# ===========================================================================


class TestLoadGolden(unittest.TestCase):
    """Golden JSON loading."""

    def test_loads_valid_golden(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"expected_accuracy": 98.5, "tolerance": 2.0}, f)
            path = f.name
        try:
            expected, tol = se._load_golden(path)
            self.assertAlmostEqual(expected, 98.5)
            self.assertAlmostEqual(tol, 2.0)
        finally:
            os.unlink(path)

    def test_returns_none_for_missing_file(self):
        expected, tol = se._load_golden("/nonexistent/path.json")
        self.assertIsNone(expected)
        self.assertAlmostEqual(tol, se.TOLERANCE_DEFAULT)

    def test_fixes_decimal_accuracy(self):
        # Groq sometimes returns 0.985 instead of 98.5
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"expected_accuracy": 0.985, "tolerance": 2.0}, f)
            path = f.name
        try:
            expected, _ = se._load_golden(path)
            self.assertAlmostEqual(expected, 98.5)
        finally:
            os.unlink(path)

    def test_returns_none_for_null_accuracy(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"expected_accuracy": None, "tolerance": 2.0}, f)
            path = f.name
        try:
            expected, _ = se._load_golden(path)
            self.assertIsNone(expected)
        finally:
            os.unlink(path)

    def test_handles_malformed_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{")
            path = f.name
        try:
            expected, tol = se._load_golden(path)
            self.assertIsNone(expected)
        finally:
            os.unlink(path)


# ===========================================================================
# Test Group 5: _score_methodology() -- LLM mocked
# ===========================================================================


class TestMethodologyScore(unittest.TestCase):
    """Dimension 2: methodology scoring with mocked LLM."""

    @patch("agents.scoring_engine._get_llm")
    def test_perfect_scores_return_100(self, mock_llm):
        mock_llm.return_value.invoke.return_value = _mock_methodology_response({
            "architecture_match": 1.0,
            "optimizer_match":    1.0,
            "loss_match":         1.0,
            "hyperparams_present": 1.0,
            "contribution_implemented": 1.0,
        })
        score, detail = se._score_methodology("paper text", "code")
        self.assertEqual(score, 100.0)

    @patch("agents.scoring_engine._get_llm")
    def test_zero_scores_return_0(self, mock_llm):
        mock_llm.return_value.invoke.return_value = _mock_methodology_response({
            "architecture_match": 0.0,
            "optimizer_match":    0.0,
            "loss_match":         0.0,
            "hyperparams_present": 0.0,
            "contribution_implemented": 0.0,
        })
        score, detail = se._score_methodology("paper text", "code")
        self.assertEqual(score, 0.0)

    @patch("agents.scoring_engine._get_llm")
    def test_partial_scores_weighted_correctly(self, mock_llm):
        # Only architecture (weight 0.30) is 1.0, rest 0
        mock_llm.return_value.invoke.return_value = _mock_methodology_response({
            "architecture_match": 1.0,
            "optimizer_match":    0.0,
            "loss_match":         0.0,
            "hyperparams_present": 0.0,
            "contribution_implemented": 0.0,
        })
        score, _ = se._score_methodology("paper text", "code")
        # 1.0 * 0.30 * 100 = 30.0
        self.assertAlmostEqual(score, 30.0)

    @patch("agents.scoring_engine._get_llm")
    def test_llm_json_error_returns_0(self, mock_llm):
        msg = MagicMock()
        msg.content = "This is not JSON at all"
        mock_llm.return_value.invoke.return_value = msg
        score, detail = se._score_methodology("paper text", "code")
        self.assertEqual(score, 0.0)
        self.assertEqual(detail, {})

    @patch("agents.scoring_engine._get_llm")
    def test_llm_exception_returns_0(self, mock_llm):
        mock_llm.return_value.invoke.side_effect = Exception("API timeout")
        score, detail = se._score_methodology("paper text", "code")
        self.assertEqual(score, 0.0)


# ===========================================================================
# Test Group 6: ScoringResult.to_dict() backward compatibility
# ===========================================================================


class TestScoringResultToDict(unittest.TestCase):
    """Verify to_dict() preserves all original tester_agent keys."""

    ORIGINAL_KEYS = {
        "paper_id", "timestamp", "expected_accuracy", "actual_accuracy",
        "difference", "tolerance", "status", "reproducibility_score",
        "has_errors", "stderr_preview",
    }
    NEW_KEYS = {
        "final_score", "dimension_scores", "methodology_detail",
        "completeness_reason",
    }

    def _make_result(self) -> se.ScoringResult:
        return se.ScoringResult(
            paper_id="1512.03385",
            timestamp="2026-01-01 00:00:00 IST",
            final_score=78.5,
            dimensions=se.DimensionScores(
                execution=100.0, methodology=80.0,
                results=70.0, completeness=60.0,
            ),
            expected_accuracy=95.0,
            actual_accuracy=93.0,
            difference=2.0,
            tolerance=2.0,
            status="PASS",
            has_errors=False,
        )

    def test_all_original_keys_present(self):
        d = self._make_result().to_dict()
        for key in self.ORIGINAL_KEYS:
            self.assertIn(key, d, f"Missing original key: {key}")

    def test_all_new_keys_present(self):
        d = self._make_result().to_dict()
        for key in self.NEW_KEYS:
            self.assertIn(key, d, f"Missing new key: {key}")

    def test_reproducibility_score_equals_final_score(self):
        result = self._make_result()
        d = result.to_dict()
        self.assertEqual(d["reproducibility_score"], d["final_score"])

    def test_dimension_scores_has_all_4_keys(self):
        d = self._make_result().to_dict()
        for dim in ("execution", "methodology", "results", "completeness"):
            self.assertIn(dim, d["dimension_scores"])

    def test_scores_are_rounded_to_1dp(self):
        result = self._make_result()
        result.final_score = 78.555
        d = result.to_dict()
        # Should be rounded to 1dp
        self.assertEqual(d["final_score"], round(78.555, 1))


# ===========================================================================
# Test Group 7: full score_paper() integration (LLM mocked)
# ===========================================================================


class TestScorePaperIntegration(unittest.TestCase):
    """End-to-end score_paper() with LLM mocked out."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.golden = os.path.join(self.tmp, "test_expected.json")

    @patch("agents.scoring_engine._get_llm")
    def test_full_pipeline_with_golden(self, mock_llm):
        _make_golden(self.golden, 95.0, 2.0)
        mock_llm.return_value.invoke.side_effect = [
            _mock_methodology_response({
                "architecture_match": 1.0, "optimizer_match": 1.0,
                "loss_match": 1.0, "hyperparams_present": 0.5,
                "contribution_implemented": 1.0,
            }),
            _mock_completeness_response(80, "Most contributions implemented"),
        ]
        stdout = "Epoch 5\nFinal Accuracy: 94.50%"
        result = se.score_paper("test", stdout, "", self.golden,
                                "paper text", "some code")
        d = result.to_dict()
        self.assertGreater(d["final_score"], 0)
        self.assertLessEqual(d["final_score"], 100)
        self.assertAlmostEqual(d["actual_accuracy"], 94.5)
        self.assertAlmostEqual(d["expected_accuracy"], 95.0)

    @patch("agents.scoring_engine._get_llm")
    def test_crashed_code_scores_low(self, mock_llm):
        mock_llm.return_value.invoke.side_effect = [
            _mock_methodology_response({"architecture_match": 0, "optimizer_match": 0,
                "loss_match": 0, "hyperparams_present": 0, "contribution_implemented": 0}),
            _mock_completeness_response(10, "Code crashed"),
        ]
        result = se.score_paper("test", "", "RuntimeError: crash", self.golden,
                                "paper", "code")
        self.assertEqual(result.dimensions.execution, 0.0)
        self.assertLess(result.final_score, 30.0)

    @patch("agents.scoring_engine._get_llm")
    def test_no_golden_still_produces_result(self, mock_llm):
        missing_path = os.path.join(self.tmp, "nonexistent.json")
        mock_llm.return_value.invoke.side_effect = [
            _mock_methodology_response({
                "architecture_match": 0.5, "optimizer_match": 0.5,
                "loss_match": 0.5, "hyperparams_present": 0.5,
                "contribution_implemented": 0.5,
            }),
            _mock_completeness_response(50, "Half implemented"),
        ]
        stdout = "Final Accuracy: 88.0%"
        result = se.score_paper("test", stdout, "", missing_path, "paper", "code")
        self.assertIsNone(result.expected_accuracy)
        self.assertGreater(result.final_score, 0)

    def test_run_test_shim_returns_dict(self):
        """run_test() backward compat shim must return a plain dict."""
        with patch("agents.scoring_engine.score_paper") as mock_sp:
            mock_result = MagicMock()
            mock_result.to_dict.return_value = {"reproducibility_score": 75}
            mock_sp.return_value = mock_result
            result = se.run_test("pid", "stdout", "stderr", "path.json")
            self.assertIsInstance(result, dict)
            self.assertIn("reproducibility_score", result)


# ===========================================================================
# Test Group 8: weight sanity
# ===========================================================================


class TestWeightSanity(unittest.TestCase):
    """Verify weights sum to 1.0 and final score stays in 0-100."""

    def test_weights_sum_to_1(self):
        total = (se._W_EXECUTION + se._W_METHODOLOGY
                 + se._W_RESULTS + se._W_COMPLETENESS)
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_all_100_gives_100(self):
        score = (
            se._W_EXECUTION    * 100
            + se._W_METHODOLOGY  * 100
            + se._W_RESULTS      * 100
            + se._W_COMPLETENESS * 100
        )
        self.assertAlmostEqual(score, 100.0)

    def test_all_0_gives_0(self):
        score = (
            se._W_EXECUTION    * 0
            + se._W_METHODOLOGY  * 0
            + se._W_RESULTS      * 0
            + se._W_COMPLETENESS * 0
        )
        self.assertEqual(score, 0.0)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)