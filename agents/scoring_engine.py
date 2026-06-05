"""
agents/scoring_engine.py
========================
Upgrade 2 -- 4-Dimension Weighted Scoring System for Cogniterra
----------------------------------------------------------------
Replaces the binary reproducibility score in tester_agent.py with a
four-dimension weighted score that captures execution quality,
methodology fidelity, numerical results accuracy, and code completeness.

Dimensions and weights:
  Dimension 1 -- Execution Score    (weight: 0.20)
  Dimension 2 -- Methodology Score  (weight: 0.35)  LLM-based
  Dimension 3 -- Results Score      (weight: 0.30)  relative diff
  Dimension 4 -- Completeness Score (weight: 0.15)  LLM-based

Final score: 0.20*E + 0.35*M + 0.30*R + 0.15*C  (0-100)

Public API:
  score_paper(
      paper_id:       str,
      stdout:         str,
      stderr:         str,
      golden_path:    str,
      filtered_text:  str = "",   # paper text -- needed for M + C dimensions
      code:           str = "",   # generated code -- needed for M + C dimensions
  ) -> ScoringResult

  The returned ScoringResult is a dataclass whose .to_dict() is a drop-in
  replacement for the dict that tester_agent.run_test() currently returns,
  so pipeline.py needs only a one-line import change.

Integration with pipeline.py (Step 6):
  # Replace:
  #   from agents.tester_agent import run_test
  #   test_result = run_test(paper_id, stdout, stderr, golden_path)
  # With:
  from agents.scoring_engine import score_paper
  test_result = score_paper(
      paper_id, stdout, stderr, golden_path,
      filtered_text=filtered_text,
      code=code,
  ).to_dict()

  Everything downstream (hallucination check, HTML report) continues to
  work because to_dict() preserves all existing keys and adds new ones.

Dependencies (already in project):
  langchain_groq, python-dotenv
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# LLM setup -- temperature 0.1 for deterministic scoring
# ---------------------------------------------------------------------------

try:
    from config import GROQ_MODEL, GOLDEN_DIR, TOLERANCE_DEFAULT
except ImportError:
    GROQ_MODEL = "llama-3.3-70b-versatile"
    GOLDEN_DIR = os.path.join("tests", "golden")
    TOLERANCE_DEFAULT = 2.0

_llm: Optional[ChatGroq] = None        # primary key instance
_llm_fallback: Optional[ChatGroq] = None  # fallback key instance


def _get_llm(use_fallback: bool = False) -> ChatGroq:
    """
    Return LLM instance, lazy-initialised on first call.

    Pass use_fallback=True to use GROQ_API_KEY_2 instead of the primary key.
    Lazy init allows test imports without any API key set.

    To add a second key, set GROQ_API_KEY_2 in your .env file:
        GROQ_API_KEY_2=gsk_your_second_key_here
    """
    global _llm, _llm_fallback

    if use_fallback:
        fallback_key = os.getenv("GROQ_API_KEY_2", "")
        if not fallback_key:
            logger.warning(
                "GROQ_API_KEY_2 not set -- falling back to primary key. "
                "Add GROQ_API_KEY_2 to your .env to use a second account."
            )
            fallback_key = os.getenv("GROQ_API_KEY", "")
        if _llm_fallback is None:
            _llm_fallback = ChatGroq(
                model=GROQ_MODEL,
                temperature=0.1,
                max_tokens=1024,
                groq_api_key=fallback_key,
            )
        return _llm_fallback

    if _llm is None:
        _llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=0.1,
            max_tokens=1024,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
    return _llm

# ---------------------------------------------------------------------------
# Dimension weights -- must sum to 1.0
# ---------------------------------------------------------------------------

_W_EXECUTION    = 0.20
_W_METHODOLOGY  = 0.35
_W_RESULTS      = 0.30
_W_COMPLETENESS = 0.15

assert abs(_W_EXECUTION + _W_METHODOLOGY + _W_RESULTS + _W_COMPLETENESS - 1.0) < 1e-9

# ---------------------------------------------------------------------------
# LLM prompt constants -- defined at module top per code quality spec
# ---------------------------------------------------------------------------

_METHODOLOGY_PROMPT = """\
You are an ML paper reproducibility auditor evaluating a PROXY implementation.

IMPORTANT CONTEXT: This code runs on a small benchmark (MNIST, CIFAR-10, CartPole,
20Newsgroups, MovieLens) to verify the paper's KEY ARCHITECTURAL CONTRIBUTION is
correctly implemented -- not to replicate the paper's exact experimental setup
(e.g. ImageNet, MuJoCo, WMT14). Evaluate whether the core ideas are present,
accounting for proxy scale.

PAPER METHODOLOGY:
{methods_text}

GENERATED CODE (proxy implementation):
{code}

Score each sub-dimension 0, 0.5, or 1.0:
  0.0 = completely absent or fundamentally wrong
  0.5 = partially present or approximated
  1.0 = correctly implemented (accounting for proxy scale)

Sub-dimensions:
  architecture_match (0.30):
    Is the key architectural pattern present?
    ResNet -> skip/residual connections? VGG -> deep sequential conv blocks?
    Transformer -> self-attention + FFN? GCN -> graph convolution with adj matrix?
    DDPG -> separate actor + critic networks? NCF -> user+item embeddings + MLP?
    Score 1.0 if defining structure present even if scaled down.
    Score 0.5 if partially present. Score 0.0 if completely generic.

  optimizer_match (0.20):
    Is the optimizer consistent with the paper?
    Score 1.0 exact match. Score 0.5 reasonable alternative. Score 0.0 wrong family.

  loss_match (0.20):
    Is the loss appropriate for the task?
    Score 1.0 correct. Score 0.5 approximate. Score 0.0 wrong task entirely.

  hyperparams_present (0.15):
    Are key hyperparameter values from the paper referenced in the code?
    Score 1.0 most match. Score 0.5 some match. Score 0.0 all arbitrary defaults.

  contribution_implemented (0.15):
    Is the paper's PRIMARY contribution clearly present?
    Score 1.0 if the defining novelty is implemented.
    Score 0.5 if partially. Score 0.0 if absent.

Return ONLY valid JSON, no markdown, no explanation:
{{
  "architecture_match": <0|0.5|1.0>,
  "optimizer_match": <0|0.5|1.0>,
  "loss_match": <0|0.5|1.0>,
  "hyperparams_present": <0|0.5|1.0>,
  "contribution_implemented": <0|0.5|1.0>,
  "reasoning": "<one sentence: what key contribution is present and what is missing>"
}}"""

_COMPLETENESS_PROMPT = """\
You are an ML paper reproducibility auditor evaluating a PROXY implementation.

CONTEXT: This is a scaled-down proxy experiment. The code should implement the
paper's core contribution at a functional level. It is NOT expected to replicate
every experiment. Focus only on the PRIMARY contribution.

Ignore: dataset scale differences, hyperparameter tuning, ablation studies,
        appendix experiments. Focus on: key algorithms, architectural components,
        training procedures central to the main contribution.

PAPER TEXT:
{paper_text}

GENERATED CODE (proxy implementation):
{code}

Scoring guide:
  80-100: Core contribution fully implemented, proxy is appropriate
  60-79:  Core contribution present but some key components missing
  40-59:  Partial implementation, main idea visible but incomplete
  20-39:  Minimal, barely related to the paper's contribution
  0-19:   Generic code with no connection to the paper

Return ONLY valid JSON, no markdown, no explanation:
{{
  "completeness_score": <0-100 integer>,
  "reason": "<one sentence: what is present and what key component is missing>"
}}"""

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DimensionScores:
    """Raw scores for each of the 4 dimensions (0-100 each)."""
    execution:    float = 0.0
    methodology:  float = 0.0
    results:      float = 0.0
    completeness: float = 0.0

    # Methodology sub-scores (stored for report transparency)
    architecture_match:       float = 0.0
    optimizer_match:          float = 0.0
    loss_match:               float = 0.0
    hyperparams_present:      float = 0.0
    contribution_implemented: float = 0.0
    methodology_reasoning:    str   = ""
    completeness_reason:      str   = ""


@dataclass
class ScoringResult:
    """
    Complete scoring result.

    .to_dict() is backward-compatible with the dict returned by
    tester_agent.run_test(), so pipeline.py needs no structural changes.
    """
    paper_id:              str
    timestamp:             str
    final_score:           float          # weighted composite 0-100
    dimensions:            DimensionScores = field(default_factory=DimensionScores)

    # Fields preserved from original tester_agent for backward compatibility
    expected_accuracy:     Optional[float] = None
    actual_accuracy:       Optional[float] = None
    difference:            Optional[float] = None
    tolerance:             float           = 2.0
    status:                str             = ""
    has_errors:            bool            = False
    stderr_preview:        Optional[str]   = None

    def to_dict(self) -> dict:
        """
        Return a dict compatible with the original run_test() output.

        New keys are added under 'dimension_scores' so nothing downstream breaks.
        'reproducibility_score' maps to final_score for full backward compatibility.
        """
        return {
            # --- Original keys (unchanged) ---
            "paper_id":              self.paper_id,
            "timestamp":             self.timestamp,
            "expected_accuracy":     self.expected_accuracy,
            "actual_accuracy":       self.actual_accuracy,
            "difference":            self.difference,
            "tolerance":             self.tolerance,
            "status":                self.status,
            "reproducibility_score": round(self.final_score, 1),
            "has_errors":            self.has_errors,
            "stderr_preview":        self.stderr_preview,
            # --- New keys (Upgrade 2) ---
            "final_score":           round(self.final_score, 1),
            "dimension_scores": {
                "execution":    round(self.dimensions.execution, 1),
                "methodology":  round(self.dimensions.methodology, 1),
                "results":      round(self.dimensions.results, 1),
                "completeness": round(self.dimensions.completeness, 1),
            },
            "methodology_detail": {
                "architecture_match":       self.dimensions.architecture_match,
                "optimizer_match":          self.dimensions.optimizer_match,
                "loss_match":               self.dimensions.loss_match,
                "hyperparams_present":      self.dimensions.hyperparams_present,
                "contribution_implemented": self.dimensions.contribution_implemented,
                "reasoning":                self.dimensions.methodology_reasoning,
            },
            "completeness_reason": self.dimensions.completeness_reason,
        }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def score_paper(
    paper_id:      str,
    stdout:        str,
    stderr:        str,
    golden_path:   str,
    filtered_text: str = "",
    code:          str = "",
) -> ScoringResult:
    """
    Compute the 4-dimension reproducibility score for a paper run.

    Args:
        paper_id:      arXiv paper ID, used for logging and result labelling.
        stdout:        Captured stdout from the executed code.
        stderr:        Captured stderr from the executed code.
        golden_path:   Path to the golden JSON file (may not exist yet).
        filtered_text: Paper text from parse_paper() -- used for M + C scoring.
        code:          Generated code string -- used for M + C scoring.

    Returns:
        ScoringResult with all dimensions populated and .to_dict() ready.
    """
    logger.info("=== score_paper(%s) -- Upgrade 2: 4-Dimension Scoring ===", paper_id)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    dims = DimensionScores()

    # ── Load golden JSON ──────────────────────────────────────────────────────
    expected_accuracy, tolerance = _load_golden(golden_path)

    # ── Extract actual accuracy from stdout ───────────────────────────────────
    actual_accuracy = _extract_accuracy(stdout)

    # ── Dimension 1: Execution Score (weight 0.20) ────────────────────────────
    dims.execution = _score_execution(stdout, stderr)
    logger.info("D1 Execution:    %.1f", dims.execution)

    # ── Dimension 2: Methodology Score (weight 0.35) ──────────────────────────
    if filtered_text and code:
        dims.methodology, method_detail = _score_methodology(filtered_text, code)
        dims.architecture_match       = method_detail.get("architecture_match", 0.0)
        dims.optimizer_match          = method_detail.get("optimizer_match", 0.0)
        dims.loss_match               = method_detail.get("loss_match", 0.0)
        dims.hyperparams_present      = method_detail.get("hyperparams_present", 0.0)
        dims.contribution_implemented = method_detail.get("contribution_implemented", 0.0)
        dims.methodology_reasoning    = method_detail.get("reasoning", "")
    else:
        logger.warning("D2 Methodology: skipped -- no paper text or code provided")
        dims.methodology = 0.0
    logger.info("D2 Methodology:  %.1f", dims.methodology)

    # ── Dimension 3: Results Score (weight 0.30) ──────────────────────────────
    dims.results = _score_results(actual_accuracy, expected_accuracy)
    logger.info("D3 Results:      %.1f", dims.results)

    # ── Dimension 4: Completeness Score (weight 0.15) ─────────────────────────
    if filtered_text and code:
        dims.completeness, dims.completeness_reason = _score_completeness(
            filtered_text, code
        )
    else:
        logger.warning("D4 Completeness: skipped -- no paper text or code provided")
        dims.completeness = 0.0
    logger.info("D4 Completeness: %.1f", dims.completeness)

    # ── Weighted final score ──────────────────────────────────────────────────
    final_score = (
        _W_EXECUTION    * dims.execution
        + _W_METHODOLOGY  * dims.methodology
        + _W_RESULTS      * dims.results
        + _W_COMPLETENESS * dims.completeness
    )
    final_score = round(min(100.0, max(0.0, final_score)), 1)
    logger.info("Final score:     %.1f / 100", final_score)

    # ── Status string ─────────────────────────────────────────────────────────
    status = _build_status(actual_accuracy, expected_accuracy, tolerance, final_score)

    # ── Difference (for backward compat with HTML report) ─────────────────────
    difference: Optional[float] = None
    if actual_accuracy is not None and expected_accuracy is not None:
        difference = round(abs(actual_accuracy - expected_accuracy), 2)

    return ScoringResult(
        paper_id=paper_id,
        timestamp=timestamp,
        final_score=final_score,
        dimensions=dims,
        expected_accuracy=expected_accuracy,
        actual_accuracy=actual_accuracy,
        difference=difference,
        tolerance=tolerance,
        status=status,
        has_errors=bool(stderr and ("Error" in stderr or "Exception" in stderr)),
        stderr_preview=stderr[:200] if stderr else None,
    )


# ---------------------------------------------------------------------------
# Dimension 1: Execution Score
# ---------------------------------------------------------------------------

def _score_execution(stdout: str, stderr: str) -> float:
    """
    Score how cleanly the code ran.

      Ran cleanly (output + no errors):          100
      Ran with warnings only:                     75
      Ran with minor errors but produced output:  50
      Crashed (no accuracy output):                0

    Returns float 0-100.
    """
    has_output    = bool(stdout and "Final Accuracy" in stdout)
    has_errors    = bool(stderr and ("Error" in stderr or "Exception" in stderr))
    has_warnings  = bool(stderr and ("Warning" in stderr or "warning" in stderr))

    if has_output and not has_errors:
        if has_warnings:
            return 75.0
        return 100.0
    elif has_output and has_errors:
        return 50.0
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Dimension 2: Methodology Score
# ---------------------------------------------------------------------------

_METHODOLOGY_SUB_WEIGHTS = {
    "architecture_match":       0.30,
    "optimizer_match":          0.20,
    "loss_match":               0.20,
    "hyperparams_present":      0.15,
    "contribution_implemented": 0.15,
}


def _score_methodology(filtered_text: str, code: str) -> tuple[float, dict]:
    """
    Ask the LLM to score methodology fidelity across 5 sub-dimensions.

    Calls the LLM TWICE (primary key + fallback key) and averages sub-scores
    to reduce single-call variance. If only one call succeeds, uses that result.

    Returns (score_0_to_100, raw_detail_dict).
    Falls back to 0.0 with empty detail if both calls fail.
    """
    methods_text = filtered_text[:4000]
    code_preview = code[:3000]

    prompt = _METHODOLOGY_PROMPT.format(
        methods_text=methods_text,
        code=code_preview,
    )

    results = []  # collect up to 2 successful detail dicts

    for attempt in range(2):
        use_fb = (attempt == 1)
        if use_fb:
            logger.info("Methodology: second call with fallback key for score averaging...")
        try:
            response = _get_llm(use_fallback=use_fb).invoke(prompt)
            raw = response.content.strip()

            if "```" in raw:
                raw = re.sub(r"```[a-z]*", "", raw).replace("```", "").strip()

            detail = json.loads(raw)

            # Validate and clamp sub-scores
            for key in _METHODOLOGY_SUB_WEIGHTS:
                val = detail.get(key, 0.0)
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    val = 0.0
                if val >= 0.75:
                    val = 1.0
                elif val >= 0.25:
                    val = 0.5
                else:
                    val = 0.0
                detail[key] = val

            results.append(detail)

        except json.JSONDecodeError as exc:
            logger.warning("Methodology LLM attempt %d returned invalid JSON: %s", attempt + 1, exc)
        except Exception as exc:
            if "429" in str(exc):
                logger.warning("Methodology attempt %d: rate limited (429)", attempt + 1)
            else:
                logger.warning("Methodology attempt %d failed: %s", attempt + 1, exc)

    if not results:
        return 0.0, {}

    # Average sub-scores across successful calls
    if len(results) == 2:
        averaged = {}
        for key in _METHODOLOGY_SUB_WEIGHTS:
            averaged[key] = round((results[0].get(key, 0.0) + results[1].get(key, 0.0)) / 2, 2)
        # Clamp averaged values back to 0 / 0.5 / 1.0
        for key in _METHODOLOGY_SUB_WEIGHTS:
            v = averaged[key]
            if v >= 0.75:
                averaged[key] = 1.0
            elif v >= 0.25:
                averaged[key] = 0.5
            else:
                averaged[key] = 0.0
        # Use reasoning from the first call
        averaged["reasoning"] = results[0].get("reasoning", "")
        logger.info("Methodology: averaged 2 LLM calls for stability")
        detail = averaged
    else:
        detail = results[0]
        logger.info("Methodology: only 1 LLM call succeeded — no averaging")

    # Weighted sum -> 0-100
    weighted = sum(
        detail[k] * w
        for k, w in _METHODOLOGY_SUB_WEIGHTS.items()
    )
    score = round(min(100.0, weighted * 100.0), 1)

    logger.info(
        "Methodology sub-scores: arch=%.1f opt=%.1f loss=%.1f hp=%.1f contrib=%.1f",
        detail.get("architecture_match", 0),
        detail.get("optimizer_match", 0),
        detail.get("loss_match", 0),
        detail.get("hyperparams_present", 0),
        detail.get("contribution_implemented", 0),
    )
    return score, detail


# ---------------------------------------------------------------------------
# Dimension 3: Results Score
# ---------------------------------------------------------------------------

def _score_results(
    actual: Optional[float],
    expected: Optional[float],
) -> float:
    """
    Score numerical results accuracy using relative difference.

    Uses RELATIVE diff (not absolute) so a 2% miss on a 95% accuracy paper
    is weighted the same as a 2% miss on a 50% accuracy paper.

      rel_diff = |actual - expected| / expected

      Within  2%: 100
      Within  5%:  85
      Within 10%:  70
      Within 20%:  50
      Beyond 20%:  max(0, 30 - rel_diff*100)

    Graceful fallback when no golden JSON:
      If actual accuracy exists but no expected: score based on execution level.
      If neither exists: 0.

    Returns float 0-100.
    """
    if actual is None:
        return 0.0

    if expected is None:
        # No golden baseline -- score by how reasonable the accuracy looks
        if actual >= 90:
            return 80.0
        elif actual >= 75:
            return 65.0
        elif actual >= 60:
            return 50.0
        else:
            return 30.0

    if expected == 0:
        # Avoid division by zero -- fall back to absolute diff capped at 100
        return max(0.0, 100.0 - abs(actual - expected) * 2)

    rel_diff = abs(actual - expected) / expected  # e.g. 0.02 = 2%

    if rel_diff <= 0.02:
        return 100.0
    elif rel_diff <= 0.05:
        return 85.0
    elif rel_diff <= 0.10:
        return 70.0
    elif rel_diff <= 0.20:
        return 50.0
    else:
        return max(0.0, 30.0 - rel_diff * 100.0)


# ---------------------------------------------------------------------------
# Dimension 4: Completeness Score
# ---------------------------------------------------------------------------

def _score_completeness(filtered_text: str, code: str) -> tuple[float, str]:
    """
    Ask the LLM to estimate what % of the paper's contributions are in the code.

    Returns (score_0_to_100, reason_string).
    Falls back to (0.0, "LLM unavailable") on failure.
    """
    paper_preview = filtered_text[:3000]
    code_preview  = code[:3000]

    prompt = _COMPLETENESS_PROMPT.format(
        paper_text=paper_preview,
        code=code_preview,
    )

    for attempt in range(2):
        use_fb = (attempt == 1)
        if use_fb:
            logger.info("Completeness: retrying with fallback API key...")
        try:
            response = _get_llm(use_fallback=use_fb).invoke(prompt)
            raw = response.content.strip()

            if "```" in raw:
                raw = re.sub(r"```[a-z]*", "", raw).replace("```", "").strip()

            data = json.loads(raw)

            score = float(data.get("completeness_score", 0))
            score = min(100.0, max(0.0, score))
            reason = str(data.get("reason", ""))

            logger.info("Completeness score: %.1f -- %s", score, reason[:80])
            return round(score, 1), reason

        except json.JSONDecodeError as exc:
            logger.warning("Completeness LLM returned invalid JSON: %s", exc)
            return 0.0, "LLM returned invalid JSON"
        except Exception as exc:
            if "429" in str(exc) and attempt == 0:
                logger.warning("Completeness: rate limited (429) -- trying fallback key")
                continue
            logger.warning("Completeness scoring failed: %s", exc)
            return 0.0, f"Scoring failed: {exc}"
    return 0.0, "Both API keys rate limited"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_golden(golden_path: str) -> tuple[Optional[float], float]:
    """
    Load expected accuracy and tolerance from golden JSON.

    Returns (expected_accuracy, tolerance).
    Returns (None, TOLERANCE_DEFAULT) if file missing or malformed.
    """
    try:
        with open(golden_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        expected = data.get("expected_accuracy")
        tolerance = float(data.get("tolerance", TOLERANCE_DEFAULT))
        if expected is not None:
            expected = float(expected)
            # Fix common issue: Groq returns 0.99 instead of 99.0
            if expected < 1.0:
                expected = round(expected * 100, 2)
        return expected, tolerance
    except FileNotFoundError:
        logger.info("No golden JSON at %s -- results dimension uses fallback", golden_path)
        return None, TOLERANCE_DEFAULT
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Invalid golden JSON at %s: %s", golden_path, exc)
        return None, TOLERANCE_DEFAULT


def _extract_accuracy(stdout: str) -> Optional[float]:
    """
    Extract the final accuracy value from stdout.

    Tries multiple patterns in order of specificity.
    Returns float percentage (e.g. 98.5) or None if not found.
    """
    if not stdout:
        return None

    patterns = [
        r"[Ff]inal\s+[Aa]ccuracy\s*[:\s=]+([0-9]+\.?[0-9]*)",
        r"[Tt]est\s+[Aa]ccuracy\s*[:\s=]+([0-9]+\.?[0-9]*)",
        r"[Aa]ccuracy\s*[:\s=]+([0-9]+\.?[0-9]*)\s*%",
        r"[Aa]ccuracy\s*[:\s=]+([0-9]+\.?[0-9]*)",
        r"[Aa]cc\s*[:\s=]+([0-9]+\.?[0-9]*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, stdout, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            if value < 1.0:
                value = round(value * 100, 2)
            return round(value, 2)
    return None


def _build_status(
    actual:   Optional[float],
    expected: Optional[float],
    tolerance: float,
    final_score: float,
) -> str:
    """Build a human-readable status string."""
    if actual is None:
        return "FAIL -- No accuracy reported"
    if expected is None:
        return f"PASS -- Executed successfully (no baseline to compare)"
    diff = abs(actual - expected)
    if diff <= tolerance:
        return "PASS"
    elif final_score >= 60:
        return f"PARTIAL -- Outside tolerance (diff={diff:.2f}%)"
    else:
        return f"FAIL -- Poor reproduction (diff={diff:.2f}%)"


# ---------------------------------------------------------------------------
# Backward-compatibility shim
# ---------------------------------------------------------------------------

def run_test(
    paper_id:    str,
    stdout:      str,
    stderr:      str,
    golden_path: str,
    filtered_text: str = "",
    code:          str = "",
) -> dict:
    """
    Drop-in replacement for tester_agent.run_test().

    Accepts the same positional arguments as the original function and
    returns a dict with all original keys plus new dimension_scores keys.

    To use without touching pipeline.py:
      In pipeline.py Step 6, change only the import:
        # from agents.tester_agent import run_test
        from agents.scoring_engine import run_test
    """
    return score_paper(
        paper_id=paper_id,
        stdout=stdout,
        stderr=stderr,
        golden_path=golden_path,
        filtered_text=filtered_text,
        code=code,
    ).to_dict()