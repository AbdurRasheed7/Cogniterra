import json
import os
import re
from datetime import datetime
from config import REPORTS_DIR, TOLERANCE_DEFAULT

def extract_accuracy(output_text):
    """Extract accuracy number from code output - more patterns"""
    patterns = [
        r'[Tt]est [Aa]ccuracy[:\s=]+([0-9]+\.?[0-9]*)',
        r'[Ff]inal [Aa]ccuracy[:\s=]+([0-9]+\.?[0-9]*)',
        r'[Aa]ccuracy[:\s=]+([0-9]+\.?[0-9]*)',
        r'[Aa]cc[:\s=]+([0-9]+\.?[0-9]*)',
        r'[Tt]op-1 [Aa]ccuracy[:\s=]+([0-9]+\.?[0-9]*)',
        r'[Aa]ccuracy\s*=\s*([0-9]+\.?[0-9]*)',
        r'accuracy\s*[:=]\s*([0-9]+\.?[0-9]*)',
        r'Accuracy:\s*([0-9]+\.?[0-9]*)%',
    ]
    for pattern in patterns:
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            if value < 1.0:
                value *= 100
            return round(value, 2)
    return None

def calculate_score(actual, expected, tolerance=2.0):
    """Calculate reproducibility score out of 100"""
    if actual is None:
        return 0
    difference = abs(actual - expected)
    if difference <= tolerance:
        # Perfect or close: high score
        score = 100 - (difference / tolerance * 20)
    else:
        # Larger difference: steeper drop
        score = max(0, 80 - (difference - tolerance) * 15)
    return round(score, 1)

def run_test(paper_id, stdout, stderr, expected_json_path):
    """Compare results and generate test report"""

    try:
        # Load expected values
        with open(expected_json_path, 'r') as f:
            expected = json.load(f)
    except FileNotFoundError:
        print(f"   Warning: No golden JSON found for {paper_id}. Using defaults.")
        expected = {"expected_accuracy": None, "tolerance": TOLERANCE_DEFAULT}
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON: {expected_json_path}")
        expected = {"expected_accuracy": None, "tolerance": TOLERANCE_DEFAULT}

    # Extract actual accuracy
    actual_accuracy = extract_accuracy(stdout)

    expected_accuracy = expected.get('expected_accuracy')
    tolerance = expected.get('tolerance', 2.0)

    # Calculate score
    if expected_accuracy is not None:
        # Golden JSON exists — score against expected
        score = calculate_score(actual_accuracy, expected_accuracy, tolerance)
    elif actual_accuracy is not None:
        # No golden JSON but code ran and produced accuracy — give execution score
        # Base score: 60 for running, up to 85 based on accuracy level
        if actual_accuracy >= 90:
            score = 85
        elif actual_accuracy >= 80:
            score = 75
        elif actual_accuracy >= 70:
            score = 65
        else:
            score = 60
    else:
        # No golden JSON and no accuracy — complete failure
        score = 0

    # Determine status
    if actual_accuracy is None:
        status = "❌ FAIL - No accuracy reported"
    elif expected_accuracy is None:
        status = f"✅ PASS - Executed successfully (no baseline to compare)"
    elif abs(actual_accuracy - expected_accuracy) <= tolerance:
        status = "✅ PASS"
    else:
        status = "⚠️ PARTIAL - Outside tolerance"

    # Build result — ALL keys always present with safe defaults
    result = {
        "paper_id":             paper_id,
        "timestamp":            datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "expected_accuracy":    expected_accuracy,
        "actual_accuracy":      actual_accuracy,
        "difference":           round(abs((actual_accuracy or 0) - (expected_accuracy or 0)), 2)
                                if actual_accuracy is not None and expected_accuracy is not None else None,
        "tolerance":            tolerance,
        "status":               status,
        "reproducibility_score": score,
        "has_errors":           bool(stderr and ("Error" in stderr or "Exception" in stderr)),
        "stderr_preview":       stderr[:200] if stderr else None
    }

    return result

"""
The new report adds:
  - 4-dimension score cards with color-coded progress bars
  - Methodology sub-scores breakdown
  - Completeness reason
  - Weighted formula display
  - Source type badge (latex / pdf / html)
"""


def generate_html_report(result: dict, code_path: str) -> str:
    """Generate enhanced HTML report with 4-dimension scoring visualization."""
    try:
        from config import REPORTS_DIR
    except ImportError:
        REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Read generated code
    code_content = "Code file not found"
    if os.path.exists(code_path):
        try:
            with open(code_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
        except Exception as e:
            code_content = f"Error reading code: {e}"

    # Safe key access
    paper_id          = result.get('paper_id',             'Unknown')
    timestamp         = result.get('timestamp',            'N/A')
    score             = result.get('reproducibility_score', 0)
    status            = result.get('status',               'FAILED')
    expected_accuracy = result.get('expected_accuracy',    None)
    actual_accuracy   = result.get('actual_accuracy',      None)
    difference        = result.get('difference',           None)
    tolerance         = result.get('tolerance',            2.0)
    has_errors        = result.get('has_errors',           False)

    # 4-dimension scores
    dims              = result.get('dimension_scores', {})
    e_sc              = dims.get('execution',    0)
    m_sc              = dims.get('methodology',  0)
    r_sc              = dims.get('results',      0)
    c_sc              = dims.get('completeness', 0)

    # Methodology detail
    mdetail           = result.get('methodology_detail', {})
    arch              = mdetail.get('architecture_match',       0)
    opt               = mdetail.get('optimizer_match',          0)
    loss              = mdetail.get('loss_match',               0)
    hp                = mdetail.get('hyperparams_present',      0)
    contrib           = mdetail.get('contribution_implemented', 0)
    reasoning         = mdetail.get('reasoning',                '')
    comp_reason       = result.get('completeness_reason',       '')

    # Score colors
    def score_color(s):
        if s >= 75: return '#22c55e'
        if s >= 50: return '#f59e0b'
        return '#ef4444'

    main_color  = score_color(score)
    status_cls  = 'pass' if 'PASS' in status else ('fail' if 'FAIL' in status else 'partial')

    def sub_label(v):
        if v >= 0.9: return ('1.0', '#22c55e')
        if v >= 0.4: return ('0.5', '#f59e0b')
        return ('0.0', '#ef4444')

    arch_l,    arch_c    = sub_label(arch)
    opt_l,     opt_c     = sub_label(opt)
    loss_l,    loss_c    = sub_label(loss)
    hp_l,      hp_c      = sub_label(hp)
    contrib_l, contrib_c = sub_label(contrib)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cogniterra Report — {paper_id}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
        :root {{
            --bg: #0f1117; --surf: #1a1d27; --surf2: #22263a;
            --border: #2e3248; --accent: #f97316;
            --text: #f1f5f9; --text2: #94a3b8; --text3: #475569;
            --success: #22c55e; --error: #ef4444; --warn: #f59e0b;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; padding: 2rem; max-width: 1100px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #1a1d27, #22263a); border: 1px solid var(--border); border-left: 4px solid var(--accent); border-radius: 14px; padding: 2rem 2.5rem; margin-bottom: 1.5rem; display: flex; justify-content: space-between; align-items: center; }}
        .header-left h1 {{ font-family: 'Sora', sans-serif; font-size: 1.6rem; font-weight: 800; letter-spacing: -0.03em; }}
        .header-left h1 span {{ color: var(--accent); }}
        .header-left p {{ font-size: 0.78rem; color: var(--text3); margin-top: 4px; font-family: 'JetBrains Mono', monospace; }}
        .big-score {{ font-family: 'Sora', sans-serif; font-size: 4rem; font-weight: 800; color: {main_color}; line-height: 1; }}
        .big-score-lbl {{ font-size: 0.62rem; color: var(--text3); text-transform: uppercase; letter-spacing: 0.12em; margin-top: 4px; text-align: right; }}
        .status-badge {{ display: inline-block; padding: 5px 14px; border-radius: 7px; font-size: 0.72rem; font-weight: 700; margin-top: 8px; }}
        .pass {{ background: rgba(34,197,94,0.12); color: var(--success); border: 1px solid rgba(34,197,94,0.3); }}
        .fail {{ background: rgba(239,68,68,0.12); color: var(--error); border: 1px solid rgba(239,68,68,0.3); }}
        .partial {{ background: rgba(245,158,11,0.12); color: var(--warn); border: 1px solid rgba(245,158,11,0.3); }}
        .section {{ background: var(--surf); border: 1px solid var(--border); border-radius: 12px; padding: 1.4rem 1.6rem; margin-bottom: 1rem; }}
        .section-title {{ font-size: 0.6rem; font-weight: 700; color: var(--text3); text-transform: uppercase; letter-spacing: 0.13em; margin-bottom: 1rem; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 1rem; }}
        .metric {{ background: var(--surf2); border: 1px solid var(--border); border-radius: 10px; padding: 0.9rem 1rem; }}
        .metric-lbl {{ font-size: 0.58rem; color: var(--text3); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 5px; font-weight: 700; }}
        .metric-val {{ font-family: 'Sora', sans-serif; font-size: 1.5rem; font-weight: 800; color: var(--text); }}
        .metric-unt {{ font-size: 0.7rem; color: var(--text2); }}
        .dim-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 1.2rem; }}
        .dim-card {{ background: var(--surf2); border: 1px solid var(--border); border-radius: 10px; padding: 1rem; position: relative; overflow: hidden; }}
        .dim-card::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; }}
        .dim-exec::before {{ background: linear-gradient(90deg, #22c55e, #16a34a); }}
        .dim-meth::before {{ background: linear-gradient(90deg, #f97316, #ea580c); }}
        .dim-res::before  {{ background: linear-gradient(90deg, #3b82f6, #2563eb); }}
        .dim-comp::before {{ background: linear-gradient(90deg, #a855f7, #9333ea); }}
        .dim-label {{ font-size: 0.58rem; font-weight: 700; color: var(--text3); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px; }}
        .dim-score {{ font-family: 'Sora', sans-serif; font-size: 1.8rem; font-weight: 800; line-height: 1; }}
        .dim-weight {{ font-size: 0.6rem; color: var(--text3); margin-top: 4px; }}
        .bar-wrap {{ background: #1a1d27; border-radius: 3px; height: 4px; margin-top: 8px; overflow: hidden; }}
        .bar {{ height: 4px; border-radius: 3px; }}
        .bar-exec {{ background: linear-gradient(90deg, #22c55e, #16a34a); }}
        .bar-meth {{ background: linear-gradient(90deg, #f97316, #ea580c); }}
        .bar-res  {{ background: linear-gradient(90deg, #3b82f6, #2563eb); }}
        .bar-comp {{ background: linear-gradient(90deg, #a855f7, #9333ea); }}
        .sub-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin: 1rem 0 0.5rem; }}
        .sub-card {{ background: #0f1117; border: 1px solid var(--border); border-radius: 8px; padding: 0.65rem 0.7rem; text-align: center; }}
        .sub-lbl {{ font-size: 0.54rem; font-weight: 700; color: var(--text3); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 5px; line-height: 1.3; }}
        .sub-val {{ font-family: 'Sora', sans-serif; font-size: 1.1rem; font-weight: 800; }}
        .formula {{ background: #0f1117; border: 1px solid var(--border); border-radius: 8px; padding: 0.8rem 1rem; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--text2); margin-top: 1rem; }}
        .formula span {{ color: var(--accent); font-weight: 700; }}
        .row {{ display: flex; justify-content: space-between; align-items: center; padding: 7px 0; border-bottom: 1px solid var(--border); font-size: 0.78rem; }}
        .row:last-child {{ border-bottom: none; }}
        .row-lbl {{ color: var(--text2); }}
        .row-val {{ font-weight: 600; color: var(--text); }}
        .reasoning {{ font-size: 0.72rem; color: var(--text3); font-style: italic; margin-top: 0.5rem; padding: 0.6rem 0.8rem; background: #0f1117; border-radius: 6px; border-left: 2px solid var(--accent); }}
        .code-wrap {{ background: #0a0d14; border: 1px solid var(--border); border-radius: 10px; padding: 1.2rem; overflow-x: auto; max-height: 500px; overflow-y: auto; }}
        code {{ font-family: 'JetBrains Mono', monospace; font-size: 0.73rem; color: #c3d2df; white-space: pre-wrap; }}
        .footer {{ text-align: center; color: var(--text3); font-size: 0.68rem; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--border); }}
    </style>
</head>
<body>

<div class="header">
    <div class="header-left">
        <h1>Cogni<span>terra</span> Report</h1>
        <p>Paper: {paper_id} &nbsp;|&nbsp; {timestamp}</p>
    </div>
    <div style="text-align:right">
        <div class="big-score">{score}</div>
        <div class="big-score-lbl">/ 100</div>
        <div class="status-badge {status_cls}">{status}</div>
    </div>
</div>

<!-- Accuracy metrics -->
<div class="section">
    <div class="section-title">Results Comparison</div>
    <div class="metrics-grid">
        <div class="metric">
            <div class="metric-lbl">Expected</div>
            <div class="metric-val">{f"{expected_accuracy:.1f}" if expected_accuracy is not None else "N/A"}<span class="metric-unt">%</span></div>
        </div>
        <div class="metric">
            <div class="metric-lbl">Achieved</div>
            <div class="metric-val">{f"{actual_accuracy:.1f}" if actual_accuracy is not None else "N/A"}<span class="metric-unt">%</span></div>
        </div>
        <div class="metric">
            <div class="metric-lbl">Difference</div>
            <div class="metric-val">{f"{difference:.2f}" if difference is not None else "N/A"}<span class="metric-unt">%</span></div>
        </div>
        <div class="metric">
            <div class="metric-lbl">Tolerance</div>
            <div class="metric-val">±{tolerance}<span class="metric-unt">%</span></div>
        </div>
    </div>
    <div class="row"><span class="row-lbl">Runtime Errors</span><span class="row-val" style="color:{'#ef4444' if has_errors else '#22c55e'}">{'Yes' if has_errors else 'None'}</span></div>
</div>

<!-- 4-Dimension scores -->
<div class="section">
    <div class="section-title">4-Dimension Scoring Breakdown</div>
    <div class="dim-grid">
        <div class="dim-card dim-exec">
            <div class="dim-label">Execution</div>
            <div class="dim-score" style="color:{score_color(e_sc)}">{e_sc:.0f}</div>
            <div class="dim-weight">weight 0.20</div>
            <div class="bar-wrap"><div class="bar bar-exec" style="width:{e_sc}%"></div></div>
        </div>
        <div class="dim-card dim-meth">
            <div class="dim-label">Methodology</div>
            <div class="dim-score" style="color:{score_color(m_sc)}">{m_sc:.0f}</div>
            <div class="dim-weight">weight 0.35</div>
            <div class="bar-wrap"><div class="bar bar-meth" style="width:{m_sc}%"></div></div>
        </div>
        <div class="dim-card dim-res">
            <div class="dim-label">Results</div>
            <div class="dim-score" style="color:{score_color(r_sc)}">{r_sc:.0f}</div>
            <div class="dim-weight">weight 0.30</div>
            <div class="bar-wrap"><div class="bar bar-res" style="width:{r_sc}%"></div></div>
        </div>
        <div class="dim-card dim-comp">
            <div class="dim-label">Completeness</div>
            <div class="dim-score" style="color:{score_color(c_sc)}">{c_sc:.0f}</div>
            <div class="dim-weight">weight 0.15</div>
            <div class="bar-wrap"><div class="bar bar-comp" style="width:{c_sc}%"></div></div>
        </div>
    </div>

    <!-- Methodology sub-scores -->
    <div class="section-title" style="margin-top:1rem">Methodology Sub-Scores</div>
    <div class="sub-grid">
        <div class="sub-card">
            <div class="sub-lbl">Architecture</div>
            <div class="sub-val" style="color:{arch_c}">{arch_l}</div>
        </div>
        <div class="sub-card">
            <div class="sub-lbl">Optimizer</div>
            <div class="sub-val" style="color:{opt_c}">{opt_l}</div>
        </div>
        <div class="sub-card">
            <div class="sub-lbl">Loss Fn</div>
            <div class="sub-val" style="color:{loss_c}">{loss_l}</div>
        </div>
        <div class="sub-card">
            <div class="sub-lbl">Hyperparams</div>
            <div class="sub-val" style="color:{hp_c}">{hp_l}</div>
        </div>
        <div class="sub-card">
            <div class="sub-lbl">Contribution</div>
            <div class="sub-val" style="color:{contrib_c}">{contrib_l}</div>
        </div>
    </div>

    {f'<div class="reasoning">{reasoning}</div>' if reasoning else ''}
    {f'<div class="reasoning" style="border-left-color:#a855f7;margin-top:0.5rem"><b style="color:#c4b5fd">Completeness:</b> {comp_reason}</div>' if comp_reason else ''}

    <div class="formula">
        Final = <span>0.20</span>&times;{e_sc:.0f} + <span>0.35</span>&times;{m_sc:.0f} + <span>0.30</span>&times;{r_sc:.0f} + <span>0.15</span>&times;{c_sc:.0f} = <span>{score}</span> / 100
    </div>
</div>

<!-- Generated code -->
<div class="section">
    <div class="section-title">Generated Code</div>
    <div class="code-wrap"><code>{code_content.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')}</code></div>
</div>

<div class="footer">
    Generated by Cogniterra &nbsp;|&nbsp; Powered by Groq + Llama 3.3 70B &nbsp;|&nbsp; {timestamp}
</div>

</body>
</html>"""

    report_path = os.path.join(REPORTS_DIR, f"{paper_id}_report.html")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return report_path