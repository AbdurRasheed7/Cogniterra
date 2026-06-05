import sys
import io
import os
import re
import argparse
from datetime import datetime

# Fix Windows emoji encoding issue with CrewAI
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(paper_id, run_crew=False, force_domain=None, fast_mode=False):
    print(f"\n{'='*60}")
    print(f"🚀 Starting Reproducibility Pipeline for paper: {paper_id}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    if fast_mode:
        print(f"⚡ Fast mode enabled — fewer epochs + dataset subset")
    print(f"{'='*60}\n")

    from agents.latex_ingestion import parse_paper, get_last_source_type
    from agents.rag_agent import get_relevant_context
    from agents.coder_agent import generate_code
    from agents.domain_detector import detect_domain, format_domain_report, get_code_domain
    from agents.tester_agent import generate_html_report
    from agents.scoring_engine import run_test
    from agents.hallucination_agent import analyze_hallucinations, format_hallucination_report   # ← Fixed
    from agents.debugger_agent import run_with_debug
    from agents.crew_agents import run_crew_analysis
    from data.download_movielens import download_movielens
    from utils.docker_helper import run_code_in_docker


    # ── Step 1: Fetch & parse paper ───────────────────────────────────────────
    print("Step 1: Fetching & parsing paper...")
    try:
        filtered_text = parse_paper(paper_id)
        print(f"   Source type: {get_last_source_type()}")
        if not filtered_text or not filtered_text.strip():
            print("❌ Failed to extract paper text. Exiting.")
            return
        print(f"   Extracted {len(filtered_text)} chars")
    except Exception as e:
        print(f"❌ Parser error: {e}")
        return

    # ── Step 2: Detect domain ─────────────────────────────────────────────────
    print("\nStep 2: Detecting domain...")
    detection = detect_domain(filtered_text, paper_id=paper_id)
    domain      = detection["domain"]
    print(format_domain_report(detection))

    # Manual override from CLI
    if force_domain:
        domain = force_domain
        print(f"   ⚠️  Domain overridden to: {domain}")

    final_domain = get_code_domain(detection) if not force_domain else force_domain
    print(f"   Final domain for code gen: {final_domain}")

    # ── Step 3: Build RAG context ─────────────────────────────────────────────
    print("\nStep 3: Building RAG context...")
    try:
        rag_context = get_relevant_context(filtered_text, domain=final_domain)
        print(f"   Retrieved {len(rag_context)} chars of relevant context")
    except Exception as e:
        print(f"⚠️  RAG failed: {e} — using full text")
        rag_context = filtered_text[:3000]

    # ── Step 4: Generate code ─────────────────────────────────────────────────
    print("\nStep 4: Generating code with Groq...")
    try:
        code = generate_code(
            rag_context,
            domain=final_domain,
            paper_id=paper_id,
            fast_mode=fast_mode
        )
    except Exception as e:
        print(f"❌ Code generation failed: {e}")
        return

    # Preview
    print(f"\n--- GENERATED CODE PREVIEW (first 500 chars) ---\n")
    print(code[:500])
    print("...")

    # ── Step 5: Save & execute in Docker ──────────────────────────────────────
    print("\nStep 5: Saving & executing code in Docker...")
    code_dir  = os.path.join(os.path.dirname(__file__), "generated_code")
    os.makedirs(code_dir, exist_ok=True)
    code_path = os.path.join(code_dir, f"{paper_id}_solution.py")
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"💾 Code saved: {code_path}")

    # Try Docker first, fall back to local debugger
    success, stdout, stderr = run_code_in_docker(code, paper_id)

    if not success:
        print(f"Docker FAILED — falling back to local debugger...")
        try:
            stdout, stderr, code, attempts = run_with_debug(code, code_path, domain=final_domain)
            success = True
            # Save fixed code
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code)
        except Exception as e:
            print(f"❌ Debugger also failed: {e}")
            stdout = ""
            stderr = str(e)
            success = False
    else:
        print(f"Docker SUCCESS! Logs preview:")
        print(stdout[:500] if stdout else "(no output)")

    # ── Step 6: Reproducibility check ────────────────────────────────────────
    print("\nStep 6: Reproducibility check...")
    golden_dir  = os.path.join(os.path.dirname(__file__), "tests", "golden")
    os.makedirs(golden_dir, exist_ok=True)
    golden_path = os.path.join(golden_dir, f"{paper_id}_expected.json")

    test_result = run_test(
    paper_id, stdout, stderr, golden_path,
    filtered_text=filtered_text,   # already exists from Step 1
    code=code,                     # already exists from Step 4
    )

    print(f"   Expected Accuracy : {test_result.get('expected_accuracy')}%")
    print(f"   Actual Accuracy   : {test_result.get('actual_accuracy')}%")
    print(f"   Difference        : {test_result.get('difference')}%")
    print(f"   🏆 Reproducibility Score: {test_result.get('reproducibility_score')}/100")
    print(f"   Status: {test_result.get('status')}")

    # ── Step 7: Hallucination check ───────────────────────────────────────────
    print("\nStep 7: Hallucination check...")
    try:
        hallucination = analyze_hallucinations(code, filtered_text)
        hall_score    = hallucination.get("hallucination_score", 0)
        from_paper    = hallucination.get("total_from_paper", 0)
        assumed       = hallucination.get("total_assumptions", 0)
        print(f"\n--- HALLUCINATION ANALYSIS ---")
        print(f"🧠 Hallucination Score: {hall_score}/100 (higher = better)")
        print(f"📄 From Paper: {from_paper} values")
        print(f"⚠️  Assumptions: {assumed} values")
        print(f"📊 Summary: {from_paper} values from paper, {assumed} assumptions made (weighted score)")
        if hallucination.get("assumptions"):
            print(f"\n⚠️  AI ASSUMPTIONS (not clearly in paper):")
            for a in hallucination["assumptions"][:10]:
                print(f"  - {a.get('detail', '')}: {a.get('severity','MEDIUM')} -- {a.get('message','')}")
        if hallucination.get("from_paper"):
            print(f"\n✅ VALUES FROM PAPER:")
            for v in hallucination["from_paper"][:10]:
                print(f"  - {v.get('detail', '')}: {v.get('message','')}")
    except Exception as e:
        print(f"⚠️  Hallucination check failed: {e}")
        hallucination = {}

    # ── Step 8: Generate HTML report ─────────────────────────────────────────
    print("\nStep 8: Generating HTML report...")
    try:
        report_path = generate_html_report(test_result, code_path)
        print(f"   Report saved: {report_path}")
        print(f"   Open in browser: file://{report_path}")
    except Exception as e:
        print(f"⚠️  Report generation failed: {e}")

    # ── Step 9: CrewAI (optional) ─────────────────────────────────────────────
    if run_crew:
        print("\nStep 9: Running CrewAI multi-agent analysis...")
        try:
            from agents.crew_agents import run_crew_analysis
            crew_result = run_crew_analysis(
                paper_id, filtered_text, rag_context, code,
                stdout, stderr, test_result, hallucination
            )
        except Exception as e:
            print(f"❌ CrewAI failed: {e}")
    else:
        print("\n⏭️ CrewAI skipped (use --crew to enable)")

    print(f"\n🏁 Pipeline complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Paper Reproducibility Pipeline")
    parser.add_argument("--paper",  required=True, help="ArXiv paper ID (e.g. 1512.03385)")
    parser.add_argument("--crew",   action="store_true", help="Enable CrewAI multi-agent analysis")
    parser.add_argument("--domain", default=None,
                        choices=["ml","nlp","recommendation","rl","graph","algorithm"],
                        help="Override domain detection")
    parser.add_argument("--fast",   action="store_true",
                        help="Fast mode: fewer epochs + dataset subset (5-10x faster, ~2-5% accuracy drop)")
    args = parser.parse_args()

    main(
        paper_id=args.paper,
        run_crew=args.crew,
        force_domain=args.domain,
        fast_mode=args.fast
    )