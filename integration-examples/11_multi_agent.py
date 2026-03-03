"""
11_multi_agent.py — Multi-step agentic pipeline with budget tracking
No API key needed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llmauditor import auditor

def main():
    auditor.clear_history()

    # ── TEST 1: Single agent, multiple steps ──
    print("=" * 60)
    print("TEST 1: Single agent — 3-step pipeline")
    print("=" * 60)
    auditor.set_budget(0.50)
    auditor.set_alert_mode(True)

    auditor.execute(
        model="gpt-4o", input_tokens=500, output_tokens=300,
        raw_response="Plan: 1) Research topic 2) Draft outline 3) Write content"
    )
    auditor.execute(
        model="gpt-4o", input_tokens=800, output_tokens=400,
        raw_response="Research findings: The topic covers three main areas..."
    )
    auditor.execute(
        model="gpt-4o", input_tokens=1200, output_tokens=600,
        raw_response="Final article: Introduction to the topic..."
    )

    status = auditor.get_budget_status()
    assert status["executions"] == 3
    assert status["cumulative_cost"] > 0
    print(f"  Total cost: ${status['cumulative_cost']:.4f}")
    print(f"  Remaining: ${status['remaining']:.4f}")
    print(f"  Executions: {status['executions']}")

    # ── TEST 2: Multi-agent (4 different models) ──
    print("\n" + "=" * 60)
    print("TEST 2: Multi-agent — Researcher → Writer → Reviewer → Editor")
    print("=" * 60)
    auditor.clear_history()
    auditor.set_budget(1.00)
    auditor.set_alert_mode(True)
    auditor.start_evaluation("Multi-Agent Pipeline", version="1.0.0")

    # Agent 1: Researcher (GPT-4o)
    auditor.execute(
        model="gpt-4o", input_tokens=200, output_tokens=500,
        raw_response="Research: Found 5 key points about AI safety..."
    )
    # Agent 2: Writer (Claude)
    auditor.execute(
        model="claude-3.5-sonnet", input_tokens=600, output_tokens=1000,
        raw_response="Article draft: AI Safety in 2025..."
    )
    # Agent 3: Reviewer (Gemini)
    auditor.execute(
        model="gemini-2.0-flash", input_tokens=1200, output_tokens=300,
        raw_response="Review: The article covers key points well. Suggest adding..."
    )
    # Agent 4: Editor (GPT-4o-mini)
    auditor.execute(
        model="gpt-4o-mini", input_tokens=1500, output_tokens=1200,
        raw_response="Final edited article with comprehensive analysis..."
    )

    # History check
    h = auditor.history()
    assert len(h) == 4
    for i, r in enumerate(h):
        print(f"  Step {i+1}: {r.model_name} — ${r.estimated_cost:.6f}")

    # Budget status
    status2 = auditor.get_budget_status()
    print(f"\n  Total: ${status2['cumulative_cost']:.6f} / ${status2['budget_limit']:.2f}")

    # Certification
    auditor.end_evaluation()
    report = auditor.generate_evaluation_report()
    assert report.score.overall > 0
    assert len(report.metrics.models_used) == 4
    print(f"  Certification: {report.score.level} ({report.score.overall:.1f}/100)")
    print(f"  Models: {report.metrics.models_used}")

    auditor.clear_history()
    print("\n✅ ALL MULTI-AGENT TESTS PASSED")

if __name__ == "__main__":
    main()
