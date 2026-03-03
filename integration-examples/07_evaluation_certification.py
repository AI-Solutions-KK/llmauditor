"""
07_evaluation_certification.py — Full evaluation session + certification scoring
No API key needed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llmauditor import auditor

def main():
    auditor.clear_history()

    # ── TEST 1: Basic evaluation session ──
    print("=" * 60)
    print("TEST 1: Evaluation session + certification")
    print("=" * 60)
    auditor.start_evaluation("Integration Test App", version="1.0.0")

    for i in range(5):
        auditor.execute(
            model="gpt-4o",
            input_tokens=150 + i * 20,
            output_tokens=80 + i * 10,
            raw_response=f"Detailed response number {i+1} with enough content for analysis.",
            execution_time=0.5 + i * 0.1,
            input_text=f"Test question number {i+1}"
        )

    auditor.end_evaluation()
    report = auditor.generate_evaluation_report()
    report.display()

    assert report.score.overall > 0
    assert report.score.level in ["Platinum", "Gold", "Silver", "Conditional Pass", "Fail"]
    assert len(report.score.subscores) > 0
    assert len(report.execution_reports) == 5

    print(f"\n  Score: {report.score.overall:.1f}/100")
    print(f"  Level: {report.score.level} {report.score.level_emoji}")
    print(f"  Subscores: {report.score.subscores}")
    print(f"  Executions: {len(report.execution_reports)}")

    # ── TEST 2: Multi-model evaluation ──
    print("\n" + "=" * 60)
    print("TEST 2: Multi-model evaluation")
    print("=" * 60)
    auditor.clear_history()
    auditor.start_evaluation("Multi-Model App", version="2.0.0")

    models = ["gpt-4o", "claude-3.5-sonnet", "gemini-2.0-flash", "gpt-4o-mini"]
    for m in models:
        auditor.execute(
            model=m, input_tokens=200, output_tokens=100,
            raw_response=f"Response from {m} with detailed analysis content here.",
            input_text=f"Question for {m}"
        )

    auditor.end_evaluation()
    report2 = auditor.generate_evaluation_report()

    assert len(report2.metrics.models_used) == len(models)
    print(f"  Models used: {report2.metrics.models_used}")
    print(f"  Score: {report2.score.overall:.1f}/100 — {report2.score.level}")

    # ── TEST 3: Custom thresholds ──
    print("\n" + "=" * 60)
    print("TEST 3: Custom certification thresholds")
    print("=" * 60)
    auditor.clear_history()
    auditor.set_certification_thresholds(
        weights={
            "stability": 0.15,
            "factual_reliability": 0.30,
            "governance_compliance": 0.25,
            "cost_predictability": 0.10,
            "risk_profile": 0.20,
        },
        levels={
            "Platinum": 95,
            "Gold": 85,
            "Silver": 75,
            "Conditional Pass": 65,
        }
    )

    auditor.start_evaluation("Custom Threshold App", version="3.0.0")
    for i in range(5):
        auditor.execute(
            model="gpt-4o", input_tokens=200, output_tokens=100,
            raw_response=f"High quality response {i+1} with substantial content for evaluation."
        )
    auditor.end_evaluation()
    report3 = auditor.generate_evaluation_report()

    assert report3.score.overall > 0
    print(f"  Score: {report3.score.overall:.1f}/100 — {report3.score.level}")
    print(f"  Weights: {report3.score.weights}")

    # ── TEST 4: to_dict() serialization ──
    print("\n" + "=" * 60)
    print("TEST 4: EvaluationReport.to_dict()")
    print("=" * 60)
    d = report.to_dict()
    assert "session" in d
    assert "metrics" in d
    assert "score" in d
    print(f"  ✓ Dict keys: {list(d.keys())}")

    auditor.clear_history()
    print("\n✅ ALL EVALUATION/CERTIFICATION TESTS PASSED")

if __name__ == "__main__":
    main()
