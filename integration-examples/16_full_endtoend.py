"""
16_full_endtoend.py — Complete pipeline: config → eval → cert → export
No API key needed. Mirrors the DOCUMENTATION.md "Complete End-to-End Example".
"""
import sys, os, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llmauditor import auditor

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports_endtoend_test")

def main():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    auditor.clear_history()

    # ── Configuration ──
    print("=" * 60)
    print("FULL END-TO-END PIPELINE")
    print("=" * 60)

    auditor.set_budget(0.50)
    auditor.set_alert_mode(True)
    auditor.set_pricing_table({
        "llama-3.3-70b-versatile": {"input": 0.00059, "output": 0.00079}
    })

    # ── Start evaluation ──
    auditor.start_evaluation("My Production App", version="3.0.0")

    # ── 5 executions across different models ──
    executions = [
        ("gpt-4o", 250, 180,
         "Detailed analysis of market trends shows growth in AI sector.",
         "Analyze current market trends in AI"),
        ("claude-3.5-sonnet", 300, 200,
         "The key findings suggest three primary factors driving adoption.",
         "Summarize key findings from the research"),
        ("gemini-2.0-flash", 150, 100,
         "Based on the data, the recommendation is to proceed with phase 2.",
         "What do you recommend based on this data?"),
        ("gpt-4o-mini", 200, 150,
         "The comparison reveals that Option A is 23% more cost-effective.",
         "Compare Option A vs Option B"),
        ("llama-3.3-70b-versatile", 180, 120,
         "In conclusion, the strategy should focus on three pillars.",
         "Draft a conclusion for the strategy document"),
    ]

    for model, in_tok, out_tok, response, question in executions:
        auditor.execute(
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            raw_response=response,
            input_text=question
        )

    # ── End evaluation and certify ──
    auditor.end_evaluation()
    report = auditor.generate_evaluation_report()

    # ── Display ──
    report.display()

    # ── Assertions ──
    assert report.score.overall > 0
    assert report.score.level in ["Platinum", "Gold", "Silver", "Conditional Pass", "Fail"]
    assert len(report.execution_reports) == 5
    assert len(report.metrics.models_used) == 5

    print(f"\n  Certification: {report.score.level} ({report.score.overall:.1f}/100)")
    print(f"  Models: {', '.join(report.metrics.models_used)}")

    # ── Export all formats ──
    paths = report.export_all(output_dir=OUT_DIR)
    for fmt, path in paths.items():
        assert os.path.exists(path)
        size = os.path.getsize(path)
        print(f"  {fmt.upper()}: {os.path.basename(path)} ({size:,} bytes)")

    # ── Budget status ──
    status = auditor.get_budget_status()
    print(f"\n  Budget: ${status['cumulative_cost']:.4f} / ${status['budget_limit']:.2f}")
    print(f"  Remaining: ${status['remaining']:.4f}")
    print(f"  Executions: {status['executions']}")

    # ── to_dict() ──
    d = report.to_dict()
    assert "session" in d
    assert "metrics" in d
    assert "score" in d
    assert "suggestions" in d
    print(f"  Dict keys: {list(d.keys())}")

    # Cleanup
    auditor.clear_history()
    shutil.rmtree(OUT_DIR)
    print(f"\n  (Cleaned up {OUT_DIR})")
    print("\n✅ FULL END-TO-END TEST PASSED")

if __name__ == "__main__":
    main()
