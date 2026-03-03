"""
15_error_safety.py — Error isolation, fallbacks, warnings, decorator safety
No API key needed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llmauditor import auditor

def main():
    auditor.clear_history()

    # ── TEST 1: Normal execution (baseline) ──
    print("=" * 60)
    print("TEST 1: Normal execution — no errors")
    print("=" * 60)
    report = auditor.execute(
        model="gpt-4o",
        input_tokens=100,
        output_tokens=50,
        raw_response="Normal response with sufficient content."
    )
    assert report is not None
    assert report.total_tokens == 150
    print(f"  ✓ Report OK: {report.model_name}, ${report.estimated_cost:.6f}")

    # ── TEST 2: Edge case — zero tokens ──
    print("\n" + "=" * 60)
    print("TEST 2: Edge case — zero tokens")
    print("=" * 60)
    report2 = auditor.execute(
        model="gpt-4o", input_tokens=0, output_tokens=0,
        raw_response=""
    )
    assert report2 is not None
    assert report2.total_tokens == 0
    print(f"  ✓ Zero tokens handled: cost=${report2.estimated_cost:.6f}")

    # ── TEST 3: Edge case — very large tokens ──
    print("\n" + "=" * 60)
    print("TEST 3: Edge case — very large tokens")
    print("=" * 60)
    report3 = auditor.execute(
        model="gpt-4o", input_tokens=1000000, output_tokens=500000,
        raw_response="Large token test."
    )
    assert report3 is not None
    assert report3.total_tokens == 1500000
    print(f"  ✓ Large tokens: {report3.total_tokens:,} tokens, ${report3.estimated_cost:.4f}")

    # ── TEST 4: Unknown model ──
    print("\n" + "=" * 60)
    print("TEST 4: Unknown model — should not crash")
    print("=" * 60)
    report4 = auditor.execute(
        model="nonexistent-model-xyz",
        input_tokens=100, output_tokens=50,
        raw_response="Response from unknown model."
    )
    assert report4 is not None
    assert report4.estimated_cost == 0.0
    print(f"  ✓ Unknown model: cost=${report4.estimated_cost:.6f} (graceful)")

    # ── TEST 5: Monitor decorator safety ──
    print("\n" + "=" * 60)
    print("TEST 5: @monitor decorator — never breaks decorated function")
    print("=" * 60)

    @auditor.monitor(model="gpt-4o")
    def my_fn(prompt):
        return {"response": "Working!", "input_tokens": 10, "output_tokens": 5}

    result = my_fn("test")
    assert result["response"] == "Working!"
    print(f"  ✓ Decorator preserved return: {result['response']}")

    # Decorator with plain string
    @auditor.monitor(model="gpt-4o")
    def str_fn(prompt):
        return "Just a string"

    r = str_fn("test")
    assert r == "Just a string"
    print(f"  ✓ String return preserved: {r}")

    # ── TEST 6: Warnings collection ──
    print("\n" + "=" * 60)
    print("TEST 6: Warnings are collected, not raised")
    print("=" * 60)
    auditor.set_budget(0.001)
    auditor.set_alert_mode(True)
    auditor.guard_mode(confidence_threshold=99)

    report5 = auditor.execute(
        model="gpt-4", input_tokens=5000, output_tokens=5000,
        raw_response="x"  # expensive + low confidence
    )
    assert len(report5.warnings) > 0
    print(f"  ✓ Warnings collected ({len(report5.warnings)}):")
    for w in report5.warnings:
        print(f"    - {w}")

    # Cleanup
    auditor.clear_history()
    auditor.set_budget(None)
    auditor.set_alert_mode(False)
    auditor.guard_mode(confidence_threshold=0)

    print("\n✅ ALL ERROR SAFETY TESTS PASSED")

if __name__ == "__main__":
    main()
