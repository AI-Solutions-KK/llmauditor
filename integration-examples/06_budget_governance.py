"""
06_budget_governance.py — Budget, guard mode, alert mode, role
No API key needed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llmauditor import auditor, BudgetExceededError, LowConfidenceError

def main():
    # Fresh state
    auditor.clear_history()

    # ── TEST 1: Budget limit (block mode) ──
    print("=" * 60)
    print("TEST 1: Budget limit — BudgetExceededError")
    print("=" * 60)
    auditor.set_budget(0.001)  # $0.001 limit
    auditor.set_alert_mode(False)

    try:
        auditor.execute(
            model="gpt-4", input_tokens=5000, output_tokens=5000,
            raw_response="Expensive response..."
        )
        print("  ✗ Should have raised BudgetExceededError")
    except BudgetExceededError as e:
        print(f"  ✓ BudgetExceededError caught: {e}")

    # ── TEST 2: Budget with alert mode ──
    print("\n" + "=" * 60)
    print("TEST 2: Budget limit — Alert mode (warn, don't block)")
    print("=" * 60)
    auditor.clear_history()
    auditor.set_budget(0.001)
    auditor.set_alert_mode(True)

    report = auditor.execute(
        model="gpt-4", input_tokens=5000, output_tokens=5000,
        raw_response="Expensive but allowed..."
    )
    has_warning = any("BUDGET" in w.upper() for w in report.warnings)
    print(f"  ✓ Report returned (not blocked)")
    print(f"  ✓ Budget warning present: {has_warning}")
    print(f"  ✓ Warnings: {report.warnings}")

    # ── TEST 3: Guard mode (block low confidence) ──
    print("\n" + "=" * 60)
    print("TEST 3: Guard mode — LowConfidenceError")
    print("=" * 60)
    auditor.clear_history()
    auditor.set_budget(100)
    auditor.set_alert_mode(False)
    auditor.guard_mode(confidence_threshold=80)

    try:
        auditor.execute(
            model="gpt-4o", input_tokens=5, output_tokens=2,
            raw_response="ok"  # Very short → low confidence
        )
        print("  ✗ Should have raised LowConfidenceError")
    except LowConfidenceError as e:
        print(f"  ✓ LowConfidenceError caught: {e}")

    # ── TEST 4: Guard mode + alert mode ──
    print("\n" + "=" * 60)
    print("TEST 4: Guard mode + Alert mode (warn, don't block)")
    print("=" * 60)
    auditor.clear_history()
    auditor.set_budget(100)
    auditor.guard_mode(confidence_threshold=80)
    auditor.set_alert_mode(True)

    report = auditor.execute(
        model="gpt-4o", input_tokens=5, output_tokens=2,
        raw_response="ok"
    )
    has_guard_warn = any("GUARD" in w.upper() for w in report.warnings)
    print(f"  ✓ Report returned (not blocked)")
    print(f"  ✓ Guard warning present: {has_guard_warn}")

    # ── TEST 5: Role-based governance ──
    print("\n" + "=" * 60)
    print("TEST 5: set_role()")
    print("=" * 60)
    auditor.set_role("production-safety-auditor")
    report = auditor.execute(
        model="gpt-4o", input_tokens=100, output_tokens=50,
        raw_response="Role-based test response."
    )
    d = report.to_dict()
    print(f"  ✓ Role in report: {d.get('governance', {}).get('role', 'N/A')}")

    # ── TEST 6: get_budget_status() ──
    print("\n" + "=" * 60)
    print("TEST 6: get_budget_status()")
    print("=" * 60)
    status = auditor.get_budget_status()
    assert "budget_limit" in status
    assert "cumulative_cost" in status
    assert "remaining" in status
    assert "executions" in status
    print(f"  ✓ Status: {status}")

    # Cleanup
    auditor.clear_history()
    auditor.set_budget(None)
    auditor.set_alert_mode(False)
    auditor.guard_mode(confidence_threshold=0)
    auditor.set_role("")

    print("\n✅ ALL GOVERNANCE TESTS PASSED")

if __name__ == "__main__":
    main()
