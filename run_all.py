"""
run_all.py — Run all offline integration tests in sequence.
Live API tests (02, 03, 04, 05, 08) are skipped unless --live flag is passed.
"""
import sys, os, time, importlib.util

# Ensure UTF-8 output on Windows (for checkmark/emoji characters)
if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "integration-examples")

# Offline tests (no API key needed)
OFFLINE = [
    "01_quickstart.py",
    "06_budget_governance.py",
    "07_evaluation_certification.py",
    "09_export_reports.py",
    "10_custom_pricing.py",
    "11_multi_agent.py",
    "12_fastapi_app.py",
    "13_flask_app.py",
    "14_streamlit_app.py",
    "15_error_safety.py",
    "16_full_endtoend.py",
]

# Live API tests (require API keys in .env)
LIVE = [
    "02_openai_live.py",
    "03_openai_decorator.py",
    "04_groq_live.py",
    "05_langchain_openai.py",
    "08_hallucination.py",
]


def run_example(filepath: str) -> bool:
    name = os.path.basename(filepath)
    print(f"\n{'━' * 70}")
    print(f"  Running: {name}")
    print(f"{'━' * 70}")

    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), filepath)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        if hasattr(mod, "main"):
            mod.main()
        return True
    except Exception as e:
        print(f"\n  ❌ FAILED: {name} — {type(e).__name__}: {e}")
        return False


def main():
    live_mode = "--live" in sys.argv
    tests = OFFLINE + (LIVE if live_mode else [])

    print("=" * 70)
    print("  LLMAuditor — Integration Test Suite")
    print(f"  Mode: {'OFFLINE + LIVE API' if live_mode else 'OFFLINE ONLY'}")
    print(f"  Tests: {len(tests)}")
    print("=" * 70)

    passed = 0
    failed = 0
    failures = []
    start = time.time()

    for filename in tests:
        filepath = os.path.join(EXAMPLES_DIR, filename)
        if not os.path.exists(filepath):
            print(f"\n  ⚠ File not found: {filename}")
            failed += 1
            failures.append(filename)
            continue

        ok = run_example(filepath)
        if ok:
            passed += 1
        else:
            failed += 1
            failures.append(filename)

    elapsed = time.time() - start

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {passed} passed, {failed} failed ({elapsed:.1f}s)")
    if failures:
        print(f"  Failed: {', '.join(failures)}")
    print(f"{'=' * 70}")

    if not live_mode:
        print(f"\n  Tip: Run with --live to include API tests:")
        print(f"  python run_all.py --live")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
