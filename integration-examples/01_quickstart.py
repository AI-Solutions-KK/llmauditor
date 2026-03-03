"""
01_quickstart.py — Core API: execute, observe, monitor, display, to_dict
No API key needed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llmauditor import auditor

def main():
    print("=" * 60)
    print("TEST 1: execute()")
    print("=" * 60)
    report = auditor.execute(
        model="gpt-4o",
        input_tokens=150,
        output_tokens=80,
        raw_response="The capital of France is Paris.",
        input_text="What is the capital of France?"
    )
    report.display()
    assert report.model_name == "gpt-4o"
    assert report.total_tokens == 230
    assert report.estimated_cost > 0
    print(f"  ✓ Cost: ${report.estimated_cost:.6f}, Tokens: {report.total_tokens}")

    print("\n" + "=" * 60)
    print("TEST 2: observe()")
    print("=" * 60)
    report2 = auditor.observe(
        model="claude-3.5-sonnet",
        input_text="Explain quantum computing.",
        output_text="Quantum computing uses qubits that can be 0 and 1 simultaneously.",
        input_tokens=50,
        output_tokens=100,
        execution_time=1.5
    )
    assert report2.model_name == "claude-3.5-sonnet"
    print(f"  ✓ Cost: ${report2.estimated_cost:.6f}")

    print("\n" + "=" * 60)
    print("TEST 3: monitor() decorator")
    print("=" * 60)

    @auditor.monitor(model="gpt-4o-mini")
    def mock_llm(prompt):
        return {
            "response": "The answer is 42.",
            "input_tokens": 30,
            "output_tokens": 10
        }

    result = mock_llm("What is the meaning of life?")
    assert isinstance(result, dict)
    assert result["response"] == "The answer is 42."
    print(f"  ✓ Decorator returned: {result['response']}")

    print("\n" + "=" * 60)
    print("TEST 4: monitor() with plain string")
    print("=" * 60)

    @auditor.monitor(model="gpt-4o-mini")
    def simple_fn(prompt):
        return "Hello, world!"

    r = simple_fn("Say hello")
    assert r == "Hello, world!"
    print(f"  ✓ String return: {r}")

    print("\n" + "=" * 60)
    print("TEST 5: to_dict()")
    print("=" * 60)
    d = report.to_dict()
    assert "execution_id" in d
    assert "estimated_cost" in d
    assert "confidence_score" in d
    assert "hallucination" in d
    assert "risk_level" in d
    print(f"  ✓ Dict keys: {list(d.keys())}")

    print("\n" + "=" * 60)
    print("TEST 6: history()")
    print("=" * 60)
    h = auditor.history()
    assert len(h) >= 4  # at least 4 executions above
    print(f"  ✓ History length: {len(h)}")

    print("\n✅ ALL QUICKSTART TESTS PASSED")

if __name__ == "__main__":
    main()
