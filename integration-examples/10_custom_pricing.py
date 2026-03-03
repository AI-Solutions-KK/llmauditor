"""
10_custom_pricing.py — set_pricing_table, override built-in, unpriced models
No API key needed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llmauditor import auditor

def main():
    auditor.clear_history()

    # ── TEST 1: Built-in pricing ──
    print("=" * 60)
    print("TEST 1: Built-in model pricing")
    print("=" * 60)
    report = auditor.execute(
        model="gpt-4o", input_tokens=1000, output_tokens=500,
        raw_response="Built-in pricing test response with sufficient content."
    )
    cost1 = report.estimated_cost
    assert cost1 > 0
    print(f"  gpt-4o (1K in / 500 out): ${cost1:.6f}")

    # ── TEST 2: Unpriced model → $0 ──
    print("\n" + "=" * 60)
    print("TEST 2: Unpriced model → cost = $0.00")
    print("=" * 60)
    report2 = auditor.execute(
        model="my-unknown-model", input_tokens=1000, output_tokens=500,
        raw_response="Unpriced model response — still gets full audit."
    )
    assert report2.estimated_cost == 0.0
    assert report2.total_tokens == 1500
    print(f"  my-unknown-model: ${report2.estimated_cost:.6f} (tokens: {report2.total_tokens})")

    # ── TEST 3: Custom pricing ──
    print("\n" + "=" * 60)
    print("TEST 3: Add custom pricing via set_pricing_table()")
    print("=" * 60)
    auditor.set_pricing_table({
        "llama-3.3-70b-versatile": {"input": 0.00059, "output": 0.00079},
        "mixtral-8x7b-32768": {"input": 0.00024, "output": 0.00024},
        "deepseek-v3": {"input": 0.00014, "output": 0.00028},
    })

    for model in ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "deepseek-v3"]:
        r = auditor.execute(
            model=model, input_tokens=1000, output_tokens=500,
            raw_response=f"Response from {model} with custom pricing."
        )
        assert r.estimated_cost > 0
        print(f"  {model}: ${r.estimated_cost:.6f}")

    # ── TEST 4: Override built-in pricing ──
    print("\n" + "=" * 60)
    print("TEST 4: Override built-in pricing")
    print("=" * 60)
    auditor.set_pricing_table({
        "gpt-4o": {"input": 0.0025, "output": 0.01}  # half price
    })
    report_override = auditor.execute(
        model="gpt-4o", input_tokens=1000, output_tokens=500,
        raw_response="Override pricing test."
    )
    cost_override = report_override.estimated_cost
    print(f"  gpt-4o original: ${cost1:.6f}")
    print(f"  gpt-4o override: ${cost_override:.6f}")
    assert cost_override != cost1  # different price

    auditor.clear_history()
    print("\n✅ ALL CUSTOM PRICING TESTS PASSED")

if __name__ == "__main__":
    main()
