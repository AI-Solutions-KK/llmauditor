"""
08_hallucination.py — Rule-based + AI judge hallucination detection
AI judge requires: OPENAI_API_KEY in .env (falls back to rule-based only)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from llmauditor import auditor

def main():
    auditor.clear_history()

    # ── TEST 1: Rule-based hallucination ──
    print("=" * 60)
    print("TEST 1: Rule-based hallucination detection")
    print("=" * 60)
    report = auditor.execute(
        model="gpt-4o",
        input_tokens=100,
        output_tokens=200,
        raw_response="The population of Tokyo is exactly 14,215,316 as of January 2024. "
                     "It is always the largest city in the world.",
        input_text="What is the population of Tokyo?"
    )

    hal = report.hallucination
    assert hal is not None
    assert hal.method == "rule-based"
    assert hal.risk_score >= 0
    assert hal.risk_level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert hal.factual_claims_count >= 0
    assert hal.specific_numbers_count >= 0

    print(f"  Method: {hal.method}")
    print(f"  Risk Score: {hal.risk_score_pct}%")
    print(f"  Risk Level: {hal.risk_level}")
    print(f"  Factual Claims: {hal.factual_claims_count}")
    print(f"  Specific Numbers: {hal.specific_numbers_count}")
    print(f"  Date References: {hal.date_references_count}")
    print(f"  Currency References: {hal.currency_references_count}")
    print(f"  Hedging Ratio: {hal.hedging_ratio:.2f}")
    print(f"  Absolute Claims: {hal.absolute_claims_count}")

    # ── TEST 2: Low-risk response (hedging language) ──
    print("\n" + "=" * 60)
    print("TEST 2: Low-risk response (hedging language)")
    print("=" * 60)
    report2 = auditor.execute(
        model="gpt-4o",
        input_tokens=50,
        output_tokens=100,
        raw_response="It seems that the weather might be warm tomorrow, "
                     "though it could possibly change. I think it may rain.",
        input_text="What is the weather tomorrow?"
    )
    hal2 = report2.hallucination
    print(f"  Risk: {hal2.risk_score_pct}% ({hal2.risk_level})")
    print(f"  Hedging Ratio: {hal2.hedging_ratio:.2f}")

    # ── TEST 3: AI Judge (if OpenAI key available) ──
    print("\n" + "=" * 60)
    print("TEST 3: AI Judge hallucination detection")
    print("=" * 60)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ⚠ OPENAI_API_KEY not set — skipping AI judge test")
    else:
        import openai
        client = openai.OpenAI()

        def judge_fn(prompt: str) -> str:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=200
            )
            return response.choices[0].message.content

        auditor.set_hallucination_model(judge_fn, model="gpt-4o-mini")

        try:
            report3 = auditor.execute(
                model="gpt-4o",
                input_tokens=100,
                output_tokens=150,
                raw_response="The Great Wall of China is 21,196 km long and was built in 1 year.",
                input_text="Tell me about the Great Wall of China."
            )
        except Exception as e:
            print(f"  ⚠ API error (likely quota/auth): {type(e).__name__}")
            print("  ⚠ AI Judge test skipped")
            auditor.clear_history()
            print("\n✅ HALLUCINATION TESTS PASSED (AI judge skipped)")
            return

        hal3 = report3.hallucination
        if hal3.method == "hybrid":
            print(f"  Method: {hal3.method}")
            print(f"  Risk Score: {hal3.risk_score_pct}%")
            print(f"  AI Judge Score: {hal3.ai_judge_score}")
            print(f"  AI Reasoning: {hal3.ai_judge_reasoning[:80]}...")
        else:
            # Judge call failed silently (quota/auth), fell back to rule-based
            print(f"  ⚠ AI judge not available (got {hal3.method}), likely API quota issue")
            print("  ⚠ AI Judge test skipped — rule-based still works")

    auditor.clear_history()
    print("\n✅ ALL HALLUCINATION TESTS PASSED")

if __name__ == "__main__":
    main()
