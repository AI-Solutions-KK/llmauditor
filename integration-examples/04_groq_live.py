"""
04_groq_live.py — Groq SDK + custom pricing table
Requires: GROQ_API_KEY in .env (skip gracefully if not set)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from llmauditor import auditor

def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("⚠ GROQ_API_KEY not set — running with simulated data")
        # Simulated test
        auditor.set_pricing_table({
            "llama-3.3-70b-versatile": {"input": 0.00059, "output": 0.00079}
        })
        report = auditor.execute(
            model="llama-3.3-70b-versatile",
            input_tokens=100,
            output_tokens=80,
            raw_response="Machine learning is a subset of AI that learns from data.",
            input_text="What is machine learning?"
        )
        report.display()
        assert report.estimated_cost > 0
        print(f"  Cost: ${report.estimated_cost:.6f}")
        print("\n✅ GROQ SIMULATED TEST PASSED")
        return

    from groq import Groq
    client = Groq(api_key=api_key)

    auditor.set_pricing_table({
        "llama-3.3-70b-versatile": {"input": 0.00059, "output": 0.00079}
    })

    print("Calling Groq llama-3.3-70b-versatile...")
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "What is Python? One sentence."}],
        max_tokens=50
    )

    choice = response.choices[0].message.content
    usage = response.usage

    report = auditor.execute(
        model="llama-3.3-70b-versatile",
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        raw_response=choice,
        input_text="What is Python? One sentence."
    )
    report.display()

    assert report.estimated_cost > 0
    print(f"\n  Answer: {choice}")
    print(f"  Cost: ${report.estimated_cost:.6f}")
    print("\n✅ GROQ LIVE TEST PASSED")

if __name__ == "__main__":
    main()
