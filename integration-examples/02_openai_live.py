"""
02_openai_live.py — Live OpenAI SDK call + auditor.execute()
Requires: OPENAI_API_KEY in .env
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import openai
from llmauditor import auditor

def main():
    client = openai.OpenAI()
    prompt = "What is the capital of Japan? Reply in one sentence."

    print("Calling OpenAI gpt-4o-mini...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50
        )
    except Exception as e:
        print(f"  ⚠ API error (likely quota/auth): {type(e).__name__}: {e}")
        print("\n⚠ OPENAI LIVE TEST SKIPPED (API not available)")
        return

    choice = response.choices[0].message.content
    usage = response.usage

    report = auditor.execute(
        model="gpt-4o-mini",
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        raw_response=choice,
        input_text=prompt
    )
    report.display()

    assert report.model_name == "gpt-4o-mini"
    assert report.total_tokens > 0
    assert report.estimated_cost >= 0
    assert "Tokyo" in choice or "tokyo" in choice.lower()

    print(f"\n  Answer: {choice}")
    print(f"  Cost: ${report.estimated_cost:.6f}")
    print(f"  Tokens: {report.total_tokens}")
    print("\n✅ OPENAI LIVE TEST PASSED")

if __name__ == "__main__":
    main()
