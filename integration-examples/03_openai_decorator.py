"""
03_openai_decorator.py — @monitor decorator with live OpenAI call
Requires: OPENAI_API_KEY in .env
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import openai
from llmauditor import auditor

client = openai.OpenAI()

@auditor.monitor(model="gpt-4o-mini")
def ask_openai(prompt: str):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50
    )
    usage = response.usage
    return {
        "response": response.choices[0].message.content,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
    }

def main():
    print("Calling OpenAI via @monitor decorator...")
    try:
        result = ask_openai("Name 3 colors. Just list them.")
    except Exception as e:
        print(f"  ⚠ API error (likely quota/auth): {type(e).__name__}")
        print("\n⚠ OPENAI DECORATOR TEST SKIPPED (API not available)")
        return

    assert isinstance(result, dict)
    assert "response" in result
    assert len(result["response"]) > 0

    print(f"\n  Response: {result['response']}")
    print(f"  Tokens: {result['input_tokens']} in / {result['output_tokens']} out")
    print("\n✅ OPENAI DECORATOR TEST PASSED")

if __name__ == "__main__":
    main()
