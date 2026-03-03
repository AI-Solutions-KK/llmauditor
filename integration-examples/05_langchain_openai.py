"""
05_langchain_openai.py — LangChain ChatOpenAI + auditor
Requires: OPENAI_API_KEY in .env
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from llmauditor import auditor

def main():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=50)
    question = "What is 2 + 2? Just the number."

    print("Calling LangChain ChatOpenAI (gpt-4o-mini)...")
    try:
        response = llm.invoke([HumanMessage(content=question)])
    except Exception as e:
        print(f"  ⚠ API error (likely quota/auth): {type(e).__name__}")
        print("\n⚠ LANGCHAIN OPENAI TEST SKIPPED (API not available)")
        return

    usage = response.usage_metadata or {}

    report = auditor.execute(
        model="gpt-4o-mini",
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        raw_response=response.content,
        input_text=question
    )
    report.display()

    assert report.total_tokens > 0
    assert "4" in response.content

    print(f"\n  Answer: {response.content}")
    print(f"  Cost: ${report.estimated_cost:.6f}")
    print("\n✅ LANGCHAIN OPENAI TEST PASSED")

if __name__ == "__main__":
    main()
