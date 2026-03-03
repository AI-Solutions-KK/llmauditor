"""
14_streamlit_app.py — Streamlit chat with audit sidebar
Run: streamlit run integration-examples/14_streamlit_app.py

Self-test (no Streamlit server needed):
    python integration-examples/14_streamlit_app.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Self-test mode (no Streamlit required) ──
if __name__ == "__main__" and "streamlit" not in sys.modules.get("__spec__", "").__class__.__name__:
    from llmauditor import auditor

    print("=" * 60)
    print("Streamlit Integration Test (self-test, no server)")
    print("=" * 60)

    auditor.clear_history()
    auditor.set_budget(1.00)

    # Simulate what the Streamlit app would do
    for i, q in enumerate(["What is AI?", "Explain ML.", "Define NLP."]):
        answer = f"Simulated answer {i+1} for: {q}"
        report = auditor.execute(
            model="gpt-4o",
            input_tokens=len(q.split()) * 3,
            output_tokens=len(answer.split()) * 3,
            raw_response=answer,
            input_text=q
        )
        hal_risk = report.hallucination.risk_level if report.hallucination else "N/A"
        print(f"  Q{i+1}: Cost=${report.estimated_cost:.6f}, "
              f"Tokens={report.total_tokens}, Hal={hal_risk}")

    status = auditor.get_budget_status()
    pct = status["cumulative_cost"] / status["budget_limit"] * 100
    print(f"\n  Budget: ${status['cumulative_cost']:.4f} / ${status['budget_limit']:.2f} ({pct:.1f}%)")

    auditor.clear_history()
    print("\n✅ STREAMLIT SELF-TEST PASSED")
    sys.exit(0)

# ── Streamlit app (runs with `streamlit run`) ──
import streamlit as st
from llmauditor import auditor

auditor.set_budget(1.00)

st.title("AI Chat with LLMAuditor")

prompt = st.text_input("Ask a question:")
if prompt:
    answer = f"Simulated answer to: {prompt}"
    in_tok = len(prompt.split()) * 3
    out_tok = len(answer.split()) * 3

    report = auditor.execute(
        model="gpt-4o",
        input_tokens=in_tok,
        output_tokens=out_tok,
        raw_response=answer,
        input_text=prompt
    )

    st.write(answer)

    with st.sidebar:
        st.metric("Cost", f"${report.estimated_cost:.6f}")
        st.metric("Tokens", report.total_tokens)
        if report.hallucination:
            st.metric("Hallucination Risk", report.hallucination.risk_level)

        status = auditor.get_budget_status()
        st.progress(min(status["cumulative_cost"] / status["budget_limit"], 1.0))
        st.caption(f"${status['cumulative_cost']:.4f} / ${status['budget_limit']:.2f}")
