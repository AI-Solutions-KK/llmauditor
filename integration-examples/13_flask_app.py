"""
13_flask_app.py — Flask /ask + /audit endpoints
Run: python integration-examples/13_flask_app.py
Then: curl http://localhost:5001/audit/budget
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, request, jsonify
from llmauditor import auditor

app = Flask(__name__)
auditor.set_budget(5.00)
auditor.set_alert_mode(True)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data["question"]

    # Simulated LLM response
    answer = f"Simulated Flask answer to: {question}"
    report = auditor.execute(
        model="gpt-4o",
        input_tokens=len(question.split()) * 2,
        output_tokens=len(answer.split()) * 2,
        raw_response=answer,
        input_text=question
    )

    return jsonify({
        "answer": answer,
        "cost": report.estimated_cost,
        "confidence": report.to_dict()["confidence_score"],
        "hallucination_risk": report.hallucination.risk_level if report.hallucination else "N/A"
    })

@app.route("/audit/budget", methods=["GET"])
def budget():
    return jsonify(auditor.get_budget_status())

def main():
    """Quick self-test without starting server."""
    print("=" * 60)
    print("Flask Integration Test (using test_client)")
    print("=" * 60)

    with app.test_client() as client:
        # Test /ask
        r = client.post("/ask", json={"question": "What is Flask?"})
        assert r.status_code == 200
        data = r.get_json()
        assert "answer" in data
        assert "cost" in data
        assert "hallucination_risk" in data
        print(f"  ✓ POST /ask → {data['answer'][:50]}...")
        print(f"    Cost: ${data['cost']:.6f}, Hallucination: {data['hallucination_risk']}")

        # Test /audit/budget
        r = client.get("/audit/budget")
        assert r.status_code == 200
        status = r.get_json()
        assert "budget_limit" in status
        print(f"  ✓ GET /audit/budget → ${status['cumulative_cost']:.6f} / ${status['budget_limit']}")

    auditor.clear_history()
    print("\n✅ FLASK INTEGRATION TEST PASSED")

if __name__ == "__main__":
    main()
